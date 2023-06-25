from torch.utils.data import DataLoader
import numpy as np
from config import Config
from VIP_data import MyDataset
from Model.Decoder.panoptic_decoder import DecoderArch
import torch
from Model.post_processing import post_processing,depth_prediction
import matplotlib.pyplot as plt
#from torchvision.transforms import ToPILImage
from PIL import Image

from Model.post_processing import colorize_visual 
from Model.post_processing import Metric

# same parameters
label_divisor=1000 # used to convert panoptic id = semantic id * label_divisor + instance_id
stuff_area = 2048 # same in bowen code. ??not sure is it right or?
ignore_label=32 #  get from bowen is 255, now is 32 unlabelled



# load model params and call postprocessing
def use():
    # configure
    cfg = Config()
    device = None
    if cfg.enable_cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    

    # load data from dataset (a pair) in sequence ///  IF use in reality, do not have dataloader but input separate frames from buffer.
    dataset_val = MyDataset(cfg.gt_dir_DVPS,split='val_1024_512')
    dataloader_val = DataLoader(dataset_val, batch_size=cfg.batch_size_test, shuffle=False, num_workers=4,drop_last=True)

    # initialize network
    model = DecoderArch()
    model.eval()
    model = model.to(device)

    # load model parameters
    save_path = 'VIP_model.pth'
    print('Loading model from {}...'.format(save_path))
    model.load_state_dict(torch.load('VIP_model.pth'))
    print('Parameters loaded')

    # prepare save data
    semantic_figures = []
    sem_confidence= []

    heatmaps = []
    depth_figures=[]
    offsets_maps=[]
    
    instance_figures = []
    instance_centers=[]
    panoptic_figures = []
    panoptic_centers=[]

    mIoU_list = []
    absRel_list =[]

    # Run the model to get output
    with torch.no_grad():
        for batch_idx, (images_pair, class_labels_pair,_,depth_labels_pair) in enumerate(dataloader_val): # get one batch each time
            H = images_pair.size(-2)
            W = images_pair.size(-1)
            images_pair = [sample for sample in images_pair if torch.sum(sample)!=0]  # ignore specific figures from getitem!!!
            images_pair= torch.stack(images_pair,dim=0)
            class_labels_pair = [sample for sample in class_labels_pair if torch.sum(sample)!=-2*H*W]
            class_labels_pair= torch.stack(class_labels_pair,dim=0)
            depth_labels_pair = [sample for sample in depth_labels_pair if torch.sum(sample)!=-2*H*W]
            depth_labels_pair= torch.stack(depth_labels_pair,dim=0)  
            
            # convert from pair to single
            images_t = images_pair[:,0,:,:,:].to(device=device, dtype=torch.float32)
            images_tp1 = images_pair[:,1,:,:,:].to(device=device, dtype=torch.float32)
            class_labels = class_labels_pair[:,0,:,:].to(device=device, dtype=torch.long) 
            depth_labels = depth_labels_pair[:,0,:,:].to(device=device, dtype=torch.float32)   # torch.Size([N, 512, 1024])

            ## feed data to model
            out_depth, out_class,out_ins_cen_pred,out_ins_cen_reg, out_ins_next = model(featuresT0=images_t,featuresT1=images_tp1)
            
            #normalize
            out_ins_cen_pred=(out_ins_cen_pred-torch.min(out_ins_cen_pred))/(torch.max(out_ins_cen_pred)-torch.min(out_ins_cen_pred))

            ### post processing.  # input need single image, so use loop  
            # semantic segmentation
            for i in range(out_class.size(dim=0)): # let input be [C, H, W]
                semantic,confidence =post_processing.get_semantic_segmentation(out_class[i].unsqueeze(dim=0),threshold=0.1)  # output is [ H, W] pixel is class label if pixel maximum score<threshold, view it as unlabel.32, 
                semantic_figures.append(semantic) # in list, each element is [1, H, W] #[0,..18,32]
                sem_confidence.append(confidence) # can be used for IoU inference.(considering the operation of low score pixel assignment as unlabel)
            # depth prediction
            for i in range(out_depth.size(dim=0)):
                depth=depth_prediction.depth_act(out_depth[i].unsqueeze(dim=0))
                depth_figures.append(depth) # in list, each element is [1, H, W]
            # center prediction, heat map
            for i in range(out_ins_cen_pred.size(dim=0)):  # out_ins_cen_pred = ([5, 1, 1024, 2048])
                heatmaps.append(out_ins_cen_pred[i].unsqueeze(dim=0)) # in list, each element is [1, H, W]
            # center regression,  offset map [2] 0->y 1->x
            for i in range(out_ins_cen_reg.size(dim=0)):
                offsets_maps.append(out_ins_cen_reg[i].unsqueeze(dim=0)) # in list, each element is [2,H,W]

            # ??? next frame center regression ??? wait for stiching.!>??????

            # instance segmentation based above data.
            for idx in range(batch_idx*out_class.size(dim=0), (batch_idx+1)*out_class.size(dim=0) ): # swift to different batch. use out_class to get batch value, to avoid batch size fluctuate (throw away wrong pairs )
                ins_figure,center=post_processing.get_instance_segmentation(sem_seg=semantic_figures[idx],ctr_hmp=heatmaps[idx],
                                                                            offsets=offsets_maps[idx],thing_list=cfg.cityscape_thing_list,
                                                                            top_k=200,nms_kernel=7,threshold=0.1)  # input offset is [1,2,H,W]  in dim=1 , 0=> offset y, 1=> offset x
                # ins_figure: each pixel means which instance it belongs to (stuff =0)
                instance_figures.append(ins_figure)
                instance_centers.append(center)

            # panoptic segmentation/ each pixel means 1000*class_id+ins_id / this method do not include next frame! 
            for idx in range(batch_idx*out_class.size(dim=0), (batch_idx+ 1)*out_class.size(dim=0) ): # cause need figures after previous append!
                pan_figure,center=post_processing.get_panoptic_segmentation(sem=semantic_figures[idx],ctr_hmp=heatmaps[idx],
                                                                            offsets=offsets_maps[idx],thing_list=cfg.cityscape_thing_list,
                                                                            label_divisor=label_divisor,stuff_area=stuff_area,
                                                                            void_label=label_divisor*ignore_label,top_k=200,
                                                                            nms_kernel=7,threshold=0.1)
                panoptic_figures.append(pan_figure)
                panoptic_centers.append(center)


            ## visualization  从存好的数据中依次取值来播放，按batch？暂停？
            
            #  image, semantic, instance, panoptic, depth (+point cloud)
            batch_size = images_t.size(dim=0) #N
                         # images_t : torch.Size([5, 3, 512, 1024])
            for i in range(batch_size): # go through batch
                fig, axs = plt.subplots(2,3, figsize=(20, 10))#
                images_t_rgb=np.transpose(images_t[i,:,:,:].numpy(), (1,2,0))/255  #[h,w,c] nput data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
                axs[0,0].imshow(np.asarray(images_t_rgb))
                axs[0,0].set_title('Image')
                axs[0,0].axis('off')

                # color map for semantic_figure.!!!调用convert函数
                sem_rgb=colorize_visual.sem_color_convert(semantic_figures[i+batch_size*batch_idx]) #input: get from list each element [1, H, W]
                axs[0,1].imshow(sem_rgb)  # output is numpy
                axs[0,1].set_title('Semantic Segmentation')
                axs[0,1].axis('off')
                

                # cmap = plt.get_cmap('tab10')
                # ins  = instance_figures[i+batch_size*batch_idx].numpy().squeeze()
                # axs[0,2].imshow(ins ,cmap=cmap) # from list, ith  ins_figure [1,H,W] # tensor need convert
                # axs[0,2].set_title('Instance Segmentation')
                # axs[0,2].axis('off')
                #cmap = plt.get_cmap('tab10')
                offsets_t_y  = (offsets_maps[i+batch_size*batch_idx].numpy().squeeze())[0,:,:]
                print(offsets_maps[i+batch_size*batch_idx].shape)
                print(offsets_t_y.shape)
                axs[0,2].imshow(offsets_t_y) # from list, ith  ins_figure [1,H,W] # tensor need convert
                axs[0,2].set_title('Offset map y')
                axs[0,2].axis('off')
                
                # cmap = plt.get_cmap('tab10')
                # axs[2].imshow(offsets_maps[i+batch_size*batch_idx][0,0,:,:].numpy().squeeze(),cmap=cmap)
                # axs[2].set_title('Heatmap')

                pan_rgb= colorize_visual.pan_color_convert(panoptic_figures[i+batch_size*batch_idx],label_divisor=1000) # return Image type
                #print(f'pan rgb unique{np.unique(pan_rgb)}')
                axs[1,0].imshow(pan_rgb)
                axs[1,0].set_title('Panoptic segmentation')
                axs[1,0].axis('off')
                
               
                dep=depth_figures[i+batch_size*batch_idx].numpy().squeeze()
                #normalize to be suitable with imshow
                normalized_dep = (dep-np.min(dep))/(np.max(dep)-np.min(dep)) # float => (0,1)
                axs[1,1].imshow(normalized_dep,cmap='gray') # only channel tensor will be convet to gray figure.
                axs[1,1].set_title('Depth prediction')
                axs[1,1].axis('off')

                heatmap=heatmaps[i+batch_size*batch_idx].numpy().squeeze()
                axs[1,2].imshow(heatmap)
                axs[1,2].set_title('Center prediction heatmap')
                axs[1,2].axis('off')
               
                #print(f'depth max is {np.max(dep)}')
                # point cloud 图？？

                plt.tight_layout()
                plt.show() # for one image, we get sem,ins,pan,dep figures in a row.
                user_input = input("type any to continue or input 'q' to exist:")
                if user_input == 'q':
                    break
           
    # 
                    

    ## define a function for evluate? or seperately write and call them here.
        #evaluate
        # IoU for semantic  # input is a batch images # we can say cause they are consecutive frames/
        mIoU=Metric.compute_IoU(sem=out_class,target=class_labels)
        mIoU_list.append(mIoU)
        # absRel for depth 
        absRel = Metric.abs_rel_error(depth_pred=out_depth,depth_target=depth_labels)
        absRel_list.append(absRel)
        # VPQ for panoptic?

        # DVPQ for depth + panoptic?
    
    ## visualize metrics
    # mIoU hist map  --- semantic
    bin_width =0.2
    bins = np.arange(min(mIoU_list),max(mIoU_list)+bin_width,bin_width)
    plt.hist(mIoU_list,bins=bins)
    plt.xlabel('mIoU value [batch]')
    plt.ylabel('Count')
    plt.title('Histogram for semantic mIoU (average in a batch)')
    plt.show()
    # absRel for depth
    plt.hist(absRel_list,bins=10)
    plt.xlabel('absRel value [batch]')
    plt.ylabel('Count')
    plt.title('Histogram for depth absRel error (average in a batch)')
    plt.show()





### below just show some orginal maps to check while coding.
def Convert_ins2heatmap(instance_labels,sigma=8,threshold = 0.05): # input is tensor  like torch.Size([32, 512, 1024])
        # 8 pixels, as paper panoptic deeplab shown 

        # create a same size heatmap tensor
        heatmap_center = torch.zeros_like(instance_labels,dtype=torch.float,device=instance_labels.device)
        points_x = torch.arange(start=0,end=instance_labels.size(dim=-1)) # [0,..1023] tensor[1024,]
        points_x = points_x.repeat(instance_labels.size(dim=-2),1)
        print(f'points_x shape {points_x.size()}')
        points_y = torch.arange(start=0,end=instance_labels.size(dim=-2)).unsqueeze(1) # [0,1,2...511] tensor[512,1]
        points_y = points_y.repeat(1,instance_labels.size(dim=-1))
        print(f'points_y shape {points_y.size()}')

        # calculate instance center position
        for i in range(instance_labels.size(dim=0)): # repeat N times for each image in this batch
            # get unique ID value
            instance_IDs = torch.unique(instance_labels[i])
            for id in instance_IDs: # in each image, go over all instance
                coordinates = torch.nonzero(instance_labels[i]==id) # [[y,x],[y,x],..]
                if coordinates.numel()>0:
                    center = torch.mean(coordinates.float(),dim=0) # 跨行求平均，即最后压缩到只有一行。列上元素加起来平均
                    # 2D gaussian----based on x,y calculate the gaussian value for each
                    #guassian= torch.exp(-0.5*(((coordinates[:,0]-center[0])/sigma)**2+((coordinates[:,1]-center[1])/sigma)**2)) # get a vector with the value
                    gaussian= torch.exp(-0.5*(((points_y-center[0])/sigma)**2+((points_x-center[1])/sigma)**2)) # get a vector with the value
                    
                    # Threshold --filter
                    heatmap =torch.where(gaussian<threshold,torch.tensor(0),gaussian)
                    
                    # # fill the value to cooresponding x,y 
                    # heatmap_center[i,points_y,points_x] = guassian 
                
                heatmap_center[i] = torch.maximum(heatmap,heatmap_center[i])
    
        return heatmap_center  # reasult is [N,H,W] tensor value is probability of being a center.


### Instance center offset -----target offset map ,for offset loss calculate! 
def generate_offset_map(target_map):# target_map means "instance_labels" ground truth!
    # Generate an offset map of the nearest center position corresponding to each pixel whose ID is not equal to 0
    batch_size, height, width = target_map.size()
    
    # empty offset map [batch_size, 2, height, width]
    offset_map = torch.zeros(batch_size, 2, height, width)
    
    # Find the nearest center position corresponding to each pixel whose ID is not equal to 0
    for i in range(batch_size):
        # Remove the background ID and keep the instance ID
        instance_ids = torch.unique(target_map[i]) # in each figure, have different num of instances

        if instance_ids.size()==1:  ##  to avoid figure witouht instance id, I mean only have 0. if have id=other than o, at least have [0,x] len>1
            offset = torch.zeros_like(target_map[i]) 

        instance_ids = instance_ids[instance_ids != 0] # except 0, [0,1,2,...,32] ->[0,1,1,1,1...]   then get [1,2,3,...] ids without 0
        
        # Calculate the center position corresponding to each instance ID
        centers = calculate_centers(target_map[i], instance_ids)
        
        # Calculate the offset of the nearest center position corresponding to each pixel
        offset = calculate_offset(target_map[i], centers)
        
        # save
        offset_map[i] = offset #[2,H,W]  # cause initial is tensor so output is still tensor! 
    
    return offset_map # [B,2,H,W]

def calculate_centers(target, instance_ids): # instance_ids = [1,2...] unique id sequence without 0
    # Calculate the center position corresponding to each instance ID
    centers = torch.zeros(len(instance_ids),2, dtype=torch.float32, device=target.device)  # num instance = num center, tensor
    for idx, instance_id in enumerate(instance_ids): # for each Instance
        # Find the pixel coordinates corresponding to the instance ID
        coordinates = torch.nonzero(target == instance_id, as_tuple=False) # N*2  [[h1,w1];[h2,w2];..] == [[y1,x1];..]
        
        if coordinates.numel() > 0:
            # Calculate the center position corresponding to the instance ID
            center = torch.mean(coordinates.float(), dim=0) #=>p[mean_x,mean_y]
            centers[idx] = center
    
    return centers

def calculate_offset(target, centers):  # input is single one figure in a batch
    # Calculate the offset of the nearest center position corresponding to each pixel
    height, width = target.size()
    
    # Find the index of the nearest center position corresponding to each pixel
    indices = torch.nonzero(target, as_tuple=False) # all the indices !=0 , in one target figure [[y,x].[y,x],..]
    
    offset = torch.zeros(2, height, width, device=target.device)
    
    if indices.numel() > 0:
        # Calculate the offset of the nearest center position corresponding to each pixel
        distances = torch.cdist(indices.float().unsqueeze(0), centers.unsqueeze(1)) #(1,n,2) (k,1,2) # can use each point in indices to calculate distacen wrt. each center points!
        nearest_center_idx = torch.argmin(distances, dim=2) # then return the mini index, indicate which center is close to me. each row/col means point. 
        
        for i, idx in enumerate(nearest_center_idx[0]): # [[c1,c3,c2,c1,c3,c2,..]] reduce dimension
            pixel = indices[i] # i means which row/col, means which point in indices  #[y1,x1]
            center = centers[idx] # when i=1, c1. #[c_y,c_x]
            
            # offset
            dy = center[0] - pixel[0]  # 我改了下dy 和dx的对应。【0】-> height 
            dx = center[1] - pixel[1]
            
            offset[0, pixel[0], pixel[1]] = dy
            offset[1, pixel[0], pixel[1]] = dx    # 0-> y , 1 -> x！
    
    return offset     # instance_labels torch.Size([2, 1024, 2048])  # 因为使用的输入不是一个batch而是单张，所以output是[2,H,W]



device='cpu'
# show heatmap and offset target figure:
def showmap():
    cfg = Config()
    device = None
    if cfg.enable_cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # initialize network
    model = DecoderArch()
    model.eval()
    model = model.to(device)

    # load model parameters
    # save_path = 'VIP_model.pth'
    # print('Loading model from {}...'.format(save_path))
    # model.load_state_dict(torch.load('VIP_model.pth'))
    # print('Parameters loaded')

    cfg = Config()
    dataset_val = MyDataset(cfg.gt_dir_DVPS,split='val_1024_512')
    dataloader_val = DataLoader(dataset_val, batch_size=cfg.batch_size_test, shuffle=False, num_workers=4,drop_last=True)
    for batch_idx, (images_pair, class_labels_pair,instance_labels_pair,depth_labels_pair) in enumerate(dataloader_val):
        H = images_pair.size(-2)
        W = images_pair.size(-1)
        # throw away pair indicate , image is 0, others is array full of -1
        images_pair = [sample for sample in images_pair if torch.sum(sample)!=0]  # ignore specific figures from getitem!!! RGB so 3
        images_pair= torch.stack(images_pair,dim=0)
        class_labels_pair = [sample for sample in class_labels_pair if torch.sum(sample)!=-2*H*W]
        class_labels_pair= torch.stack(class_labels_pair,dim=0)
        instance_labels_pair = [sample for sample in instance_labels_pair if torch.sum(sample)!=-2*H*W]
        instance_labels_pair =  torch.stack(instance_labels_pair,dim=0)
        depth_labels_pair = [sample for sample in depth_labels_pair if torch.sum(sample)!=-2*H*W]
        depth_labels_pair= torch.stack(depth_labels_pair,dim=0)

        images_t = images_pair[:,0,:,:,:].to(device=device, dtype=torch.float32)
        images_tp1 = images_pair[:,1,:,:,:].to(device=device, dtype=torch.float32)
        depth_labels = depth_labels_pair[:,0,:,:].to(device=device, dtype=torch.float32)  # [N,H,W]
        # convert instacne label figures inot offset map ground truth
        offsetmap_t = generate_offset_map(instance_labels_pair[:,0,:,:])
        offsetmap_tp1 = generate_offset_map(instance_labels_pair[:,1,:,:])
        #print(f'less than 0 number {(depth_labels<0).sum()}')
            
        class_labels = class_labels_pair[:,0,:,:].to(device=device, dtype=torch.long)
        #out_depth, out_class,out_ins_cen_pred,out_ins_cen_reg, out_ins_next = model(featuresT0=images_t,featuresT1=images_tp1)
        # heatmap_center=Convert_ins2heatmap( instance_labels_pair[:,0,:,:]) # instance_labels torch.Size([32, 1024, 2048]) // each batch
        # cmap = plt.get_cmap('tab10')

        # for i in range(10):
        #     plt.figure()
        #     plt.imshow(heatmap_center[i].numpy())
        #     plt.show()
        # for i in range(10):
        #     plt.figure()
        #     plt.imshow(torch.argmax(out_class[i,:,:,:],dim=-3).numpy())
        #     plt.show()
        # for i in range(10):
        #     plt.figure()
        #     plt.imshow(class_labels[i].numpy())
        #     plt.show()
        # for i in range(10):
        #     plt.figure()
        #     plt.imshow(depth_labels[i].numpy())
        #     plt.show()
        for i in range(10):
            plt.figure()
            plt.imshow(offsetmap_t[i,0,:,:].numpy()) # y --height figure
            plt.show()
        
        



def metric():
    # configure
    cfg = Config()
    device = None
    if cfg.enable_cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    

    # load data from dataset (a pair) in sequence ///  IF use in reality, do not have dataloader but input separate frames from buffer.
    dataset_val = MyDataset(cfg.gt_dir_DVPS,split='val_1024_512')
    dataloader_val = DataLoader(dataset_val, batch_size=cfg.batch_size_test, shuffle=False, num_workers=4,drop_last=True)

    # initialize network
    model = DecoderArch()
    model.eval()
    model = model.to(device)

    # load model parameters
    save_path = 'VIP_model.pth'
    print('Loading model from {}...'.format(save_path))
    model.load_state_dict(torch.load('VIP_model.pth'))
    print('Parameters loaded')

    # prepare save data
    semantic_figures = []
    sem_confidence= []

    heatmaps = []
    depth_figures=[]
    offsets_maps=[]
    
    instance_figures = []
    instance_centers=[]
    panoptic_figures = []
    panoptic_centers=[]

    mIoU_list = []

    # Run the model to get output
    with torch.no_grad():
        for batch_idx, (images_pair, class_labels_pair,_,depth_labels_pair) in enumerate(dataloader_val): # get one batch each time
            H = images_pair.size(-2)
            W = images_pair.size(-1)
            images_pair = [sample for sample in images_pair if torch.sum(sample)!=0]  # ignore specific figures from getitem!!!
            images_pair= torch.stack(images_pair,dim=0)
            class_labels_pair = [sample for sample in class_labels_pair if torch.sum(sample)!=-2*H*W]
            class_labels_pair= torch.stack(class_labels_pair,dim=0)
            depth_labels_pair = [sample for sample in depth_labels_pair if torch.sum(sample)!=-2*H*W]
            depth_labels_pair= torch.stack(depth_labels_pair,dim=0)  
            
            # convert from pair to single
            images_t = images_pair[:,0,:,:,:].to(device=device, dtype=torch.float32)
            images_tp1 = images_pair[:,1,:,:,:].to(device=device, dtype=torch.float32)
            class_labels = class_labels_pair[:,0,:,:].to(device=device, dtype=torch.long) 
            # depth_labels = depth_labels_pair[:,0,:,:].to(device=device, dtype=torch.float32)   # torch.Size([N, 512, 1024])

            # feed data to model
            out_depth, out_class,out_ins_cen_pred,out_ins_cen_reg, out_ins_next = model(featuresT0=images_t,featuresT1=images_tp1)

    ## define a function for evluate? or seperately write and call them here.
            #evaluate
            # IoU for semantic  # input is a batch images # we can say cause they are consecutive frames/
            mIoU=Metric.compute_IoU(sem=out_class,target=class_labels)
            mIoU_list.append(mIoU)
            # MSE or L2 for depth 

            # VPQ for panoptic?

            # DVPQ for depth + panoptic?
    
    ## visualize metrics
    # mIoU hist map  --- semantic
    print(mIoU_list)
    bin_width =0.1
    bins = np.arange(min(mIoU_list),max(mIoU_list)+bin_width,bin_width)
    plt.hist(mIoU_list,bins=bins)
    plt.xlabel('mIoU value [batch]')
    plt.ylabel('Count')
    plt.title('Histogram for semantic mIoU')
    plt.show()


if __name__ == '__main__':
    use() # since in VIP_train, we already validate, when we use it, we only ues it postprocessing to see visualization.
    #showmap()
    #metric()