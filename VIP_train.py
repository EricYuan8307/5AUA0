import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import numpy as np
from config import Config
from VIP_data import MyDataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import sys
import torch.nn.functional as F

from Model.Decoder.panoptic_decoder import DecoderArch
from EarlyStopping import EarlyStopping

from Model.post_processing import Metric

class Trainer:
    def __init__(self,model:nn.Module,cfg):
        # choose a device 
        if cfg.enable_cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.cfg = cfg
        # move model to device
        self.model = model.to(self.device)

        # learning rate
        self.lr = self.cfg.lr

        # optimizer  # maybe need change for different branch????
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        # scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5,min_lr=1e-9)
        
        # crtitereon # cross entropy 
        self.critic_class = nn.CrossEntropyLoss(ignore_index=32) # we can tune the weight here like panoptic-deeplab did, to address lees common objects
        # instance center predict , use MSE
        self.critic_ins_cen_pred = nn.MSELoss(reduction='none') # (output,target) neeed change targe as 2D gaussian heat map
        # instance center regression--actually offset, use
        self.critic_ins_cen_offset = nn.L1Loss(reduction='none') # (output,target) neeed change targe ??? what is the target?
        # ???? depth use ?? loss??    
        #self.critic_dep = 
        
        self.early_stopping = EarlyStopping(patience=4,delta=0.001,path='VIP_model.pth')
        ###################
        assert self.optimizer is not None, "Have no optimizer "
        # assert self.critereon is not None, "You have not defined a loss"





##################### calculate loss ##################

    ### convert instance label figures into Gaussian heatmap , variance is 8, mean of each instance is through calculate it's mean position
    def Convert_ins2heatmap(self,instance_labels,sigma=8,threshold = 0.05): # input is tensor  like torch.Size([5, 512, 1024])
        # 8 pixels, as paper panoptic deeplab shown 

        # create a same size heatmap tensor
        heatmap_center = torch.zeros_like(instance_labels,dtype=torch.float,device=instance_labels.device)
        points_x = torch.arange(start=0,end=instance_labels.size(dim=-1)) # [0,..1023] tensor[1024,]
        points_x = points_x.repeat(instance_labels.size(dim=-2),1)
        points_y = torch.arange(start=0,end=instance_labels.size(dim=-2)).unsqueeze(1) # [0,1,2...511] tensor[512,1]
        points_y = points_y.repeat(1,instance_labels.size(dim=-1))

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



    ### Depth loss
    def depth_act(self, input, MaxDepth=88):  # Depth = MaxDepth X Sigmoid(f_d)
        depth = MaxDepth * torch.sigmoid(input)
        return depth


    def log_mse(self, output, gt):  # the first loss
        # output dimension: (B,H,W)
        ignore_label = 0.0
        label_mask = gt != ignore_label
        # print(label_mask)

        masked_output = torch.masked_select(output, label_mask)  # Apply mask to output
        masked_gt = torch.masked_select(gt, label_mask)  # Apply mask to gt

        log_output = torch.log(masked_output)  # log(di_hat)
        log_gt = torch.log(masked_gt)  # log(di)
        diff = log_gt - log_output
        diff = torch.pow(diff, 2)
        mse = torch.mean(diff)

        return mse.item()

    def n2_log_mse(self, output, gt):  # the second loss
        # output dimension: (B,H,W)
        ignore_label = 0
        label_mask = gt != ignore_label

        masked_output = torch.masked_select(output, label_mask)  # Apply mask to output
        masked_gt = torch.masked_select(gt, label_mask)  # Apply mask to gt

        log_output = torch.log(masked_output)  # log(di_hat)
        log_gt = torch.log(masked_gt)  # log(di)
        diff = log_gt - log_output
        diff = torch.mean(diff)
        n2_mse = torch.pow(diff, 2)

        return n2_mse.item()

    def rmse(self, output, gt):  # the third loss
        ignore_label = 0
        label_mask = gt != ignore_label

        masked_output = torch.masked_select(output, label_mask)  # Apply mask to output
        masked_gt = torch.masked_select(gt, label_mask)  # Apply mask to gt

        diff = (masked_gt - masked_output) / masked_gt
        diff = torch.pow(diff, 2)
        mse = torch.mean(diff)
        mse = torch.sqrt(mse)

        return mse.item()

    def loss_depth(self, output, gt):  # loss for depth
        loss = 0
        iter = 0

        for m, n in zip(output, gt):
            if torch.sum(n) == 0:
                continue


            output_norm = self.depth_act(m)
            gt_norm = n/255

            los_depth = self.log_mse(output_norm, gt_norm) - self.n2_log_mse(output_norm, gt_norm) + self.rmse(output_norm, gt_norm)
            loss += los_depth
            iter += 1

        return loss / iter



### Instance center offset -----target offset map ,for offset loss calculate! 
    def generate_offset_map(self,target_map):# target_map means "instance_labels" ground truth!
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
            centers = self.calculate_centers(target_map[i], instance_ids)
            
            # Calculate the offset of the nearest center position corresponding to each pixel
            offset = self.calculate_offset(target_map[i], centers)
            
            # save
            offset_map[i] = offset  # cause initial is tensor so output is still tensor! 
        
        return offset_map

    def calculate_centers(self,target, instance_ids): # instance_ids = [1,2...] unique id sequence without 0
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

    def calculate_offset(self,target, centers):  # input is single one figure in a batch
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
        
        return offset     # instance_labels torch.Size([32, 1024, 2048])  # 因为使用的输入不是一个batch而是单张，所以output是[2,H,W]

    # 示例使用


    # offset_map = generate_offset_map(target_map)
    # print(offset_map.size())

########################################################## finish calculate loss def  #################




    def train_epoch(self,dataloader:DataLoader,loss_weight):
        # model in train mode
        self.model.train()

        # store acc and loss of each epoch
        train_metrics = {
            "total loss":[],"loss_depth":[],"loss_semantic":[],"loss_center_predict":[],"loss_center_offset_t":[],"loss_center_offset_t+1":[],
            "sem_accuracy(mIoU)":[],"depth_absRel":[]
        }

        # iterate over dataset
        sys.stdout.flush()
        with tqdm(total=len(dataloader), desc=f'Training') as pbar:  #???
        
            for (images_pair, class_labels_pair,instance_labels_pair,depth_labels_pair) in  dataloader:
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

                # convert instance label figures into Gaussian center heat-map  ==== ready for calculate loss
                heatmap_center=self.Convert_ins2heatmap( instance_labels_pair[:,0,:,:],sigma=8) # instance_labels torch.Size([32, 1024, 2048]) // each batch
                center_weight = self.Convert_ins2heatmap( instance_labels_pair[:,0,:,:],sigma=12)

                # convert instacne label figures inot offset map ground truth
                offsetmap_t = self.generate_offset_map(instance_labels_pair[:,0,:,:])
                offsetmap_tp1 = self.generate_offset_map(instance_labels_pair[:,1,:,:])

                # move info to target device
                images_t = images_pair[:,0,:,:,:].to(device=self.device, dtype=torch.float32)
                images_t.requires_grad = True # Fix for older PyTorch versions
                images_tp1 = images_pair[:,1,:,:,:].to(device=self.device, dtype=torch.float32)
                images_tp1.requires_grad= True

                class_labels = class_labels_pair[:,0,:,:].to(device=self.device, dtype=torch.long) # [N,H,W]
                heatmap_center = heatmap_center.to(device=self.device, dtype=torch.float32)
                center_weight = center_weight.to(device = self.device, dtype = torch.float32)
                
                offsetmap_t = offsetmap_t.to(device=self.device, dtype=torch.float32) # [B,2,H,W]
                offset_weight_t = (instance_labels_pair[:,0,:,:]!=0).to(device=self.device, dtype=torch.float32)
                offset_weight_t = offset_weight_t.to(device=self.device, dtype=torch.float32)
                offsetmap_tp1 = offsetmap_tp1.to(device=self.device, dtype=torch.float32)
                offset_weight_p1 = instance_labels_pair[:,1,:,:]!=0
                offset_weight_p1 = offset_weight_p1.to(device=self.device, dtype=torch.float32)
                
                depth_labels = depth_labels_pair[:,0,:,:].to(device=self.device, dtype=torch.float32)  # [N,H,W]

                # run model 
                out_depth, out_class,out_ins_cen_pred,out_ins_cen_reg, out_ins_next = self.model(featuresT0=images_t,featuresT1=images_tp1)
                ## squeeze to remove only 1 element dimension, do not need convert class map [N,C,H,W] caue nn.Crossentropyloss will do it automatically
                out_depth = torch.squeeze(out_depth,dim=1) # [N,H,W ]
                out_ins_cen_pred = torch.squeeze(out_ins_cen_pred,dim=1)
                out_ins_cen_reg =  torch.squeeze(out_ins_cen_reg,dim=1)
                out_ins_next = torch.squeeze(out_ins_next,dim=1)
                
                #### Backpropagation

                loss_depth = self.loss_depth(out_depth,depth_labels)
                loss_semantic = self.critic_class(out_class,class_labels)  # input=[N,C,H,W] target=[N,H,W]
                
                ## center prediction loss
                # normalize
                out_ins_cen_pred=(out_ins_cen_pred-torch.min(out_ins_cen_pred))/(torch.max(out_ins_cen_pred)-torch.min(out_ins_cen_pred))
                loss_center_prediction = self.critic_ins_cen_pred(out_ins_cen_pred, heatmap_center) # input = [N,H,W]  # target = [N,H,W]  MSE reduction=None, so return tensor
                loss_center_prediction=loss_center_prediction*center_weight
                #safe division
                if center_weight.sum() >0:
                    loss_center_prediction = loss_center_prediction.sum() / center_weight.sum()
                else:
                    loss_center_prediction = loss_center_prediction.sum()*0

                ## for t sample, offset loss 
                # add a instance mask,or weight, only consider instances pixels!
                loss_center_offset_t = self.critic_ins_cen_offset(out_ins_cen_reg, offsetmap_t) # L1 loss # offset # return[B,2,H,W]
                loss_center_offset_t=torch.mul(loss_center_offset_t,offset_weight_t.unsqueeze(1))  # [B,2,H,W].  * [B,1,H,W]
                #safe division
                if offset_weight_t.sum() >0:
                    loss_center_offset_t = loss_center_offset_t.sum()/offset_weight_t.sum()
                else:
                    loss_center_offset_t = loss_center_offset_t.sum()*0

                ## for t+1 sample, loss
                loss_center_offset_tp1 = self.critic_ins_cen_offset(out_ins_next,offsetmap_tp1)
                loss_center_offset_tp1=torch.mul(loss_center_offset_tp1,offset_weight_p1.unsqueeze(1))
                #safe division
                if offset_weight_t.sum() >0:
                    loss_center_offset_tp1 = loss_center_offset_tp1.sum()/offset_weight_p1.sum()
                else:
                    loss_center_offset_tp1 = loss_center_offset_tp1.sum()*0


                ## loss weight need tune somehow.
                loss = loss_weight['depth']*loss_depth + loss_weight['sem']*loss_semantic + loss_weight['center_pred']*loss_center_prediction + loss_weight['center_offset_t']*loss_center_offset_t + loss_weight['center_offset_t+1']*loss_center_offset_tp1  # ?? can define weight for each loss.  #

                self.optimizer.zero_grad() 
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 0.1) #clip the gradient to avoid gradient explosion
                self.optimizer.step() #update params in one batch

                ### store metrics in one batch
                batch_mertrics = {
                    "total loss":loss.item(), # convert tensor into scalar  # ？？ can add different loss to show
                    "loss_depth":loss_depth,
                    "loss_semantic":loss_semantic.item(),
                    "loss_center_predict":loss_center_prediction.item(),
                    "loss_center_offset_t":loss_center_offset_t.item(),
                    "loss_center_offset_t+1":loss_center_offset_tp1.item(),
                    'sem_accuracy(mIoU)':Metric.compute_IoU(sem=out_class,target=class_labels), # return is scalar
                    "depth_absRel":Metric.abs_rel_error(depth_pred=out_depth,depth_target=depth_labels)
                    # "accuracy":[]
                    # "accuracy semantic": compute_acc_class(out_class,class_labels), #??????? need self defined acc, #cause only loss can be get from metric
                    # "accuracy instance": compute_acc_ins(out_instance,instance_labels),
                    # "accuracy depth": compute_acc_depth(out_depth,depth_labels)
                }

                # Update the progress bar
                pbar.set_postfix(**batch_mertrics)
                pbar.update(1)

                # put batch metric into epoch metric
                for k,v in batch_mertrics.items():
                    train_metrics[k].append(v)
            sys.stdout.flush()
            #print(f' last epoch train total loss is {epoch_metrics["total loss"]}')
        return train_metrics  # actually only last epoch metrics
        

    def val_epoch(self,dataloader:DataLoader,loss_weight):
        # evaluation mode:
        self.model.eval()

        # store loss and  accuracy?  # is the DVPQ metric needs to be done here? no 在后处理阶段用于评估和分析模型的性能
        count=0
        total_loss = 0
        cum_loss_depth = 0
        cum_loss_sem = 0 
        cum_loss_cen_pre = 0
        cum_loss_cen_offset_t = 0
        cum_loss_cen_offset_tp = 0
        cum_acc_sem = 0
        cum_absrel_depth=0

        sys.stdout.flush()
        with torch.no_grad(), tqdm(total=len(dataloader), desc=f'Val') as pbar:  #???
            for (images_pair, class_labels_pair,instance_labels_pair,depth_labels_pair) in  dataloader:
                H = images_pair.size(-2)
                W = images_pair.size(-1)
                images_pair = [sample for sample in images_pair if torch.sum(sample)!=0]  # ignore specific figures from getitem!!!
                images_pair= torch.stack(images_pair,dim=0)
                class_labels_pair = [sample for sample in class_labels_pair if torch.sum(sample)!=-2*H*W]
                class_labels_pair= torch.stack(class_labels_pair,dim=0)
                instance_labels_pair = [sample for sample in instance_labels_pair if torch.sum(sample)!=-2*H*W]
                instance_labels_pair =  torch.stack(instance_labels_pair,dim=0)
                depth_labels_pair = [sample for sample in depth_labels_pair if torch.sum(sample)!=-2*H*W]
                depth_labels_pair= torch.stack(depth_labels_pair,dim=0)  

                # convert instance label figures into Gaussian center heat-map  ==== ready for calculate loss
                heatmap_center=self.Convert_ins2heatmap( instance_labels_pair[:,0,:,:]) # instance_labels torch.Size([32, 1024, 2048]) // each batch
                center_weight = self.Convert_ins2heatmap( instance_labels_pair[:,0,:,:],sigma=12)
                # convert instacne label figures inot offset map ground truth
                offsetmap_t = self.generate_offset_map(instance_labels_pair[:,0,:,:])
                offsetmap_tp1 = self.generate_offset_map(instance_labels_pair[:,1,:,:])

                # zero gradients from previous epoch
                self.optimizer.zero_grad()

                # move info to target device
                images_t = images_pair[:,0,:,:,:].to(device=self.device, dtype=torch.float32)
                images_tp1 = images_pair[:,1,:,:,:].to(device=self.device, dtype=torch.float32)


                class_labels = class_labels_pair[:,0,:,:].to(device=self.device, dtype=torch.long) 
                heatmap_center = heatmap_center.to(device=self.device, dtype=torch.float32)
                center_weight = center_weight.to(device = self.device, dtype = torch.float32)
                offsetmap_t=offsetmap_t.to(device=self.device, dtype=torch.float32)
                offset_weight_t = instance_labels_pair[:,0,:,:]!=0
                offset_weight_t=offset_weight_t.to(device=self.device, dtype=torch.float32)
                
                offsetmap_tp1=offsetmap_tp1.to(device=self.device, dtype=torch.float32) 
                offset_weight_p1 = instance_labels_pair[:,1,:,:]!=0
                offset_weight_p1 = offset_weight_p1.to(device=self.device, dtype=torch.float32) 
                
                depth_labels = depth_labels_pair[:,0,:,:].to(device=self.device, dtype=torch.float32)   # torch.Size([N, 512, 1024])

                 # run model 
                out_depth, out_class,out_ins_cen_pred,out_ins_cen_reg, out_ins_next = self.model(featuresT0=images_t,featuresT1=images_tp1)
                # from [N,C,H,W] -> [N,H,W] ## squeeze to remove only 1 element dimension, do not need convert class map [N,C,H,W] caue nn.Crossentropyloss will do it automatically
                out_depth = torch.squeeze(out_depth,dim=1)
                out_ins_cen_pred = torch.squeeze(out_ins_cen_pred,dim=1)
                out_ins_cen_reg =  torch.squeeze(out_ins_cen_reg,dim=1)
                out_ins_next = torch.squeeze(out_ins_next,dim=1)


                # calculate loss
                loss_depth = self.loss_depth(out_depth,depth_labels)  
                loss_semantic = self.critic_class(out_class,class_labels)  # input=[N,C,H,W] target=[N,H,W]
                
                out_ins_cen_pred=(out_ins_cen_pred-torch.min(out_ins_cen_pred))/(torch.max(out_ins_cen_pred)-torch.min(out_ins_cen_pred))
                loss_center_prediction = self.critic_ins_cen_pred(out_ins_cen_pred, heatmap_center) # input = [N,H,W]  # target = [N,H,W]  MSE reduction=None, so return tensor
                loss_center_prediction=loss_center_prediction*center_weight
                #safe division
                if center_weight.sum() >0:
                    loss_center_prediction = loss_center_prediction.sum() / center_weight.sum()
                else:
                    loss_center_prediction = loss_center_prediction.sum()*0
                ## for t sample, offset loss 
                # add a instance mask,or weight, only consider instances pixels!
                loss_center_offset_t = self.critic_ins_cen_offset(out_ins_cen_reg, offsetmap_t) # L1 loss # offset # return[B,2,H,W]
                loss_center_offset_t=torch.mul(loss_center_offset_t,offset_weight_t.unsqueeze(1))  # [B,2,H,W].  * [B,1,H,W]
                #safe division
                if offset_weight_t.sum() >0:
                    loss_center_offset_t = loss_center_offset_t.sum()/offset_weight_t.sum()
                else:
                    loss_center_offset_t = loss_center_offset_t.sum()*0# for t+1 sample, loss


                ## for t+1 sample, loss
                loss_center_offset_tp1 = self.critic_ins_cen_offset(out_ins_next,offsetmap_tp1)
                loss_center_offset_tp1=torch.mul(loss_center_offset_tp1,offset_weight_p1.unsqueeze(1))
                #safe division
                if offset_weight_t.sum() >0:
                    loss_center_offset_tp1 = loss_center_offset_tp1.sum()/offset_weight_p1.sum()
                else:
                    loss_center_offset_tp1 = loss_center_offset_tp1.sum()*0


                ## loss weight need tune somehow.
                loss = loss_weight['depth']*loss_depth + loss_weight['sem']*loss_semantic + loss_weight['center_pred']*loss_center_prediction + loss_weight['center_offset_t']*loss_center_offset_t + loss_weight['center_offset_t+1']*loss_center_offset_tp1  # ?? can define weight for each loss.  #

                ## store loss metric
                step_metrics = {
                    'total loss':loss.item(),
                    "loss_depth":loss_depth,
                    "loss_semantic":loss_semantic.item(),
                    "loss_center_predict":loss_center_prediction.item(),
                    "loss_center_offset_t":loss_center_offset_t.item(),
                    "loss_center_offset_t+1":loss_center_offset_tp1.item(),
                    'sem_accuracy(mIoU)':Metric.compute_IoU(sem=out_class,target=class_labels),
                    "depth_absRel":Metric.abs_rel_error(depth_pred=out_depth,depth_target=depth_labels)
                } # if need accuracy add here

                count +=1
                total_loss += step_metrics['total loss']
                cum_loss_depth += step_metrics['loss_depth']
                cum_loss_sem += step_metrics['loss_semantic']
                cum_loss_cen_pre += step_metrics['loss_center_predict']
                cum_loss_cen_offset_t += step_metrics['loss_center_offset_t']
                cum_loss_cen_offset_tp += step_metrics['loss_center_offset_t+1']

                cum_acc_sem += step_metrics['sem_accuracy(mIoU)']
                cum_absrel_depth +=step_metrics['depth_absRel']

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(1)
            sys.stdout.flush()

            # print mean of metrics
            total_loss /=count
            print(f'Validation total loss is {total_loss}')
            cum_loss_depth/=count
            cum_loss_sem/=count
            cum_loss_cen_pre/=count
            cum_loss_cen_offset_t/=count
            cum_loss_cen_offset_tp/=count
            cum_acc_sem/=count
            cum_absrel_depth/=count


            # schedular for changing learning rate
            self.scheduler.step(total_loss)

            # early stop 
            self.early_stopping(total_loss,self.model)
            if self.early_stopping.early_stop:
                print('Early stopping')
                print(f'Stop at  total val_loss is {total_loss}')
            
            return {
                "average total loss":[total_loss],  # if need to check accuracy
                "average depth loss":[cum_loss_depth],
                "average semantic loss":[cum_loss_sem],
                "average center prediction loss":[cum_loss_cen_pre],
                "average center offset loss t":[cum_loss_cen_offset_t],
                "average center offset loss t+1":[cum_loss_cen_offset_tp],
                "average semantic accuracy":[cum_acc_sem],
                "average depth absRel error":[cum_absrel_depth]
            }

            

    # def collate_fn(self,batch): # filter skip in dataset!
    #         batch = [sample for sample in batch if sample != "SKIP"]  # 过滤掉特殊值
    #         return torch.utils.data.dataloader.default_collate(batch)


    def fit(self,epochs:int): # used to iterate epochs training  and validation
        # Configuration settings
            loss_weight = self.cfg.loss_weight # {"depth":10 , "sem":1 , "center_pred":100 , "center_offset_t":0.1, "center_offset_t+1":0.1}
        # Load dataset
            dataset_train = MyDataset(self.cfg.gt_dir_DVPS, split='train_1024_512') # input,truth_class,truth_ins,truth_depth
            dataset_val = MyDataset(self.cfg.gt_dir_DVPS,split='val_1024_512')
            dataloader_train = DataLoader(dataset_train, batch_size=self.cfg.batch_size_train, shuffle=True, num_workers=4,drop_last=True) # ？视频处理需不需要shuffle？ 需要，只要连续俩一对就好了，按pair shuffle
            dataloader_val = DataLoader(dataset_val, batch_size=self.cfg.batch_size_test, shuffle=False, num_workers=4,drop_last=True)

        # save 
            list_metrics_train_batch=[] # only for last batch in a epoch
            list_metrics_val_epoch=[] # for epoch
        # train
            for epoch in range(1,epochs+1):
                print(f'Epoch{epoch}')
                metrics_train = self.train_epoch(dataloader_train,loss_weight=loss_weight)
                print(f' (Average) train total loss is {np.mean(metrics_train["total loss"])}')
                print(f' (Average) depth prediction loss is {np.mean(metrics_train["loss_depth"])}')
                print(f' (Average) semantic loss is {np.mean(metrics_train["loss_semantic"])}')
                print(f' (Average) center prediction loss is  {np.mean(metrics_train["loss_center_predict"])}')
                print(f' (Average) center regression loss for frame t is {np.mean(metrics_train["loss_center_offset_t"])}')
                print(f' (Average) center regression loss for frame t+1 is {np.mean(metrics_train["loss_center_offset_t+1"])}')
                print(f' (Average) Accuracy of semantic is {np.mean(metrics_train["sem_accuracy(mIoU)"])}')
                print(f' (Average) absRel error of depth is {np.mean(metrics_train["depth_absRel"])}')
                
                

                print('Validation')
                metrics_val = self.val_epoch(dataloader_val,loss_weight=loss_weight)
                print(f' (Average) validate total loss is {metrics_val["average total loss"]}')
                print(f' (Average) depth prediction loss is {metrics_val["average depth loss"]}')
                print(f' (Average) semantic loss is {metrics_val["average semantic loss"]}')
                print(f' (Average) center prediction loss is  {metrics_val["average center prediction loss"]}')
                print(f' (Average) center regression loss for frame t is {metrics_val["average center offset loss t"]}')
                print(f' (Average) center regression loss for frame t+1 is {metrics_val["average center offset loss t+1"]}')
                print(f' (Average) Accuracy of semantic is {metrics_val["average semantic accuracy"]}')
                print(f' (Average) absRel error of depth is {metrics_val["average depth absRel error"]}')
                
                list_metrics_train_batch.append(metrics_train) #element is dict
                list_metrics_val_epoch.append(metrics_val)

            return list_metrics_train_batch, list_metrics_val_epoch




if __name__ == '__main__':
    torch.cuda.empty_cache()
    model=DecoderArch()
    cfg=Config()
    train = Trainer(model,cfg)
    
    list_metrics_train_batch, list_metrics_val_epoch=train.fit(epochs=9) # two list can be used to draw loss changing figure

    # Visualize loss change
    loss_names = list(list_metrics_val_epoch[0].keys())
    plt.figure(figsize=(10,6))
    for i, loss_name in enumerate(loss_names):
        loss_values = [loss_dict[loss_name][0] for loss_dict in list_metrics_val_epoch]
        plt.plot(loss_values,label=loss_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    # cause using early stop, save model in earlystopping function already




# ###check shape from dataloader
#     dataset_train = MyDataset(cfg.gt_dir_DVPS, split='train_1024_512') # input,truth_class,truth_ins,truth_depth
#     dataset_val = MyDataset(cfg.gt_dir_DVPS,split='val_1024_512')
#     dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size_train, shuffle=False, num_workers=4,drop_last=True) # ？视频处理需不需要shuffle？ 需要，只要连续俩一对就好了，按pair shuffle
    
#     samllest_dim0 = 5
#     for (images_pair, class_labels_pair,instance_labels_pair,depth_labels_pair) in  dataloader_train:
#             H = images_pair.size(-2)
#             W = images_pair.size(-1)
#             images_pair = [sample for sample in images_pair if torch.sum(sample)!=0]  # ignore specific figures from getitem!!!  
#             images_pair= torch.stack(images_pair,dim=0)
#             # class_labels_pair = [sample for sample in class_labels_pair if torch.sum(sample)!=-2*H*W]
#             # class_labels_pair= torch.stack(class_labels_pair,dim=0)
#             # instance_labels_pair = [sample for sample in instance_labels_pair if torch.sum(sample)!=-2*H*W]
#             # instance_labels_pair =  torch.stack(instance_labels_pair,dim=0)
#             # depth_labels_pair = [sample for sample in depth_labels_pair if torch.sum(sample)!=-2*H*W]
#             # depth_labels_pair= torch.stack(depth_labels_pair,dim=0)

#             # convert instance label figures into Gaussian center heat-map  ==== ready for calculate loss
#             # heatmap_center=train.Convert_ins2heatmap( instance_labels_pair[:,0,:,:]) # instance_labels torch.Size([32, 1024, 2048]) // each batch
#             # if heatmap_center.size(0)<samllest_dim0:
#             #     samllest_dim0=heatmap_center.size(0)
#             # # convert instacne label figures inot offset map ground truth
#             # offsetmap_t = self.generate_offset_map(instance_labels_pair[:,0,:,:])
#             # offsetmap_tp1 = self.generate_offset_map(instance_labels_pair[:,1,:,:])

#             # move info to target device
#             images_t = images_pair[:,0,:,:,:]#.to(device=self.device, dtype=torch.float32)
#             if images_t.size(0)<samllest_dim0:
#                 samllest_dim0=images_t.size(0)
#             #images_t.requires_grad = True # Fix for older PyTorch versions
#             # images_tp1 = images_pair[:,1,:,:,:].to(device=self.device, dtype=torch.float32)
#             # images_tp1.requires_grad= True

#             # class_labels = class_labels_pair[:,0,:,:].to(device=self.device, dtype=torch.long) # [N,H,W]
#             # heatmap_center = heatmap_center.to(device=self.device, dtype=torch.float32)
#             # offsetmap_t=offsetmap_t.to(device=self.device, dtype=torch.float32)
#             # offsetmap_tp1=offsetmap_tp1.to(device=self.device, dtype=torch.float32)
#             # depth_labels = depth_labels_pair[:,0,:,:].to(device=self.device, dtype=torch.float32)  # [N,H,W]

#             print(f"images_t size:{images_t.size()}")
#     print(f'mini dim 0 :{samllest_dim0}')



            ###### test for size and so on.
            # for (images, class_labels,instance_labels,depth_labels) in  dataloader_train:
                # convert instance label figures into Gaussian center heat-map  ==== ready for calculate loss
        
                # heatmap_center=Convert_ins2heatmap( instance_labels) # instance_labels torch.Size([32, 1024, 2048]) // each batch
                # print(heatmap_center.shape)
                # heatmap_np=heatmap_center.numpy()


                # for i in range(heatmap_np.shape[0]):
                #     plt.imshow(heatmap_np[i], cmap='hot', interpolation='nearest')
                #     plt.colorbar()
                #     plt.show()

                # # check offset map # torch.Size([32, 2, 1024, 2048])
                # offset_map = generate_offset_map(instance_labels)
                # print(offset_map.size())

        # # check dataloader: image visualize 
        #     i=0
        #     while i < 1:
        #         for (images, class_labels,instance_labels,depth_labels) in  dataloader:
        #             image = class_labels[i]
        #             #image = TF.to_pil_image(image) ## only image do this, labels directly output array
        #             #cmap=ListedColormap(list(color_map_ins.values()))
        #             #color_image = cmap(image)
        #             plt.imshow(image)
        #             plt.axis('off')
        #             plt.show()
        #         i+=1


# from PIL import Image

# #test figure t and t+1
# if __name__ == '__main__':
#     model=DecoderArch()
#     cfg=Config()
#     # train = Trainer(model,cfg)
#     # train.fit(epochs=1)
#     dataset_train = MyDataset(cfg.gt_dir_DVPS, split='train')
#     dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size_train, shuffle=False, num_workers=1)
    
#     for images_pair, class_labels_pair,instance_labels_pair,depth_labels_pair in  dataloader_train:
#         images_pair = [sample for sample in images_pair if torch.sum(sample)!=0]
#         images_pair= torch.stack(images_pair,dim=0)
#         class_labels_pair = [sample for sample in class_labels_pair if torch.sum(sample)!=0]
#         class_labels_pair= torch.stack(class_labels_pair,dim=0)
#         instance_labels_pair = [sample for sample in instance_labels_pair if torch.sum(sample)!=0]
#         instance_labels_pair =  torch.stack(instance_labels_pair,dim=0)
#         depth_labels_pair = [sample for sample in depth_labels_pair if torch.sum(sample)!=0]
#         depth_labels_pair= torch.stack(depth_labels_pair,dim=0)

#         if len(images_pair)==0:
#             continue
#         #images_pair, class_labels_pair,instance_labels_pair,depth_labels_pair=tensors
#         for i in [0,1,2,3,4]:
#             print(class_labels_pair.shape)
#             # print(images_pair[:,0,:,:,:].shape)
#             # print(images_pair[1,0].shape)
#             print( torch.sum(images_pair[i,0,:,:,:]!=images_pair[i,1,:,:,:]))

#             image_array=np.transpose(images_pair[i,0,:,:,:].numpy(),(1,2,0))
#             image = Image.fromarray(image_array)
#             image.show()
#             image_array_1=np.transpose(images_pair[i,1,:,:,:].numpy(),(1,2,0))
#             image_1 = Image.fromarray(image_array_1)
#             image_1.show()

#             # image_array_1=np.transpose(images_pair[i,1,:,:,:].numpy(),(1,2,0))
#             # image_1 = Image.fromarray(image_array_1)
#             # image_1.show()
            

#             user_input = input("input a value: ")
#             print("your is input is:", user_input)
#             # plt.imshow(images_pair[1,0,:,:,:])
#             # plt.imshow(images_pair[1,1,:,:,:])

    












## check dataset shape, already pass 
# def check_dataset(dataset):
#     # Check the length of the dataset
#     print(f"Dataset length: {len(dataset)}")

#     # Check a few random samples from the dataset
#     num_samples_to_check = 5
#     random_indices = np.random.choice(len(dataset), num_samples_to_check, replace=False)
#     print("Checking random samples:")
#     for idx in random_indices:
#         sample = dataset[idx]
#         print(f"Sample {idx}:")
#         input_image, truth_class, truth_ins, truth_depth = sample

#         # Print the shapes of the data
#         print("Input image shape:", input_image.shape)
#         print("Truth class shape:", truth_class.shape)
#         print("Truth instance shape:", truth_ins.shape)
#         print("Truth depth shape:", truth_depth.shape)

#         # Check the data types
#         print("Input image type:", input_image.dtype)
#         print("Truth class type:", truth_class.dtype)
#         print("Truth instance type:", truth_ins.dtype)
#         print("Truth depth type:", truth_depth.dtype)

#         # Add additional checks specific to your dataset if needed

#         print()  # Add a newline for readability

#     print("Dataset check complete.")

# # Usage example
# cfg = Config()
# dataset = MyDataset(cfg.gt_dir_DVPS, split='train')
# check_dataset(dataset)
