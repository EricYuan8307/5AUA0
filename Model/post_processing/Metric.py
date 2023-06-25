import torch
# import post_processing
import numpy as np
import matplotlib.pyplot as plt

## semantic accuracy: 
def compute_IoU(sem,target,threshold=0.1,ignore_id=32,valid_id_num=19): # sem [B,C,H,W] target:[B,H,W]/ after modfiy dim=-3. now can also input single image.input [b,h,w] and target [h,w]
    sem=sem.detach().cpu()
    target=target.detach().cpu()

    semantic=torch.argmax(sem, dim=-3) #[B,C,H,W]=>[B,H,W]
    # add a confidence score tensor.
    confidence,_ = torch.max(sem,dim=-3) # =>[B,H,W] if confidence<0.1, view it as unlabel.
    semantic=semantic*(confidence>=threshold)+ignore_id*(confidence<threshold) #  convert pixel with max score<0.1 -> unlabel 32

    # create mask
    ingore_mask=(target==ignore_id)
    valid_mask = (target != ignore_id) # ignore unlabel id influence.
    IoU_list = []
    # IoU
    for cls in range(valid_id_num): # 0,..18 is true , for each class, calculate IoU separately
        pred_T = semantic==cls # = cls is True, != cls is False 
        target_T = target==cls
        pred_F = semantic!=cls
        target_F = target!=cls

        TP = torch.logical_and(torch.logical_and(pred_T,target_T),valid_mask).sum()
        FP = torch.logical_and(torch.logical_and(pred_T,target_F),valid_mask).sum()
        FN = torch.logical_and(torch.logical_and(pred_F,target_T),valid_mask).sum()
        if TP+FP+FN==0:
            continue
        else:
            IoU = TP/(TP+FP+FN) # for a single class
            IoU_list.append(IoU)
    # # mIoU
    mIoU=np.mean(IoU_list)
    return mIoU


## depth metric: absRel  
def abs_rel_error(depth_pred, depth_target): # [N,H,W]    [N,H,W] also can input[h,w] [h,w]
    depth_pred=depth_pred.detach().cpu()
    depth_target=depth_target.detach().cpu()
    # plt.imshow(depth_target[1], cmap='gray')  # cause target in cityscapes have pixel=0,which means background
    # plt.colorbar()
    # plt.show()
    # background masks 
    bg_mask = depth_target==0
    # min_val,_=torch.min(depth_target)
    # print(f"min pixel of target{min_val}")
    x=torch.abs(depth_pred-depth_target)/depth_target
    x = torch.where(bg_mask, torch.tensor([0.0]), x)  # convert pixels(value is inf) corresponding target =0 => 0. ignore background 
    abs_rel=torch.mean(x)
    #print(f'abs:{torch.abs(depth_pred-depth_target)}')
    return abs_rel.item()



# # test mIoU
# if __name__ == "__main__":
#     a=torch.tensor([[[1,1,1,1],
#                     [1,2,2,1],
#                     [10,10,20,20]],
#                     [[1,1,1,1],
#                      [1,1,1,1],
#                      [10,10,20,10]]
#                      ])
#     b=torch.tensor([[[1,1,1,1],
#                     [1,2,2,1],
#                     [10,10,20,20]],
#                     [[1,1,1,1],
#                      [1,1,1,1],
#                      [10,10,20,10]]
#                      ])
#     print(abs_rel_error(a,b))




#     sem= torch.tensor([[[[1,1,0,0],
#                          [0,0,0,0],
#                          [0,0,0,0]],
#                          [[0,0,1,1],
#                           [1,1,0,1],
#                           [1,1,0,1]],
#                           [[0,0,0,0],
#                            [0,0,1,0],
#                            [0,0,1,0]]]])
                        
#                         # [[[1,1,0,0],
#                         #  [0,0,0,0],
#                         #  [0,0,0,1]],
#                         #  [[0,0,1,1],
#                         #   [1,1,0,1],
#                         #   [1,1,0,1]],
#                         #   [[0,0,0,0],
#                         #    [0,0,1,0],
#                         #    [0,0,1,0]]]
                        
#     target= torch.tensor([[0,1,1,1],
#                           [1,1,1,1],
#                           [1,1,2,0]])
                        
#                         # [[[1,0,0,0],
#                         #  [0,0,0,0],
#                         #  [0,0,0,1]],
#                         #  [[0,1,1,1],
#                         #   [1,1,1,1],
#                         #   [1,1,0,0]],
#                         #   [[0,0,0,0],
#                         #    [0,0,0,0],
#                         #    [0,0,1,0]]]
                        
#     compute_IoU(sem,target,threshold=-1,ignore_id=32,valid_id_num=19)
    
        





