import torch
import torch.nn.functional as F

def get_semantic_segmentation(sem,threshold = 0.1,ignore_id=32): # threshold, <thershold means cannot believe the argmax class. view it as unlabel
    """
    Post-processing for semantic segmentation branch.
    Arguments:
        sem: A Tensor of shape [N, C, H, W], where N is the batch size, for consistent, we only
            support N=1.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.

    delete the channel and only keep information and pixels.
    only return the result that contain ID(which project it belongs) information for each pixel.
    """


    if sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    sem = sem.squeeze(0) # squeeze if input is [1,c,h,w]=>[c,h,w]
    output=torch.argmax(sem, dim=0, keepdim=True) #[1,h,w]
    # add a confidence.
    confidence,_ = torch.max(sem,dim=0,keepdim=True) # if confidence<0.1, view it as unlabel.
    output=output*(confidence>=threshold)+ignore_id*(confidence<threshold) # 

    return output,confidence


def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=7, top_k=200):
    """
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold(0.1) to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1 # only the maximum values (peaks) will remain unchanged, while other values are set to -1.

    # squeeze first two dimensions
    '''
    The purpose of this operation is to remove the dimensions with size 1, effectively collapsing them.
    By freezing the first two dimensions, the code can treat the heatmap as a 2D matrix, 
    allowing easier manipulation, thresholding, and non-maximum suppression operations.
    '''
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'
    # The subsequent operations assume this 2D shape, and the code asserts that the squeezed tensor has a size of 2.

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0) # create a tensor ctr_all containing the coordinates of these non-zero elements,
    # where each row represents a coordinate pair (y, x).
    if ctr_all.size(0) < top_k: # number of centers less than 200 
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k) # 从大到小排列k个值，第一个是value，第二个ctr_hmp中的位置
        return torch.nonzero(ctr_hmp > top_k_scores[-1]) # Obtain the indices of the center points


def group_pixels(ctr, offsets):
    """
    Gives each pixel in the image an instance id.
    Arguments:
        ctr: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    """
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0) # [2,H,W]
    height, width = offsets.size()[1:] # (H,W)

    # generates a coordinate map, where each location is the coordinate of that loc
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2) # [[[0,0,0];[1,1,1];[2,2,2];..]]
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1) # [[[0 1 2 ];[0 1 2],..]]
    coord = torch.cat((y_coord, x_coord), dim=0) # (2,H,W) 1st map is y_coord,..

    ctr_loc = coord + offsets  # eg 1st map, y value,  +  offsets[0] pixel y offset => predict each pixel corresponding mass center position!
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)  # (H*W,2)  row means point, col1 y col2 x (now x,y for each pixel means the mass center it belongs!)

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1) # based on center prediction, get instance centers coordinates 
    ctr_loc = ctr_loc.unsqueeze(0) # based on offset prediction, recover all center coordinates(for 2 pixels, if y,x value is the same, they belong to the same instance (mass) )

    # distance: [K, H*W]---> distance between each center postion "And" each pixel corresponding mass center position 
    distance = torch.norm(ctr - ctr_loc, dim=-1)  # - have broadcast, achieve: each center(predicted) - each pixels corresponding center.

    # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
    instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1 # torch.argmin(distance, dim=0) for each pixel, find the nearest center
    return instance_id # cause argmin return index of centers, the same like id!




def get_instance_segmentation(sem_seg, ctr_hmp, offsets, thing_list, threshold=0.1, nms_kernel=3, top_k=None,
                              thing_seg=None):
    """
    Post-processing for instance segmentation, gets class agnostic instance id map.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x). # 修改之后我的offset target图也是先y后x了!
        thing_list: A List of thing class id.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        thing_seg: A Tensor of shape [1, H, W], predicted foreground mask, if not provided, inference from
            semantic prediction.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
        A Tensor of shape [1, K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if thing_seg is None:
        # gets foreground segmentation
        thing_seg = torch.zeros_like(sem_seg)
        for thing_class in thing_list:
            thing_seg[sem_seg == thing_class] = 1  # assgin stuff id (which is not in thing_list) as 0 

    # ctr is center points coordinates [[y,x],[y,x],..]
    ctr = find_instance_center(ctr_hmp, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k)
    if ctr.size(0) == 0:
        return torch.zeros_like(sem_seg), ctr.unsqueeze(0)
    ins_seg = group_pixels(ctr, offsets)
    #print(f'after group_pixels, ins_seg{torch.unique(ins_seg)}')
    return thing_seg * ins_seg, ctr.unsqueeze(0)  # thing_seg * ins_seg if pixel is stuff, *0 =0!


def merge_semantic_and_instance(sem_seg, ins_seg, label_divisor, thing_list, stuff_area, void_label):
    """
    Post-processing for panoptic segmentation, by merging semantic segmentation label and class agnostic
        instance segmentation label.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.   eg:semantic uniquetensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15])
        ins_seg: A Tensor of shape [1, H, W], predicted instance label.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        thing_list: A List of thing class id.
        stuff_area: An Integer, remove stuff whose area is less tan stuff_area.
        void_label: An Integer, indicates the region has no confident prediction.  # == unlabeld semantic id * label_divisor
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.
    """
    # In case thing mask does not align with semantic prediction
    pan_seg = torch.zeros_like(sem_seg) + void_label
    thing_seg = ins_seg > 0
    semantic_thing_seg = torch.zeros_like(sem_seg)
    for thing_class in thing_list:
        semantic_thing_seg[sem_seg == thing_class] = 1
   
    # keep track of instance id for each class
    class_id_tracker = {} #按照ID来进行循环而不是pixel

    # paste stuff to unoccupied area  # change the stuff get color first, and let thing covering the right based color here. then the instance corresponding can change.
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in thing_list:
            # thing class  # meet thing id jumpy , therefore if it's instance is not extracted, here will be void. so to say black.
            #continue
            thing_mask = (sem_seg == class_id)
            pan_seg[thing_mask] = class_id * label_divisor  # give thing pixels initial colors

        # calculate stuff area
        else:
            stuff_mask = (sem_seg == class_id) & (~thing_seg)
            area = torch.nonzero(stuff_mask).size(0) # how much point here be predicted as stuff
            if area >= stuff_area:
                pan_seg[stuff_mask] = class_id * label_divisor

    # paste thing by majority voting====> vote for which class this instance belongs
    instance_ids = torch.unique(ins_seg) 
    for ins_id in instance_ids: # for each instance! 
        if ins_id == 0:
            continue
        # Make sure only do majority voting within semantic_thing_seg
        thing_mask = (ins_seg == ins_id) & (semantic_thing_seg == 1) # thing_mask = instance分割后，分割部分的ID, 而且in semantic != stuff class
        if torch.nonzero(thing_mask).size(0) == 0: #  thingmask里面必须要有东西
            continue
        class_id, _ = torch.mode(sem_seg[thing_mask].view(-1, )) # most frequent class(in the instance area), only one value
        if class_id.item() in class_id_tracker: # voting 如果新的ID在
            new_ins_id = class_id_tracker[class_id.item()] # 将新的ID从semantic传递到Instance
        else:
            class_id_tracker[class_id.item()] = 1 #
            new_ins_id = 1
        class_id_tracker[class_id.item()] += 1 # next time instance area belong this class , +1 for next instance!
        pan_seg[thing_mask] = class_id * label_divisor + new_ins_id

   

    return pan_seg


def get_panoptic_segmentation(sem, ctr_hmp, offsets, thing_list, label_divisor, stuff_area, void_label,
                              threshold=0.1, nms_kernel=3, top_k=None, foreground_mask=None):
    """
    Post-processing for panoptic segmentation.
    Arguments:
        sem: A Tensor of shape [N, C, H, W] of raw semantic output, where N is the batch size, for consistent,
            we only support N=1. Or, a processed Tensor of shape [1, H, W].
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        stuff_area: An Integer, remove stuff whose area is less tan stuff_area.
        void_label: An Integer, indicates the region has no confident prediction.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        foreground_mask: A Tensor of shape [N, 2, H, W] of raw foreground mask, where N is the batch size,
            we only support N=1. Or, a processed Tensor of shape [1, H, W].
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel), int64.
    Raises:
        ValueError, if batch size is not 1.
    """
    if sem.dim() != 4 and sem.dim() != 3:
        raise ValueError('Semantic prediction with un-supported dimension: {}.'.format(sem.dim()))
    if sem.dim() == 4 and sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if foreground_mask is not None:
        if foreground_mask.dim() != 4 and foreground_mask.dim() != 3:
            raise ValueError('Foreground prediction with un-supported dimension: {}.'.format(sem.dim()))

    if sem.dim() == 4:
        semantic = get_semantic_segmentation(sem)
    else:
        semantic = sem

    if foreground_mask is not None:
        if foreground_mask.dim() == 4:
            thing_seg = get_semantic_segmentation(foreground_mask)
        else:
            thing_seg = foreground_mask
    else:
        thing_seg = None

    instance, center = get_instance_segmentation(semantic, ctr_hmp, offsets, thing_list,
                                                 threshold=threshold, nms_kernel=nms_kernel, top_k=top_k,
                                                 thing_seg=thing_seg)
    panoptic = merge_semantic_and_instance(semantic, instance, label_divisor, thing_list, stuff_area, void_label)

    return panoptic, center