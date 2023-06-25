from dataclasses import dataclass


@dataclass
class Config:
    batch_size_train = 6
    batch_size_test = 6
    lr = 0.005
    lr_momentum = 0.9
    weight_decay = 1e-4
    num_classes = 10
    gt_dir = "./data/cifar-10-batches-py/"
    gt_dir_DVPS='./data/cityscapes-dvps/video_sequence/'
    num_iterations = 10000
    log_iterations = 100
    enable_cuda = False #True

    # backbone low,mid,high level channels
    low_channles=64
    mid_channles=512
    high_channles=1024

    #loss_weight={"depth":10 , "sem":1 , "center_pred":100 , "center_offset_t":0.1, "center_offset_t+1":0.1}
    loss_weight={"depth":1 , "sem":3 , "center_pred":200 , "center_offset_t":0.01, "center_offset_t+1":0.01}


    cityscape_thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
