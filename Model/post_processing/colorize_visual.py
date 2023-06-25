import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
import torch
import random


# 我的idea是把 输入的class id 当成 索引，这样的话要构建的list就得按对应索引来装颜色tuple
class Label:
    def __init__(self,name,num,class_id,category,category_id,hasInstances,ignoreInEval,color):
        self.name = name
        self.num = num
        self.class_id = class_id
        self.category = category
        self.category_id = category_id
        self.hasInstances = hasInstances
        self.ignoreInEval = ignoreInEval
        self.color = color

labels = [  #                          32 means unlabeled
    #       name                     num    class_id   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      32 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      32 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      32 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      32 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      32 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      32 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      32 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      32 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      32 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      32 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      32 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      32 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      32 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  10) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  20,  20) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  20,  20,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  10,  20, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  10, 60,100) ),
    Label(  'caravan'              , 29 ,      32 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      32 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  10, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  10,  20,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
]


def create_color_map(labels):
    class_color_map = {}
    for label in labels:
        if label.class_id<=18:
            key=label.class_id
            class_color_map[key]=label.color 
    class_color_map[32]=(0,0,0)# view all id=32 as black.
    #class_color_map[255]=(0, 0, 0)
    return class_color_map

def sem_color_convert(img): # input tensor [1, H, W]
    width = img.size(-1)
    height = img.size(-2)
    img=img.squeeze().numpy() # [H,W] numpy

    new_img = Image.new('RGB',(width,height))
    
    class_color_map = create_color_map(labels)

    for x in range(width):
        for y in range(height):
            pixel = img[y,x] # [H,w]
            color = class_color_map.get(pixel) # search based in pixel class_id
            new_img.putpixel((x,y),color)
    return np.array(new_img)  # output is :numpy array (convert from Image type)


# # # create a map to convert panoptic id into colors
def pan_color_convert(pan,label_divisor=1000): # input tensor [1, H, W]
    width = pan.size(-1)
    height = pan.size(-2)
    pan=pan.squeeze() # [H,W]  didn't transform it to numpy,cause unique need
    
    new_pan = Image.new('RGB',(width,height))

    class_color_map = create_color_map(labels)

    class_img=pan//label_divisor
    #print(torch.unique(class_img))
    ins_img = pan%label_divisor  # need it be tensor for unique

    # ratio for changing RGB channel.
    ins_list=torch.unique(ins_img).numpy() # tensor [1,2,3...]
    mean=np.mean(ins_list)
    ratio_ins_list=1+0.5*(ins_list-mean)/len(ins_list) # index=1 means ins_id =1, the value means ratio to covert RGB
    print(ratio_ins_list)
    np.random.seed(123)
    shift_ins_list=list(list(np.random.randint(low=-15, high=+15, size=3)) for _ in range(len(ins_list)))
    print(shift_ins_list)

    for x in range(width):
        for y in range(height):
            class_pixel = class_img[y,x].item() # [H,w] value = class_id 
            ins_pixel = ins_img[y,x].item()
            class_color = class_color_map.get(class_pixel) # [R,G,B]
            if ins_pixel!= 0:
                pan_color = tuple( np.floor(ratio_ins_list[ins_pixel-1]*x).astype(int) for x in class_color)  # ratio_ins_list start from index 0, but ins_pixel starts from 1
                pan_color = tuple(x+y for x,y in zip(list(pan_color),shift_ins_list[ins_pixel-1]))
                new_pan.putpixel((x,y),pan_color)
            else:
                new_pan.putpixel((x,y),class_color)
    return new_pan
    
    









# import matplotlib.pyplot as plt

# if __name__ == '__main__':
    # img=torch.tensor([[[0,0,0,0],[0,0,1,1],[32,32,32,32]]])
    # new=sem_color_convert(img)
    # print(np.array(new))
    # plt.imshow(np.array(new),vmin=0, vmax=255)
    # plt.show()


    # sem = torch.tensor([[[0.5,0.5,0.5,0.5],
    #                      [0.8,0.8,0.5,0.5],
    #                      [0.01,0.01,0.01,0.2]],
    #                      [[0.6,0.6,0.6,0.2],
    #                      [0.9,0.4,0.3,0.3],
    #                      [0.001,0.001,0.001,0.02]],
    #                      [[0,0,0,0],
    #                      [0.6,0.4,0.3,0.3],
    #                      [0.001,0.001,0.001,0.02]]
    #                      ])
    # sem = sem.squeeze(0) # squeeze if input is [1,c,h,w]=>[c,h,w]
    # output=torch.argmax(sem, dim=0, keepdim=True) #[1,h,w]
    # # add a confidence.
    # confidence,_ = torch.max(sem,dim=0,keepdim=True) # if confidence<0.1, view it as unlabel.
    # output=output*(confidence>0.1)+32*(confidence<0.1) # 
    # print(output)

    # pan = torch.tensor([[7001,7001,7002,7003],
    #                     [13002,13003,1004,2002],
    #                     [13005,0,0,0],
    #                     [1006,1007,1008,1009],
    #                     [5001,5001,0,0],
    #                     [1013,1010,1012,1011]
    #                     ])
    # new_pan=pan_color_convert(pan=pan,label_divisor=1000)
    # plt.imshow(new_pan)
    # plt.show()