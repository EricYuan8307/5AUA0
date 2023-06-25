# once we use processed semikkit, we don't need this prepare 
# maybe we can do data augmentation here!?? 

import os
from config import Config
from PIL import Image
import numpy as np

cfg = Config()
test_dir=os.path.join(cfg.gt_dir_city,'train','000000_000000_munster_000105_000004_gtFine_instanceTrainIds.png')

img=Image.open(test_dir)
# 检查图像的通道数
num_channels = len(img.getbands())
print("Number of channels:", num_channels)

# 检查图像的像素值
array = np.array(img)
print(array.shape)
print(array[40:60,40:60])