import os
from PIL import Image

# target_size of each figure
sample_size=(1024,512) # half   image W H  tensor H W!

# dir to file
dir_data=os.path.abspath("data")
dir_train=os.path.join(dir_data,'cityscapes-dvps','video_sequence','train')
dir_val= os.path.join(dir_data,'cityscapes-dvps','video_sequence','val')

# file to save pre-processed data
dir_train_pp, dir_val_pp = (f'{d}_{sample_size[0]}_{sample_size[1]}' for d in (dir_train,dir_val))

## pre-processing
# create folder to save
if not os.path.exists(dir_train_pp):
    os.makedirs(dir_train_pp)
if not os.path.exists(dir_val_pp):
    os.makedirs(dir_val_pp)

#go throgh all images in train folder
if len(os.listdir(dir_train_pp))==0:
    print('Train set pre-processing starts:..')
    for filename in os.listdir(dir_train): #list of all files and directories in the specified directory
        train_input_path = os.path.join(dir_train,filename)
        train_save_path = os.path.join(dir_train_pp,filename)

        image = Image.open(train_input_path)
        resized_image = image.resize(sample_size,Image.NEAREST)
        resized_image.save(train_save_path,'png',quality = 100)
        image.close()
    print('Train set pre-processing is done')

#go throgh all images in val folder
if len(os.listdir(dir_val_pp))==0:
    print('Val set pre-processing starts:..')
    for filename in os.listdir(dir_val): #list of all files and directories in the specified directory
        val_input_path = os.path.join(dir_val,filename)
        val_save_path = os.path.join(dir_val_pp,filename)

        image = Image.open(val_input_path)
        resized_image = image.resize(sample_size,Image.NEAREST)
        resized_image.save(val_save_path,'png',quality = 100)
        image.close()
    print('Val set pre-processing is done')
print('Peprocess is done')

