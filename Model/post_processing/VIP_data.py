from dataclasses import dataclass
from torch.utils.data import Dataset
import re
import os
from PIL import Image
from torchvision.io import read_image
import torch
import numpy as np

# Each sample we downloaded can be identified by the name of the city as well as a frame and sequence id
@dataclass
class CityscapesSample:
    video_id:str
    frame_id: str
    city: str
    seq1_num: str # some unclear number
    seq2_num: str

    @property
    def id(self):
        return os.path.join("_".join([self.video_id,self.frame_id,self.city,self.seq1_num,self.seq2_num]))

    @staticmethod
    def from_filename(filename: str):
        # Create a CityscapesSample from a filename, which has a fixed structure {city}_{sequence}_{frame}
        match = re.match(r"^(\d+)_(\d+)_(\w+)_(\d+)_(\d+).*.png$", filename, re.I)
        return CityscapesSample(match.group(1), match.group(2), match.group(3),match.group(4),match.group(5))


class MyDataset(Dataset):
    def __init__(self,gt_dir,split='train_1024_512'):  # gt_dir is path for "video_sequence"
        self.dir_input=os.path.join(gt_dir,split) # folder path of all images, cause gtFine, left.. are in the same folder
        
        # Walk through the inputs directory and add each file to our items list
        self.items = []
        for (_, _, filenames) in os.walk(self.dir_input):
            self.items.extend([CityscapesSample.from_filename(f) for f in filenames])
            # extend is add one by one as elements . append is add all as one element
        
        # Sanity check: do the provided directories contain any samples?
        assert len(self.items) > 0, f"No items found in {self.dir_input}"

    def __len__(self):  # will influence the maximum index request from dataloader.
        return len(self.items)//3-2 # all items in our folder, all lenght?  -2 beacuse the last pair is t and t+1,  limit t+1 = last group. # -1 will out of index, don't know why
    
    # get tensor result!
    def get_item_single(self, index):

        sample=self.items[3*index] # cause consecutive 3 images have same id except suffix(后缀) !!!!!! my id do not have suffix , so!
        input = self.load_input(sample)
        truth_class,truth_ins = self.load_truth(sample)
        truth_depth = self.load_depth(sample)
        
        # convert from image array to tensor
        #input=torch.as_tensor(input,dtype=torch.float)
        truth_class=torch.as_tensor(truth_class,dtype=torch.float)
        truth_ins = torch.as_tensor(truth_ins,dtype=torch.float)
        truth_depth = torch.as_tensor(truth_depth,dtype=torch.float)

        return input,truth_class,truth_ins,truth_depth

    def __getitem__(self, index):
        items_input = []
        items_truth_class=[]
        items_truth_ins=[]
        items_truth_depth=[]


        for i in range(2):      
            input,truth_class,truth_ins,truth_depth = self.get_item_single(index=index+i)
            items_input.append(input)
            items_truth_class.append(truth_class)
            items_truth_ins.append(truth_ins)
            items_truth_depth.append(truth_depth)

        stacked_input=torch.stack(items_input,dim=0)    # run map if use stack,     each of the item ,  [2,3,H,W]
        stacked_truth_class= torch.stack(items_truth_class,dim=0) # [2,H,W]
        stacked_truth_ins= torch.stack(items_truth_ins,dim=0) # [2,H,W]
        stacked_truth_depth = torch.stack(items_truth_depth,dim=0) # [2,H,W]

        # return indicate of throwing this pair or not
        if index%6 == 5: # 0 1 2 3 4 5|  6 7 8 9 10 11| 10..    [4 5] [9 10] cannot be a pair!  let them become full of -1 ,avoid empty instance map,# ! images use 0 to indicate throw away
            return torch.zeros_like(stacked_input),torch.zeros_like(stacked_truth_class)-1 ,torch.zeros_like(stacked_truth_ins)-1,torch.zeros_like(stacked_truth_depth)-1  # dataloader will skip none
        
        return stacked_input,stacked_truth_class,stacked_truth_ins,stacked_truth_depth
    



    # find input images
    def load_input(self,sample:CityscapesSample):
        path = os.path.join(self.dir_input,f'{sample.id}_leftImg8bit.png')
        return read_image(path)
    
    # find ground truth images gtFine( later substract class id and instance id!)
    def load_truth(self,sample:CityscapesSample):
        path = os.path.join(self.dir_input,f'{sample.id}_gtFine_instanceTrainIds.png')
        # read_image return array results
        truth_class= np.array(Image.open(path))//1000 # [ class_id * 1000 + instance_id = pixel value ]
        truth_ins= np.array(Image.open(path))% 1000  # instance 
        return truth_class,truth_ins
    
    def load_depth(self,sample:CityscapesSample):
        path = os.path.join(self.dir_input,f'{sample.id}_depth.png')
        return  np.array(Image.open(path))
    


#### should I add transform (data augment)here?