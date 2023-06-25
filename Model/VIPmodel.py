import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
#from config import Config

class Backbone(nn.Module): # resnet as backbone
    def __init__(self) :
        super().__init__()
        # load pretrained resnet-50
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # load feature drawing part (remove last FC layers)
        # self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])  # * :unpack the list  to make layers as arguments for sequential! list() : is used to let children layers have a sequence, then can do slice to remove last layer
        # -2 cause we need remove average pooling layer!

    def forward(self,x):

        # low level extractor 
        x=self.resnet.conv1(x)
        x=self.resnet.bn1(x)
        x=self.resnet.relu(x)
        z_low=self.resnet.maxpool(x)
        # mid level extractor
        x=self.resnet.layer1(z_low)
        z_mid=self.resnet.layer2(x)
        # high level extractor
        z_high=self.resnet.layer3(z_mid)
        # z_high=self.resnet.layer4(x) # add layer 4 result 1/32, 1048->32, hard to do dilation conv with rate 18

        return z_low,z_mid,z_high # high level feature, low...1/4,mid...1/8, high 1/16


# instance for Backbone
backbone = Backbone()

# freeze parameters in resnet
for param in backbone.parameters():
    param.requires_grad = False



#cfg = Config()

# ##################  test--check shape
# # Generate a random input tensor
# x = torch.randn(1, 3, 2048,2048)
# # Pass the input tensor to the forward() method of the Backbone class
# z_low, z_mid, z_high = backbone(x)
# ## Print the shapes of the output tensors
# print(z_low.shape) #[1, 64, 512, 512]
# print(z_mid.shape) #[1, 512, 256, 256]
# print(z_high.shape) #[1, 1024, 128, 128]
# #################





class ASPP(nn.Module):  # for different instance might use different channels.
    def __init__(self, in_channels,out_channels_1x1=32,out_channels_3x3_1=32,out_channels_3x3_2=32,out_channels_3x3_3=32,out_channels_pool=32):
        super().__init__()
        #  calculate final out channels
        self.final_channels=out_channels_1x1+out_channels_3x3_1+out_channels_3x3_2+out_channels_3x3_3+out_channels_pool
        
        
        # 1x1 convolution layer
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_1x1, kernel_size=1),
            nn.BatchNorm2d(out_channels_1x1),
            nn.ReLU(inplace=True))
        
        # 3x3 dilated convolution layers
        self.conv3x3_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels_3x3_1, kernel_size=3, padding=6, dilation=6,bias=False),
                                       nn.BatchNorm2d(out_channels_3x3_1),
                                       nn.ReLU(inplace=True))
        self.conv3x3_2 = nn.Sequential(nn.Conv2d(in_channels, out_channels_3x3_2, kernel_size=3, padding=12, dilation=12,bias=False),
                                       nn.BatchNorm2d(out_channels_3x3_2),
                                       nn.ReLU(inplace=True))
        self.conv3x3_3 = nn.Sequential(nn.Conv2d(in_channels, out_channels_3x3_3, kernel_size=3, padding=18, dilation=18,bias=False),
                                       nn.BatchNorm2d(out_channels_3x3_3),
                                       nn.ReLU(inplace=True))

        # Image pooling
        self.image_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(in_channels, out_channels_pool, kernel_size=1,bias=False),
                                           nn.BatchNorm2d(out_channels_pool),
                                           nn.ReLU(inplace=True))    
#self.conv_image_pool =
    def forward(self, x):
        # Apply 1x1 convolution
        feat_1x1 = self.conv1x1(x)

        # Apply dilated convolutions
        feat_dilated_1 = self.conv3x3_1(x)
        feat_dilated_2 = self.conv3x3_2(x)
        feat_dilated_3 = self.conv3x3_3(x)

        # Apply image pooling
        feat_image_pool = self.image_pooling(x)
        feat_image_pool = torch.nn.functional.interpolate(feat_image_pool, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Concatenate features along the channel dimension
        out = torch.cat((feat_1x1, feat_dilated_1, feat_dilated_2, feat_dilated_3, feat_image_pool), dim=1)
        return out


# ################ test
# #instance
# aspp=ASPP(in_channels=3)    # a problem is not flexible!
# # Generate a random input tensor
# xx = torch.randn(2, 3, 1024,1024)
# out=aspp(xx)
# print(out.shape) #torch.Size([2, 1280, 1024, 1024])
# ###############





# Cascade-ASPP 4X
class CascadeASPP(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.aspp1=ASPP(in_channels=in_channels  ,out_channels_1x1=16,out_channels_3x3_1=32,out_channels_3x3_2=32,out_channels_3x3_3=32,out_channels_pool=16)
        self.aspp2=ASPP(in_channels=self.aspp1.final_channels  ,out_channels_1x1=50,out_channels_3x3_1=32,out_channels_3x3_2=32,out_channels_3x3_3=32,out_channels_pool=16)
        self.aspp3=ASPP(in_channels=self.aspp1.final_channels+self.aspp2.final_channels  ,out_channels_1x1=16,out_channels_3x3_1=32,out_channels_3x3_2=32,out_channels_3x3_3=32,out_channels_pool=16)
        self.aspp4=ASPP(in_channels=2*self.aspp1.final_channels+self.aspp2.final_channels+self.aspp3.final_channels   ,out_channels_1x1=16,out_channels_3x3_1=32,out_channels_3x3_2=32,out_channels_3x3_3=32,out_channels_pool=16)

        #cause concate in the end , the fianl channels will be like: 4*temp1+2*temp2+temp3+temp4
        self.final_channels = 4*self.aspp1.final_channels+2*self.aspp2.final_channels+self.aspp3.final_channels+self.aspp4.final_channels

    def forward(self,x):
        temp1=self.aspp1(x)
        #print(f'aspp1_outchannels={temp1.shape}') # torch.Size([2, 128, 64, 64])
        temp2=self.aspp2(temp1)
        #print(f'aspp2_outchannels={temp2.shape}') # torch.Size([2, 162, 64, 64]) cause aspp2 channels1x1 is 34 bigger.
        c1=torch.cat((temp1,temp2),dim=1)
        temp3=self.aspp3(c1)
        c2=torch.cat((temp1,c1,temp3),dim=1)
        temp4=self.aspp4(c2)
        out=torch.cat((temp1,c1,c2,temp4),dim=1)
        return out



# ####################### test
# # Initialize network
#
# #instance
# instance_1=CascadeASPP(in_channels=3)
# #instance_1.eval()
# #if cfg.enable_cuda:
# instance_1 = instance_1
# # Generate a random input tensor
# xx = torch.randn(2, 3, 64,64)
# out=instance_1(xx)
# print(out.shape) # torch.Size([2, 1092, 64, 64])
# print(instance_1.final_channels)
# ##################