from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from Model.VIPmodel import ASPP, backbone, CascadeASPP
from Model.Decoder.conv import stacked_conv
# from Model.Decoder.fuse_conv import FuseConv

class SingleSemanticDecoder(nn.Module):
    def __init__(self, in_channel = 1024, # channel number from encoder to ASPP
                 in_decoder_channel = 160, # from ASPP to Decoder
                 input_level_channel= 256, # 1/16层的参数
                 low_level_channel_input = 512, low_level_channel_output = 64, # 1/8层参数
                 mid_level_channel_input = 64, mid_level_channel_output = 32, # 1/4层参数
                 decoder_out_channel=256
                 ):
        super(SingleSemanticDecoder, self).__init__()

        self.aspp = ASPP(in_channel) #ASPP parameters
        # # self.decoder_in_channel = decoder_in_channel #the input parameter from ASPP to Decoder
        #
        # self.input_level_channel = input_level_channel #256
        #
        # self.low_level_channel_input = low_level_channel_input #256
        # self.low_level_channel_output = low_level_channel_output # 64
        # # self.low_level_channel_project = low_level_channel_project  # the channel number from encoder added in 1/8层参数
        #
        # self.mid_level_channel_input = mid_level_channel_input #256
        # self.mid_level_channel_output = mid_level_channel_output #32
        # # self.mid_level_channel_project = mid_level_channel_project # the channel number from encoder added in 1/4层参数
        #
        # self.decoder_out_channel = decoder_out_channel # the decoder output result
        #
        #
        # # fuse
        fuse_in_channel1 = input_level_channel + low_level_channel_output # 256+64
        fuse_in_channel2 = input_level_channel + mid_level_channel_output # 256+32
        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                                 conv_type='depthwise_separable_conv')  # 1*1 conv unsampling
        self.fuse_conv1 = self.fuse_conv(fuse_in_channel1, decoder_out_channel)
        self.fuse_conv2 = self.fuse_conv(fuse_in_channel2, decoder_out_channel)

        # self.fuse_conv1 = FuseConv(fuse_in_channel1, decoder_out_channel, kernel_size=5, stride=1, padding=2)
        # self.fuse_conv2 = FuseConv(fuse_in_channel2, decoder_out_channel, kernel_size=5, stride=1, padding=2)

        # Transform low-level feature

        # from aspp to 1/16: add the channel with 1*1 conv
        self.conv0 = nn.Conv2d(in_decoder_channel, input_level_channel, 1, bias=False)  # 1x1 convolution model to decoder
        self.norm0 = nn.BatchNorm2d(input_level_channel)
        self.relu0 = nn.ReLU()

        # # 1/8 step: add the channel with 1*1 conv
        self.conv1 = nn.Conv2d(low_level_channel_input, low_level_channel_output, 1, bias=False) # 1x1 convolution model to decoder
        self.norm1 = nn.BatchNorm2d(low_level_channel_output)
        self.relu1 = nn.ReLU()

        # 1/4 step: add the channel with 1*1 conv
        self.conv2 = nn.Conv2d(mid_level_channel_input, mid_level_channel_output, 1, bias=False)# 1x1 convolution model to decoder
        self.norm2 = nn.BatchNorm2d(mid_level_channel_output)
        self.relu2 = nn.ReLU()

    def forward(self, z_high, z_mid, z_low):
        x = self.aspp(z_high)
        # print(x.shape) # torch.Size([2, 160, 128, 128])

        # from 1/16 to 1/8
        x = self.conv0(x) #From ASPP to Decoder
        x = self.norm0(x)
        x = self.relu0(x) # after the frust 1X1 Conv, next to upsample. # torch.Size([2, 256, 128, 128])
        x0 = F.interpolate(x, size=z_mid.size()[2:], mode='bilinear', align_corners=True) # 1/16 step and get to upsample # torch.Size([2, 256, 256, 256])
        # x0 = self.fuse_conv(256, self.fuse_in_channel1)(x0)  # fuse itself

        # start from 1/8 to 1/4:
        x = self.conv1(z_mid) # from encoder to decoder (Red Arrow) # torch.Size([2, 64, 256, 256])
        x = self.norm1(x)
        x = self.relu1(x)
        x = torch.cat((x0, x), dim=1) #concat # torch.Size([5, 320, 256, 256])
        x1 = self.fuse_conv1(x) # 5*5 conv upsampling # torch.Size([5, 256, 256, 256])

        # # start from 1/4 to outside:
        x = self.conv2(z_low)  # from encoder to decoder (Blue Arrow) # torch.Size([2, 32, 512, 512])
        x = self.norm2(x)
        x = self.relu2(x)
        x2 = F.interpolate(x1, size=z_low.size()[2:], mode='bilinear', align_corners=True) # torch.Size([2, 256, 512, 512])
        x = torch.cat((x2, x), dim=1)  # concat # torch.Size([2, 288, 512, 512])
        x = self.fuse_conv2(x)  # 5*5 conv unsampling. 288是concat之后的层数, 256是最后一层输出的channel数.


        return x

# instance_1 = SingleSemanticDecoder()
#
# # Generate a random input tensor
# x = torch.randn(5, 3, 1024, 2048)
# # Pass the input tensor to the forward() method of the Backbone class
# z_low, z_mid, z_high = backbone(x)
# # Print the shapes of the output tensors
# print(z_low.shape) # torch.Size([5, 64, 256, 512])
# print(z_mid.shape) # torch.Size([5, 512, 128, 256])
# print(z_high.shape) # torch.Size([5, 1024, 64, 128])
#
# out=instance_1(z_high, z_mid, z_low)
# print(out.shape) # torch.Size([5, 256, 256, 512])
# #################

# config:
#   low_channles=64
#   mid_channles=512
#   high_channles=1024

class SingleInstanceDecoder(nn.Module):
    def __init__(self, in_aspp_channel = 1024, # channel number from encoder to ASPP
                 in_decoder_channel = 160, # from ASPP to Decoder
                 input_level_channel_1= 256, # 1/16层的参数
                 low_level_channel_input=512, low_level_channel_output=32,  # 1/8层参数
                 input_level_channel_2=128 ,mid_level_channel_input=64, mid_level_channel_output=16, # 1/4层参数
                 decoder_out_channel=128
                 ):
        super(SingleInstanceDecoder, self).__init__()

        self.aspp = ASPP(in_aspp_channel) #ASPP parameters
        # # self.decoder_in_channel = decoder_in_channel #the input parameter from ASPP to Decoder
        #
        # self.input_level_channel = input_level_channel #256
        #
        # self.low_level_channel_input = low_level_channel_input #256
        # self.low_level_channel_output = low_level_channel_output # 64
        # # self.low_level_channel_project = low_level_channel_project  # the channel number from encoder added in 1/8层参数
        #
        # self.mid_level_channel_input = mid_level_channel_input #256
        # self.mid_level_channel_output = mid_level_channel_output #32
        # # self.mid_level_channel_project = mid_level_channel_project # the channel number from encoder added in 1/4层参数
        #
        # self.decoder_out_channel = decoder_out_channel # the decoder output result
        #
        #
        # # fuse
        fuse_in_channel1 = input_level_channel_1 + low_level_channel_output # 256+32
        fuse_in_channel2 = input_level_channel_2 + mid_level_channel_output # 128+16
        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')  # 1*1 conv unsampling
        self.fuse_conv1 = self.fuse_conv(fuse_in_channel1, decoder_out_channel)
        self.fuse_conv2 = self.fuse_conv(fuse_in_channel2, decoder_out_channel)

        # Transform low-level feature

        # from aspp to 1/16: add the channel with 1*1 conv
        self.conv0 = nn.Conv2d(in_decoder_channel, input_level_channel_1, 1, bias=False)  # 1x1 convolution model to decoder
        self.norm0 = nn.BatchNorm2d(input_level_channel_1)
        self.relu0 = nn.ReLU()

        # # 1/8 step: add the channel with 1*1 conv
        self.conv1 = nn.Conv2d(low_level_channel_input, low_level_channel_output, 1, bias=False) # 1x1 convolution model to decoder
        self.norm1 = nn.BatchNorm2d(low_level_channel_output)
        self.relu1 = nn.ReLU()

        # 1/4 step: add the channel with 1*1 conv
        self.conv2 = nn.Conv2d(mid_level_channel_input, mid_level_channel_output, 1, bias=False)# 1x1 convolution model to decoder
        self.norm2 = nn.BatchNorm2d(mid_level_channel_output)
        self.relu2 = nn.ReLU()

    def forward(self, z_high, z_mid, z_low):
        x = self.aspp(z_high)
        # print(x.shape) # torch.Size([2, 160, 128, 128])

        # from 1/16 to 1/8
        x = self.conv0(x) #From ASPP to Decoder
        x = self.norm0(x)
        x = self.relu0(x) # after the frust 1X1 Conv, next to upsample. # torch.Size([2, 256, 128, 128])
        x0 = F.interpolate(x, size=z_mid.size()[2:], mode='bilinear', align_corners=True) # 1/16 step and get to upsample # torch.Size([2, 256, 256, 256])
        # x0 = self.fuse_conv(256, self.fuse_in_channel1)(x0)  # fuse itself

        # start from 1/8 to 1/4:
        x = self.conv1(z_mid) # from encoder to decoder (Red Arrow) # torch.Size([2, 32, 256, 256])
        x = self.norm1(x)
        x = self.relu1(x)
        x = torch.cat((x0, x), dim=1) #concat # torch.Size([2, 288, 256, 256])
        x1 = self.fuse_conv1(x) # 5*5 conv upsampling # torch.Size([2, 128, 256, 256])

        # # start from 1/4 to outside:
        x = self.conv2(z_low)  # from encoder to decoder (Blue Arrow) # torch.Size([2, 16, 512, 512])
        x = self.norm2(x)
        x = self.relu2(x)
        x2 = F.interpolate(x1, size=z_low.size()[2:], mode='bilinear', align_corners=True) # torch.Size([2, 128, 512, 512])
        x = torch.cat((x2, x), dim=1)  # concat # torch.Size([2, 144, 512, 512])
        x = self.fuse_conv2(x)  # 5*5 conv unsampling. 288是concat之后的层数, 256是最后一层输出的channel数.

        return x

# instance_11 = SingleInstanceDecoder()
# #
# # Generate a random input tensor
# x = torch.randn(2, 3, 2048, 2048)
# # # Pass the input tensor to the forward() method of the Backbone class
# z_low, z_mid, z_high = backbone(x)
# # ## Print the shapes of the output tensors
# # print(z_low.shape) #[2, 64, 512, 512]
# # print(z_mid.shape) #[2, 512, 256, 256]
# # print(z_high.shape) #[2, 1024, 128, 128]
#
# out=instance_11(z_high, z_mid, z_low)
# print(out.shape) # torch.Size([2, 128, 512, 512])

class NextInstanceDecoder(nn.Module):
    def __init__(self, in_aspp_channel = 2048, # channel number from encoder to ASPP
                 in_decoder_channel = 1092, # from ASPP to Decoder
                 input_level_channel_1= 256, # 1/16层的参数
                 low_level_channel_input=512, low_level_channel_output=32,  # 1/8层参数
                 input_level_channel_2=128 ,mid_level_channel_input=64, mid_level_channel_output=16, # 1/4层参数
                 decoder_out_channel=128
                 ):
        super(NextInstanceDecoder, self).__init__()

        self.aspp = CascadeASPP(in_aspp_channel) #ASPP parameters
        # # self.decoder_in_channel = decoder_in_channel #the input parameter from ASPP to Decoder
        #
        # self.input_level_channel = input_level_channel #256
        #
        # self.low_level_channel_input = low_level_channel_input #256
        # self.low_level_channel_output = low_level_channel_output # 64
        # # self.low_level_channel_project = low_level_channel_project  # the channel number from encoder added in 1/8层参数
        #
        # self.mid_level_channel_input = mid_level_channel_input #256
        # self.mid_level_channel_output = mid_level_channel_output #32
        # # self.mid_level_channel_project = mid_level_channel_project # the channel number from encoder added in 1/4层参数
        #
        # self.decoder_out_channel = decoder_out_channel # the decoder output result
        #
        #
        # # fuse
        fuse_in_channel1 = input_level_channel_1 + low_level_channel_output # 256+32
        fuse_in_channel2 = input_level_channel_2 + mid_level_channel_output # 128+16
        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')  # 1*1 conv unsampling
        self.fuse_conv1 = self.fuse_conv(fuse_in_channel1, decoder_out_channel)
        self.fuse_conv2 = self.fuse_conv(fuse_in_channel2, decoder_out_channel)

        # Transform low-level feature

        # from aspp to 1/16: add the channel with 1*1 conv
        self.conv0 = nn.Conv2d(in_decoder_channel, input_level_channel_1, 1, bias=False)  # 1x1 convolution model to decoder
        self.norm0 = nn.BatchNorm2d(input_level_channel_1)
        self.relu0 = nn.ReLU()

        # # 1/8 step: add the channel with 1*1 conv
        self.conv1 = nn.Conv2d(low_level_channel_input, low_level_channel_output, 1, bias=False) # 1x1 convolution model to decoder
        self.norm1 = nn.BatchNorm2d(low_level_channel_output)
        self.relu1 = nn.ReLU()

        # 1/4 step: add the channel with 1*1 conv
        self.conv2 = nn.Conv2d(mid_level_channel_input, mid_level_channel_output, 1, bias=False)# 1x1 convolution model to decoder
        self.norm2 = nn.BatchNorm2d(mid_level_channel_output)
        self.relu2 = nn.ReLU()

    def forward(self, z_high, z_mid, z_low):
        x = self.aspp(z_high)
        # print(x.shape) # torch.Size([2, 160, 128, 128])

        # from 1/16 to 1/8
        x = self.conv0(x) #From ASPP to Decoder
        x = self.norm0(x)
        x = self.relu0(x) # after the frust 1X1 Conv, next to upsample. # torch.Size([2, 256, 128, 128])
        x0 = F.interpolate(x, size=z_mid.size()[2:], mode='bilinear', align_corners=True) # 1/16 step and get to upsample # torch.Size([2, 256, 256, 256])
        # x0 = self.fuse_conv(256, self.fuse_in_channel1)(x0)  # fuse itself

        # start from 1/8 to 1/4:
        x = self.conv1(z_mid) # from encoder to decoder (Red Arrow) # torch.Size([2, 32, 256, 256])
        x = self.norm1(x)
        x = self.relu1(x)
        x = torch.cat((x0, x), dim=1) #concat # torch.Size([2, 288, 256, 256])
        x1 = self.fuse_conv1(x) # 5*5 conv upsampling # torch.Size([2, 128, 256, 256])

        # # start from 1/4 to outside:
        x = self.conv2(z_low)  # from encoder to decoder (Blue Arrow) # torch.Size([2, 16, 512, 512])
        x = self.norm2(x)
        x = self.relu2(x)
        x2 = F.interpolate(x1, size=z_low.size()[2:], mode='bilinear', align_corners=True) # torch.Size([2, 128, 512, 512])
        x = torch.cat((x2, x), dim=1)  # concat # torch.Size([2, 144, 512, 512])
        x = self.fuse_conv2(x)  # 5*5 conv unsampling. 288是concat之后的层数, 256是最后一层输出的channel数.

        return x

# instance_12 = SingleInstanceDecoder()
# #
# # Generate a random input tensor
# x = torch.randn(2, 3, 2048, 2048)
# # # Pass the input tensor to the forward() method of the Backbone class
# z_low, z_mid, z_high = backbone(x)
# # ## Print the shapes of the output tensors
# # print(z_low.shape) #[2, 64, 512, 512]
# # print(z_mid.shape) #[2, 512, 256, 256]
# # print(z_high.shape) #[2, 1024, 128, 128]
#
# out=instance_12(z_high, z_mid, z_low)
# print(out.shape) # torch.Size([2, 128, 512, 512])

class GenerateHead(nn.Module):
    def __init__(self, num_class, # should be determined
                 decoder_channel = 256, #channel number from decoder
                 head_channel = 256 # Middle channel number
                 ):
        super(GenerateHead, self).__init__()
        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                                 conv_type='depthwise_separable_conv')  # 5X5 Conv
        self.fuse_con = self.fuse_conv(decoder_channel, head_channel)

        self.conv = nn.Conv2d(head_channel, num_class, 1, bias=False)  # 1X1 conv

    def forward(self, x):
        x = self.fuse_con(x)
        x = F.interpolate(x, size=(4*x.shape[2], 4*x.shape[3]), mode='bilinear',
                           align_corners=True)  # torch.Size([2, 32, 1024, 1024])
        x = self.conv(x)

        return x

# decoder_channel = 256 #channel number from decoder
# head_channel = 256 # Middle channel number

# instance_1 = SingleSemanticDecoder()
# instance_2 = GenerateHead(num_class = 2) # should be determined
#
# x = torch.randn(2, 3, 1024, 2048)
# z_low, z_mid, z_high = backbone(x)
# xx=instance_1(z_high, z_mid, z_low)
# out=instance_2(xx)
# print(out.shape) # torch.Size([2, 2, 1024, 2048])


class DepthPredictionHead(nn.Module):
    def __init__(self,
                 decoder_channel = 256, # The input channel size from Semantic decoder
                 head_channel_1 = 32, # The first layer after 5X5 Conv
                 head_channel_2 = 64, # The second layer after 3X3 Conv
                 Dout_channel = 1, # The last output of Depth Prediction
                 upsample_depth = (1024, 1024)
                 ):
        super(DepthPredictionHead, self).__init__()

        self.fuse_conv1 = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                                  conv_type='depthwise_separable_conv')  # 5X5 Conv
        self.fuse_con1 = self.fuse_conv1(decoder_channel, head_channel_1)

        self.fuse_conv2 = partial(stacked_conv, kernel_size=3, num_stack=1, padding=1,
                                  conv_type='depthwise_separable_conv')  # 3X3 Conv Should be discussed
        self.fuse_con2 = self.fuse_conv2(head_channel_1, head_channel_2)

        self.conv = nn.Conv2d(head_channel_2, Dout_channel, 1, bias=False)  # 1X1 conv

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fuse_con1(x)
        x = F.interpolate(x, size=(4*x.shape[2], 4*x.shape[3]), mode='bilinear',
                           align_corners=True)  # torch.Size([2, 32, 1024, 1024])
        x = self.fuse_con2(x)  # torch.Size([2, 64, 1024, 1024])
        x = self.conv(x)

        return x


# instance_1 = SingleSemanticDecoder()
# instance_3 = DepthPredictionHead()
#
# x = torch.randn(2, 3, 1024, 2048)
# z_low, z_mid, z_high = backbone(x)
# xx=instance_1(z_high, z_mid, z_low) # torch.Size([2, 256, 512, 512])
# out=instance_3(xx) # torch.Size([2, 1, 1024, 2048])
# # print(out.shape)


'''
    Designed three different decoder class for  training.
'''

class SemanticDecoder(nn.Module):
    def __init__(self,):
        super(SemanticDecoder, self).__init__()
        self.semantic_decoder = SingleSemanticDecoder()
        self.semantic_head = GenerateHead(num_class = 19) # Need to discuss and Mind! ignore index 32, form 0 to 31= 32 classes
        self.depth_head = DepthPredictionHead()

    def forward(self, z_high, z_mid, z_low):
        # Semantic Depth
        semantic = self.semantic_decoder(z_high, z_mid, z_low)
        depth_prediction = self.depth_head(semantic)
        semantic_prediction = self.semantic_head(semantic)
        pred_depth = depth_prediction
        pred_sematic = semantic_prediction

        return pred_depth, pred_sematic

# x = torch.randn(5, 3, 1024, 2048)
# z_low, z_mid, z_high = backbone(x)
# instance_4 = SemanticDecoder()
# depth_prediction, semantic_prediction = instance_4(z_high, z_mid, z_low)
# print(depth_prediction.shape) # torch.Size([5, 1, 1024, 2048])
# print(semantic_prediction.shape) # torch.Size([5, 32, 1024, 2048])


class InstanceDecoder(nn.Module):
    def __init__(self,):
        super(InstanceDecoder, self).__init__()
        #Build Instance Decoder
        # self.aspp_channels = ASPP(in_channels=1024)
        self.instance_decoder = SingleInstanceDecoder(low_level_channel_output = 32, mid_level_channel_output = 16)
        self.instance_head_pre = GenerateHead(num_class=1, decoder_channel = 128)
        self.instance_head_reg = GenerateHead(num_class=2, decoder_channel=128)

    def forward(self, z_high, z_mid, z_low):
        # instance center
        instance = self.instance_decoder(z_high, z_mid, z_low)
        instance_prediction = self.instance_head_pre(instance)
        instance_regression = self.instance_head_reg(instance)

        return instance_prediction, instance_regression

# x = torch.randn(2, 3, 1024, 2048)

# instance_4 = InstanceDecoder()
# instance_prediction, instance_regression = instance_4(x)
# print(instance_prediction.shape) #torch.Size([2, 1, 1024, 2048])
# print(instance_regression.shape) #torch.Size([2, 2, 1024, 2048])

class NextFrameDecoder(nn.Module):
    def __init__(self,fuse_in_channel = 2048, fuse_out_channel = 1024):
        super(NextFrameDecoder, self).__init__()
        #Build Instance Decoder
        # self.aspp_channels = ASPP(in_channels=1024)
        self.backbone = backbone
        self.instance_decoder = NextInstanceDecoder(low_level_channel_output=32, mid_level_channel_output=16)
        self.instance_head = GenerateHead(num_class=2, decoder_channel=128)

    def forward(self, z_high, z_mid1, z_low1):
        # Next-frame regression
        instance = self.instance_decoder(z_high, z_mid1, z_low1)
        next_regression = self.instance_head(instance)

        return next_regression

# x = torch.randn(5, 3, 1024, 2048)
# xx = torch.randn(5, 3, 1024, 2048)
#
# instance_14 = NextFrameDecoder()
# next_instance_regression = instance_14(x, xx)
# print(next_instance_regression.shape) # torch.Size([5, 2, 1024, 2048])


class DecoderArch(nn.Module):
    def __init__(self,):
        super(DecoderArch, self).__init__()
        self.backbone = backbone
        self.Semantic = SemanticDecoder()
        self.Instance = InstanceDecoder()
        self.NextFrameInstance = NextFrameDecoder()

    def forward(self, featuresT0, featuresT1):
        z_low0, z_mid0, z_high0 = self.backbone(featuresT0)  # z_high0 = torch.Size([5, 1024, 64, 128])
        z_low1, z_mid1, z_high1 = self.backbone(featuresT1)  # z_high1 = torch.Size([5, 1024, 64, 128])
        z_high = torch.cat((z_high0, z_high1), dim=1)  # torch.Size([5, 2048, 64, 128])


        depth_prediction, semantic_prediction = self.Semantic(z_high0, z_mid0, z_low0)
        instance_prediction, instace_regression = self.Instance(z_high0, z_mid0, z_low0)
        next_frame_instance = self.NextFrameInstance(z_high, z_mid1, z_low1)

        return depth_prediction, semantic_prediction, instance_prediction, instace_regression, next_frame_instance

'''
Decoder Arch:
Input: Frame T data, Frame T+1 data
Output: 
        Depth prediction, 
        Semantic prediction, 
        Instace center prediction, 
        Instance center regression, 
        Next-frame instance center regression
'''

# x = torch.randn(2, 3, 1024, 2048)
# xx = torch.randn(2, 3, 1024, 2048)
#
# instance_15 = DecoderArch()
#
# depth_prediction, semantic_prediction, instance_prediction, instace_regression, next_frame_instance = instance_15(x, xx)
#
# print(depth_prediction.shape) # torch.Size([5, 1, 1024, 2048])
# print(semantic_prediction.shape) # torch.Size([5, 19, 1024, 2048])
# print(instance_prediction.shape) # torch.Size([5, 1, 1024, 2048])
# print(instace_regression.shape) # torch.Size([5, 2, 1024, 2048])
# print(next_frame_instance.shape) # torch.Size([5, 2, 1024, 2048])