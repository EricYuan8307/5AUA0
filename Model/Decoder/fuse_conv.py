import torch.nn as nn

class FuseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FuseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # Reshape input tensor to combine batch and spatial dimensions
        batch_shape, in_channels, height, width = x.size()
        x_reshaped = x.view(batch_shape, -1, height, width)

        # Apply convolution
        output = self.conv(x_reshaped)

        # Reshape output tensor to separate batch and channel dimensions
        out_channels = output.shape[1]
        output_reshaped = output.view(batch_shape, out_channels, height, width)

        return output_reshaped