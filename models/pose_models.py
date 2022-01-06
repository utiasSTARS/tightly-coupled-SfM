'''
Posenet from https://github.com/TRI-ML/packnet-sfm, modified slightly by adding weightnorm alonside groupnorm
'''

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

class conv2d_wn(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv2d_wn, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv2D(nn.Module):
    """
    2D convolution with GroupNorm and ELU
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size
    stride : int
        Stride
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        # self.conv_base = nn.Conv2d(
        #     in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv_base = conv2d_wn(
            in_channels, out_channels, kernel_size, stride=stride)
        self.pad = nn.ConstantPad2d([kernel_size // 2] * 4, value=0)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        """Runs the Conv2D layer."""
        x = self.conv_base(self.pad(x))
        return self.activ(self.normalize(x))
    
'''
PoseNet
'''
########################################################################################################################

def conv_gn(in_planes, out_planes, kernel_size=3):
    """
    Convolutional block with GroupNorm
    Parameters
    ----------
    in_planes : int
        Number of input channels
    out_planes : int
        Number of output channels
    kernel_size : int
        Convolutional kernel size
    Returns
    -------
    layers : nn.Sequential
        Sequence of Conv2D + GroupNorm + ReLU
    """
    return nn.Sequential(
        # nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
        #           padding=(kernel_size - 1) // 2, stride=2),
        conv2d_wn(in_planes, out_planes, kernel_size, padding=(kernel_size-1) // 2, stride=2),
        nn.GroupNorm(16, out_planes),
        nn.ReLU(inplace=True)
    )

########################################################################################################################

class pose_model(nn.Module):
    """Pose network """

    def __init__(self, config, epoch=0, nb_ref_imgs=2, rotation_mode='euler', **kwargs):
        super().__init__()
        self.nb_ref_imgs = 1
        self.rotation_mode = rotation_mode

        conv_channels = [16, 32, 64, 128, 256, 256, 256]
        
        self.config = config
        if self.config['flow_type'] == 'classical':# or self.config['depth_input_to_posenet']==True:
            inputnum = 8
        else:
            inputnum = 6
        
        
        self.conv1 = conv_gn(inputnum, conv_channels[0], kernel_size=7)
        self.conv2 = conv_gn(conv_channels[0], conv_channels[1], kernel_size=5)
        self.conv3 = conv_gn(conv_channels[1], conv_channels[2])
        self.conv4 = conv_gn(conv_channels[2], conv_channels[3])
        self.conv5 = conv_gn(conv_channels[3], conv_channels[4])
        self.conv6 = conv_gn(conv_channels[4], conv_channels[5])
        self.conv7 = conv_gn(conv_channels[5], conv_channels[6])

        self.pose_pred = nn.Conv2d(conv_channels[6], 6 * self.nb_ref_imgs,
                                   kernel_size=1, padding=0)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, imgs, return_features=False):
        # if self.config['flow_type'] == 'none':
        #     imgs = imgs[0:2] # get rid third img (the optical flow image) if not wanted
        # imgs = torch.cat(imgs,1)
        imgs = (imgs - 0.45)/0.22
        out_conv1 = self.conv1(imgs)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), 6)
        
        if return_features == True:
            img_features = [out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6, out_conv7]
            # print([c.shape for c in img_features])
            return pose, img_features
        else:        
            return pose