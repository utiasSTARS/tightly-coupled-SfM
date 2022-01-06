'''
modified depth network that has extractable outputs from within the network
'''


from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from torch.nn import init
import numpy as np
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from models.depth_models import Conv3x3, Interpolate, ResNetMultiImageInput, resnet_multiimage_input, ResnetEncoder

class depth_model(nn.Module):
    def __init__(self, config, nb_ref_imgs=1): 
        super(depth_model, self).__init__()
        num_img_channels = 3
        self.num_scales = config['num_scales']
        self.nb_ref_imgs=nb_ref_imgs

        ## Encoder Layers
        self.encoder = ResnetEncoder(18,True)
        ## Upsampling
        upconv_planes = [256, 128, 64, 64,32]
        upconv_planes2 = [512]+ upconv_planes
        self.depth_upconvs = nn.ModuleList([self.upconv(upconv_planes2[i],upconv_planes2[i+1]) for i in range(0,len(upconv_planes))]) 
        self.iconvs = nn.ModuleList([self.conv(upconv_planes[i], upconv_planes[i]) for i in range(0,len(upconv_planes))])   
        
        disp_feature_sizes = list(np.cumsum(self.num_scales*[8]))
        self.feature_convs = nn.ModuleList([self.conv(s,8) for s in upconv_planes[-self.num_scales:]])
        self.predict_disps = nn.ModuleList([self.predict_disp(disp_feature_sizes[i]) for i in range(0,self.num_scales)] ) 


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x=None, skips=None, return_disp=True, epoch=0):
        
        ## fwd pass through encoder if x an image input is given
        if x is not None:
            x = (x - 0.45) / 0.22
            skips = self.encoder(x)
            if return_disp == False:
                return None, skips
                
        ''' Depth UpSampling (depth upconv, and then out_iconv)'''
        out_iconvs = [skips[-1]]
        disps = []

        ## test
        depth_features = []
        for i in range(0,len(self.iconvs)-1):
            depth_features.append(out_iconvs[-1])             
            upconv = self.depth_upconvs[i](out_iconvs[-1])
            upconv = upconv + skips[-(i+2)]
            out_iconvs.append( self.iconvs[i](upconv) )

        depth_features.append(out_iconvs[-1])
        upconv = self.depth_upconvs[-1](out_iconvs[-1]) #final layer is different, so is out of loop
        out_iconv = self.iconvs[-1](upconv)
        depth_features.append(out_iconv)

        depth_features = depth_features[-self.num_scales:]
        
        #reduce # channels to 8 before merging
        for i in range(0, self.num_scales):
            depth_features[i] = self.feature_convs[i](depth_features[i])

        

        #merge features from all scales for depth prediction (upsize smaller scales before merging)
        concat_depth_features = []
        concat_depth_features.append(depth_features[-self.num_scales])

        for i in np.arange(self.num_scales-1, 0, -1):
            upsized = []
            _, _, h, w = depth_features[-i].size()
            
            for j in range(0, self.num_scales - i):
                upsized.append(nn.functional.interpolate(depth_features[j], (h, w), mode='nearest') )
            upsized.append(depth_features[-i])
            concat_depth_features.append(torch.cat(upsized,1))

        for i in np.arange(self.num_scales,0,-1):  
            disps.append(self.predict_disps[-i](concat_depth_features[-i]))
            
        disps.reverse()
        return list(disps[0:self.num_scales]), skips

    
    def predict_disp(self, in_planes, kernel_size=3):
        return nn.Sequential(
            Conv3x3(in_planes, self.nb_ref_imgs, use_refl=True, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            # nn.Conv2d(in_planes, self.nb_ref_imgs, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            # nn.ReLU()
            nn.Sigmoid()
        )
        
    def upconv(self, in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
            Interpolate(scale_factor=2, mode ='nearest'),
            Conv3x3(in_planes,  out_planes,  use_refl=True, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ELU(inplace=True)
        )        


    def conv(self, in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
            Conv3x3(in_planes, out_planes, use_refl=True, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ELU(inplace=True),
        )

