# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:11:09 2023

@author: seongjoon kang
"""
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_img, features_c, img_size = [64, 50], n_cond=1):
        super(Critic, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(n_cond, 100),
            nn.ReLU(),
            nn.Linear(100,img_size[0]*img_size[1] )
        )
        self.img_size = img_size
        self.critic = nn.Sequential(
            # Input: N x channels
            self._conv_block(channels_img+1, features_c*2, 4, 2, 1), #1/2 [32, 25]     
            self._conv_block(features_c*2, features_c*4, 4, 2, 1), # 1/2 [16, 12]
            self._conv_block(features_c*4, 8*features_c, 4, 2, 1), # 1/2 [8, 6]
            self._conv_block(features_c*8, 16*features_c, 4, 2, 1), #1/2 [4,3]
            self._conv_block(features_c*16,1,  [6,4], 2, 1, disable_norm_act=True, is_bias=True), 
            )
    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding, 
                    disable_norm_act = False, is_bias = False):
        if disable_norm_act is False:
            output =  nn.Sequential(
                nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding, bias= is_bias),
                nn.InstanceNorm2d(out_channels, affine = True), # LayerNorm <-> Instance norm
                nn.LeakyReLU(0.2)
                )
        else:
            output =nn.Sequential(
                nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding, bias= is_bias),\
                    )
        return output
    
    def forward(self, x, cond):
        _cond = self.fc(cond)
        _cond = _cond.reshape(-1,1,self.img_size[0],self.img_size[1])
        x = torch.concat([x,_cond], dim=1)
        
        return self.critic(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, feature_g, n_cond = 1):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._conv_block(z_dim*2, feature_g*16, 3,1,0), #N*f_g  3 * 3 
            self._conv_block(feature_g*16,feature_g*8, [6,4], 2, 1), # 8 * 6
            self._conv_block(feature_g*8,feature_g*4, [4,4], 2, 1), # 16 * 12
            self._conv_block(feature_g*4,feature_g*2, [4,5], 2, 1), #  32 * 25
            self._conv_block(feature_g*2, channels_img,  kernel_size= 4, 
                             stride = 2, padding =1, disable_norm_act=True, is_bias=True),
            nn.Tanh(), #[-1,1]
            )
        self.fc = nn.Sequential(
            nn.Linear(n_cond, z_dim*10),
            nn.ReLU(),
            nn.Linear(z_dim*10, z_dim),
        )
        self.z_dim =z_dim
        
    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding, 
                    disable_norm_act= False, is_bias = False):
        if disable_norm_act is False:
            output = nn.Sequential(
              nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = is_bias,),
              nn.BatchNorm2d(out_channels),
              nn.ReLU())
        else:
            output = nn.Sequential(
              nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = is_bias,),
              )
        return output
    
    def forward(self,x, cond):
        _cond = self.fc(cond) # Batch_N * 20 * 1 * 1
        _cond = _cond.reshape(-1,self.z_dim,1,1)
        x = torch.concat([x,_cond],dim = 1)
        return self.generator(x)
    
def initialize_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.01)
            
def embedder (x, length =10):
    L= -1*(torch.arange(length)-1)
    r_1 = torch.sin(x*torch.pi*2**L)[:,None]
    r_2 = torch.cos(x*torch.pi*2**L)[:,None]

    r_total = torch.concat([r_1, r_2], dim =1)
    return r_total
