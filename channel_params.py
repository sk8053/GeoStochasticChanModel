# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:23:37 2023

@author: seongjoon kang
"""
import torch
import scipy.constants as cs
import json
from c_wgan import Generator
import numpy as np

class Channel_Params:
    
    def __init__(self, 
                 saved_path = 'save_model',
                 data_dir = 'Boston_data',
                 z_dim = 20, 
                 generator_features = 32,
                 n_cond = 2,
                 path_loss_threshold = 180,
                 device = 'cpu'):
        
        saved_file_name = f'cwgan_z_{z_dim}.pt'
        ## load generator from trained WGAN model
        model_dict = torch.load(f'{saved_path}/{saved_file_name}')
        gen_state_dict = model_dict['gen']
        self.gen = Generator(z_dim, 1, generator_features, n_cond = n_cond).to(device)
        self.gen.load_state_dict(gen_state_dict)
        self.gen.eval()
        
        ## load max and min values of the original data 
        with open(f'{data_dir}/max_min.json', 'r') as out:
            self.max_min = json.load(out)
            
        self.feature_dict = {'path_loss':0, 'delay':1, 'zoa':2, 'aoa':3, 
                             'zod':4, 'aod':5, 'phase':6, 'link state':7}
        self.device = device
        self.z_dim = z_dim
        self.path_loss_threshold = path_loss_threshold
        
    @staticmethod
    def get_channel_dict(data_mat, path_loss_threshold=180, feature_dict = None):
        '''
        Parameters
        ----------
        data_mat : matrix (batch_size, number of features, maximum number of multi-paths)
            the multi-path channel parameters            
        Returns
        -------
        chan_dict_list : list of dictionary
            multi-path channel dictionaries
        '''
        
        batch_size = data_mat.shape[0]
        keys = list(feature_dict.keys())
        L = len(keys) # total number of path components
        chan_dict_list = []
        for j in range(batch_size):
            chan_dict = dict()
            
            dly_ = data_mat[j,1]
            # sort multi-path components based on delay
            I2 = np.argsort(dly_) 
            # cut multipath based on pathloss threshold
            I = data_mat[j,0][I2]<path_loss_threshold
            
            for k in range(L):
                
                if keys[k] == 'link state':
                    ls = np.mean(data_mat[j,k])
                    if ls >0:
                        chan_dict[keys[k]] = [1]
                    else:
                        chan_dict[keys[k]] = [-1]
                else:
                    # order multipath channel params based on delay and 
                    # take only paths satisfying the pathloss threshold
                    mpath_comp = data_mat[j,k][I2][I]
                    chan_dict[keys[k]] = list(mpath_comp)
                    
            chan_dict_list.append(chan_dict)
            
        return chan_dict_list
    
    def get_chan_params(self, dist2d, height):
        '''
        Parameters
        ----------
        dist2d : array, shape = (batch_size, )
            2D distance between Tx and Rx
        height : scalar, height of Rx

        Returns
        -------
        data: multipath channel dictionaries
        '''
        
        batch_size = len(dist2d)
        dist2d = torch.tensor(dist2d, dtype = torch.float32)[:,None]
        heights = torch.repeat_interleave(torch.tensor([height]), batch_size, dim = 0)
        heights = heights[:,None]
        
        condition = torch.concat([dist2d, heights], dim =1).to(self.device)
        noise = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)
        
        fake = self.gen(noise, condition)
        fake = fake.detach().cpu().numpy().squeeze()
        # downsize the matrix, fake
        # sample fake
        sampled_fake1 = np.zeros((batch_size, 8,50))
        for i in range(8):
            sampled_fake1[:,i] = fake[:,i*8+3]
        sampled_fake2 = np.zeros((batch_size, 8,25))
        for j in range(25):
            sampled_fake2[:,:,j] = sampled_fake1[:,:,j*2+1]
        
        dist3d = torch.sqrt(dist2d**2 + heights**2).squeeze().numpy()
        
        data_test = self.reverse_processing(sampled_fake2, dist3d, self.max_min)
        
        chan_dict_list = self.get_channel_dict(data_test, self.path_loss_threshold, self.feature_dict)
        
        return chan_dict_list
            
            
    def reverse_processing(self, data, dist3D, max_min=None):
      '''
        Parameters
        ----------
        data : matrix, shape = (batch size, number of features*8, number of multi-paths*2)
            output of generative model
        dist3D : array, shape = (batch_size, )
            3D distance between Tx and Rx
        max_min : dictionary 
            the max and min values of the original data
            
        Returns
        -------
        data : the matrix, shape = shape = (batch size, number of features, number of multi-paths) 
            down-sized matrix

        '''  
      # cacluate minimum delay
      min_delay = dist3D/cs.speed_of_light 
      # calculate free space path loss
      fspl = 20*np.log10(dist3D) + 20*np.log10(28e9) -147.55
      keys = ['path_loss', 'delay', 'zoa', 'aoa', 'zod', 'aod', 'phase']

      def reverse_max_min_scalor(data, key):
          max_ = max_min[f'{key}_max']
          min_ = max_min[f'{key}_min']
          if key == 'delay':
              max_  = max_*1e7
              min_ = min_*1e7
          #data = (2*data - max_ - min_)/(max_ - min_)
          data = ((max_ - min_)*data + (max_ + min_))/2
          return data
      
      for i, key in enumerate(keys):
          data[:,i,:] = reverse_max_min_scalor(data[:,i,:], key)
          
      data[:,1,:] +=min_delay[:,None]*1e7
      data[:,1,:] /=1e7
      data[:,0,:] +=fspl[:,None]
      
      return data




if  __name__ == "__main__":
    chan_model = Channel_Params()
    batch_size = 1000
    height = 30
    dist2d_max = 400
    dist2d_min = 10
    
    dist2d = (dist2d_max - dist2d_min)*torch.rand(batch_size)+ dist2d_min
    chan_dict_list = chan_model.get_chan_params(dist2d, height)
 
    
    
    
     
