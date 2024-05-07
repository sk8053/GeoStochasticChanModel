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
    fc = 12e9 # carrier frequency is defined as static variable in class
    def __init__(self, 
                 saved_path = 'save_model',
                 data_dir = 'Boston_data',
                 z_dim = 25, 
                 generator_features = 32,
                 n_cond = 2,
                 path_loss_threshold = 180,
                 device = 'cpu',
                 fc = 28e9,
                 save_file_name = 'cwgan_z_25.pt'):
        fc = fc # assign carrier frequency
        saved_file_name = save_file_name
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
        self.dist3d_ex = None
   
    @staticmethod
    def get_channel_dict(data_mat, dist_vec = None, path_loss_threshold=180, feature_dict = None):
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
        chan_dict_list = []
        
        for j in range(batch_size):
            keys = list(feature_dict.keys())
           
            chan_dict = dict()     
            path_loss_ = data_mat[j,0]
           
            # sort multi-path components based by pathloss
            I2 = np.argsort(path_loss_) 
            # cut multipath based on pathloss threshold
            I = data_mat[j,0]<path_loss_threshold
            
            # calculate the link state value by averaging the last raw
            ls = np.mean(data_mat[j, feature_dict['link state']])
            
            if ls >0: # if link state is LOS
                chan_dict['link state']  = [1]
                # if data is output of the  model and link state is LOS
                # we need to manually calculate LOS components
             
                if dist_vec != None:
                    los_dict = Channel_Params.compute_LOS_components(dist_vec[j,:][None], fc = Channel_Params.fc)
                keys.remove('link state')
                
                for key in keys:
                    k = feature_dict[key]
                    # order multipath channel params by pathloss and 
                    # take only paths satisfying the pathloss threshold
                    # 1) I2: sort based on pathloss 2) I: cut pathloss values exceeding threshold
                    #mpath_comp = data_mat[j,k][I] 
                    mpath_comp = data_mat[j,k][I2][I] 
                    
                    if len(mpath_comp) !=0 and dist_vec != None:                
                        mpath_comp[0] = los_dict[key][0]
                        
                    chan_dict[key] = list(mpath_comp)
            
            else: # if link state is NLOS
                chan_dict['link state']  = [-1]
                keys.remove('link state')
                for key in keys:
                    k = feature_dict[key]
                    # order multipath channel params based on delay and 
                    # take only paths satisfying the pathloss threshold
                    # 1) I2: sort based on pathloss 2) I: cut pathloss values exceeding threshold
                    #mpath_comp = data_mat[j,k][I]
                    mpath_comp = data_mat[j,k][I2][I]
                    chan_dict[key] = list(mpath_comp)
            
            chan_dict_list.append(chan_dict)
            
        return chan_dict_list
    
    def get_chan_params(self, dist_vec):
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
        
        batch_size = len(dist_vec)
        dist_vec = torch.tensor(dist_vec, dtype = torch.float32)
        dx, dy, heights = dist_vec[:,0], dist_vec[:,1], dist_vec[:,2] 
        #torch.repeat_interleave(torch.tensor([height]), batch_size, dim = 0)
        dist3d = torch.sqrt(dx**2 + dy**2 + heights**2)
        dist2d = torch.sqrt(dx**2 + dy**2)
        ang = torch.arccos(heights/dist3d)*180/torch.pi
        
        condition = torch.concat([dist2d[:,None]/1000, heights[:,None]/120, 
                                  dist3d[:,None]/1000, 
                                  ang[:,None]/90], dim =1).to(self.device)
        
        noise = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)
        
        fake = self.gen(noise, condition)
        fake = fake.detach().cpu().numpy().squeeze()
        # downsize the matrix, fake
        # sample fake
        sampled_fake1 = np.zeros((batch_size, 8,50))
        for i in range(8):
            sampled_fake1[:,i] = fake[:,i*8+5]
        sampled_fake2 = np.zeros((batch_size, 8,25))
        for j in range(25):
            sampled_fake2[:,:,j] = sampled_fake1[:,:,j*2+1]
        
        dist3d = dist3d.squeeze().numpy()
        
        data_test = self.reverse_processing(sampled_fake2, dist3d, self.max_min)
        
        chan_dict_list = self.get_channel_dict(data_test,dist_vec, self.path_loss_threshold, self.feature_dict)
        
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
      fspl = 20*np.log10(dist3D) + 20*np.log10(self.fc) -147.55
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
  
    @staticmethod  
    def compute_LOS_components(dist_vec, fc = 28e9):
        #dist_vec = Rx_loc - Tx_loc
        R_3d = np.sqrt(dist_vec[:,0]**2 + dist_vec[:,1]**2 + dist_vec[:,2]**2)
        path_loss = 20*np.log10(R_3d) + 20*np.log10(fc) -147.55
        delay = R_3d/(cs.c/1.000293)#cs.speed_of_light
        # azimuth angle of departure
        LOS_AOD_phi = np.arctan2(dist_vec[:,1], dist_vec[:,0])*180/np.pi
        LOS_AOD_phi[LOS_AOD_phi<0] += 360
        LOS_AOD_phi[LOS_AOD_phi>360] -=360
        # zenith angle of departure
        LOS_ZOD_theta = np.arccos(dist_vec[:,2]/R_3d)*180/np.pi
        # azimuth angle of arrival
        LOS_AOA_phi = LOS_AOD_phi -180
        LOS_AOA_phi[LOS_AOA_phi<0] += 360
        LOS_AOA_phi[LOS_AOA_phi>360] -=360
        # zenith angle of arrival
        LOS_ZOA_theta = 180-LOS_ZOD_theta
        # make the range of angle from -180 to 180
        LOS_AOA_phi[LOS_AOA_phi>180] = LOS_AOA_phi[LOS_AOA_phi>180]-360
        LOS_AOD_phi[LOS_AOD_phi>180] = LOS_AOD_phi[LOS_AOD_phi>180]-360
        
        LOS_phase_rad = -2*np.pi*(fc*delay - np.floor(fc*delay))
        LOS_phase = np.rad2deg(LOS_phase_rad)
        LOS_phase[LOS_phase<-180] = LOS_phase[LOS_phase<-180] + 360
        
        return {'path_loss':path_loss, 
                'delay':delay, 
                'aod':LOS_AOD_phi, 
                'zod':LOS_ZOD_theta, 
                'aoa':LOS_AOA_phi, 
                'zoa':LOS_ZOA_theta, 
                'phase':LOS_phase
                }



if  __name__ == "__main__":
    chan_model = Channel_Params()
    batch_size = 1000
    height = 30
    dist2d_max = 400
    dist2d_min = 10
    
    dist2d = (dist2d_max - dist2d_min)*torch.rand(batch_size)+ dist2d_min
    chan_dict_list = chan_model.get_chan_params(dist2d, height)
 
    
    
    
     
