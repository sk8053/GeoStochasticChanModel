# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:51:13 2023

@author: seongjoon kang
"""
# recover data from GAN

import numpy as np
import pandas as pd
import torch
from c_wgan import Generator
import matplotlib.pyplot as plt
import pickle
import scipy.constants as cs
import json
from scipy import stats as st
from scipy import special as sp
from scipy import spatial as spt
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_LOS_angle(Tx_loc, Rx_loc):
    dist_vec = Rx_loc - Tx_loc
    R_3d = np.sqrt(dist_vec[:,0]**2 + dist_vec[:,1]**2 + dist_vec[:,2]**2)
   
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
    
    return LOS_AOD_phi, LOS_ZOD_theta, LOS_AOA_phi, LOS_ZOA_theta

def get_feature_value(data, key):
    feature_pd = pd.DataFrame()
    for i in range(25):
        feature_pd = pd.concat([feature_pd, data[key+'_%d'%(i+1)]], axis =1)
    return feature_pd

def compute_LOS_prob(los_prob, dist_2d):
    
    LOS_count = np.zeros(int(max(dist_2d))+2)
    NLOS_count = np.zeros(int(max(dist_2d))+2)
    dist_2d = np.array(dist_2d/10, dtype = int)
    for j, d_2d_i in enumerate(dist_2d):
        if los_prob[j]>0:
            LOS_count[d_2d_i] +=1
        else:
            NLOS_count[d_2d_i] +=1
    los_prob = LOS_count/(LOS_count + NLOS_count)
    los_prob[0] = 1.0
    return los_prob

def reverse_processing(data, dist3D, max_min=None):
    
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
  # scale data so that the range of data becomes from -1 to 1
  for i, key in enumerate(keys):
      data[:,i,:] = reverse_max_min_scalor(data[:,i,:], key)
      
  data[:,1,:] +=min_delay[:,None]*1e7
  data[:,1,:] /=1e7
  # fill in the virtual values to make the size of matrix consistent
  data[:,0,:] +=fspl[:,None]
  
  return data

def get_pdfs(dist2d,angle,  feature, min_dist = 30):
    #angle_list = []
    dist_bin_size = 10
    distance_bins = np.arange(min_dist, 1250,step = dist_bin_size)
    
    if feature == 'zod' or feature =='zoa':
        angle_bin_size = 3
        angle_bins = np.arange(-90,93, step = angle_bin_size)
        v_max = 0.4
    else:
        angle_bin_size = 5
        angle_bins = np.arange(-80, 80, step =angle_bin_size)
        v_max = 0.2
    
    L = len(distance_bins)
    M = len(angle_bins)
    ang_im = np.zeros((M-1, L))
    angle = np.array(angle)
    for i, d in enumerate(distance_bins):
        if d!= distance_bins[-1]:
            I = (dist2d>d) & (dist2d<d+dist_bin_size)
        else:
            I = (dist2d>distance_bins[-1])
        
        angle_array = angle[I].reshape(-1)
        angle_array = angle_array[~np.isnan(angle_array)]
        #plt.plot(np.sort(angle_array), np.linspace(0,1,len(angle_array)))
        
        freq, n_bins = np.histogram(angle_array, bins=angle_bins)
        ang_im[:,i] = freq/np.sum(freq)
        #angle_list.append(list(angle_array))
        
    return ang_im

z_dim = 20
img_channel = 1
generator_features = 32
n_cond = 2

batch_size = 10000


#with open('Boston_data/cond_total.pickle','rb') as f:
#    cond = pickle.load(f)

#print ('conditions is loaded, shape is ', cond.shape)
#dist2d_data = cond[:,0]

model = torch.load(f'save_model/cwgan_z_{z_dim}.pt')
gen_state_dict = model['gen']
gen = Generator(z_dim, img_channel, generator_features, n_cond = n_cond).to(device)
gen.load_state_dict(gen_state_dict)
gen.eval()

feature_dict = {'path_loss':0, 'delay':1, 'zoa':2, 
                    'aoa':3, 'zod':4, 'aod':5, 
                    'phase':6, 'link state':7}
file_name_dict = {1.6:'Boston_data/bs_1_6m.csv', 30:'Boston_data/bs_30m.csv',
                  60:'Boston_data/bs_60m.csv', 90:'Boston_data/bs_90m.csv',
                  120:'Boston_data/bs_120m.csv'}

height = 30
feature = 'zoa'
for height in [1.6, 30, 60, 90, 120]: #1.6, 30, 60, 90, 120
    dist2d_max = 1480
    dist2d_min = 50#height+10
    print(f'======================= {height} ======================')
    
    bs_data_original = pd.read_csv(file_name_dict[height])
    angle = get_feature_value(bs_data_original, key = feature)
    
    
    tx_x, tx_y, tx_z = bs_data_original['tx_x'], bs_data_original['tx_y'], bs_data_original['tx_z']
    rx_x, rx_y, rx_z = bs_data_original['rx_x'], bs_data_original['rx_y'], bs_data_original['rx_z']
    Tx_arrays = np.column_stack((tx_x.to_numpy(), tx_y.to_numpy(), tx_z.to_numpy()))
    Rx_arrays = np.column_stack((rx_x.to_numpy(), rx_y.to_numpy(), rx_z.to_numpy()))
    
    LOS_AOD_phi, LOS_ZOD_theta, LOS_AOA_phi, LOS_ZOA_theta = compute_LOS_angle(Tx_arrays, Rx_arrays)
    angle_dict = {'aoa':LOS_AOA_phi, 'aod':LOS_AOD_phi, 'zod':LOS_ZOD_theta, 'zoa':LOS_ZOA_theta}
    los_angle = angle_dict[feature]
    angle -= los_angle[:,None]
    
    dist2d_data = np.sqrt((tx_x - rx_x)**2 + (tx_y- rx_y)**2)
    #print(np.min(dist2d_data))

    ang_im = get_pdfs(dist2d_data, angle, feature,min_dist = dist2d_min)
    #print(ang_im.shape)
    plt.imshow(ang_im, cmap='jet', vmax = 0.2);
    plt.yticks(np.arange(ang_im.shape[0]+1, step=5), np.arange(-90,90+15, step=15))
    plt.xticks(np.arange(ang_im.shape[1]+1, step=10), np.arange(1300, step=100)) 
    #plt.title(f'{height} m, {feature}, data')
    np.savetxt(f'data/zenith_angles/{feature}_{height}_data.txt', ang_im)
    #plt.colorbar();
    
    
    batch_size = 10000
    data_test_all = np.zeros((5, batch_size, 8, 25))
    LOS_ZOD_theta_all, LOS_ZOA_theta_all = [], []
    dist2d_all = []
    outage_I_all = []
    for k in range(5):
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        
        height_test = np.repeat([height], batch_size, axis = 0)
        
        I2 = np.random.permutation(np.arange(len(dist2d_data)))[:batch_size]
        dist2d = dist2d_data[I2].copy().to_numpy()
        tx_z2 = tx_z[I2].copy().to_numpy()
        dist2d_all.append(dist2d)
        
        dist3d = np.sqrt(dist2d**2 + (height_test-tx_z2)**2).squeeze()
        
        LOS_ZOD_theta = np.arccos((height_test-tx_z2)/dist3d)*180/np.pi
        LOS_ZOD_theta_all.append(LOS_ZOD_theta)
        
        LOS_ZOA_theta = 180 - LOS_ZOD_theta
        LOS_ZOA_theta_all.append(LOS_ZOA_theta)
        
        height_test = height_test[:,None]
        
        
        cond_test = np.append(dist2d[:,None], height_test, axis =1)
        cond_test = torch.tensor(cond_test, dtype = torch.float32).to(device)
        
        
        fake = gen(noise, cond_test)
        fake = fake.detach().cpu().numpy().squeeze()
        
        
        # sample fake
        sampled_fake1 = np.zeros((batch_size, 8,50))
        for i in range(8):
            sampled_fake1[:,i] = fake[:,i*8+3]
        sampled_fake2 = np.zeros((batch_size, 8,25))
        for j in range(25):
            sampled_fake2[:,:,j] = sampled_fake1[:,:,j*2+1]
        
        with open('Boston_data/max_min.json', 'r') as out:
            max_min = json.load(out)
          
        data_test = reverse_processing(sampled_fake2, dist3d, max_min)
        data_test_all[k] = data_test
        outage_I = data_test[:,0]<=200
        outage_I_all.append(outage_I)
        
    data_test_all = data_test_all.reshape(-1,8,25)
    dist2d_all = np.array(dist2d_all).reshape(-1)
    outage_I_all = np.array(outage_I_all).reshape(-1, 25)
    
    angle_model = data_test_all[:,feature_dict[feature]]
    
    LOS_angle_dict={'zod':np.array(LOS_ZOD_theta_all), 
                        'zoa':np.array(LOS_ZOA_theta_all)}
    angle_model -= LOS_angle_dict[feature].reshape(-1)[:,None]
    
    dist2d_all = np.repeat(dist2d_all[:,None], 25, axis = 1)
    ang_im_test = get_pdfs(dist2d_all[outage_I_all], angle_model[outage_I_all], 
                           feature, min_dist=dist2d_min)
    plt.figure()
    plt.imshow(ang_im_test, cmap='jet', vmax = 0.2);
    plt.yticks(np.arange(ang_im.shape[0]+1, step=5), np.arange(-90,90+15, step=15))
    plt.xticks(np.arange(ang_im.shape[1]+1, step=10), np.arange(1300, step=100)) 
    
    plt.title(f'{height} m, {feature}, model')
    np.savetxt(f'data/zenith_angles/{feature}_{height}_model.txt', ang_im_test)
