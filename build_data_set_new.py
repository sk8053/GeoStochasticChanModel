# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:03:00 2023

@author: seongjoon kang
# this code performs the normalization of data so that the range of values would be from -1 to 1. 
# to do this, the code reads min and max values from the directory, Herald_square_data. 
# for pathloss and propagation delay, additional process is needed as mentioned in the paper. 
# (substract FSPL and minimum delay, multiply 10^7 for delay values)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as cs
#from tqdm import tqdm
import pickle
#import glob
import json


def data_processing(data, height  = 30, max_min=None, fc = 12e9, freq_cond = False):
  
  tx_x, tx_y, tx_z = data['tx_x'], data['tx_y'], data['tx_z']
  rx_x, rx_y, rx_z = data['rx_x'], data['rx_y'], data['rx_z']
  
  # range of angles in LOS is dependent on the height of RX and TX
  dist_vect_2d = np.column_stack((tx_x-rx_x, tx_y-rx_y))
  dist_vect_3d = np.column_stack((tx_x-rx_x, tx_y-rx_y, tx_z-rx_z))
  
  distance_2D = np.linalg.norm(dist_vect_2d, axis = 1)
  
  heights = np.repeat([height], len(distance_2D))
  freqs = np.repeat([fc],len(distance_2D))
  
  distance_3D = np.linalg.norm(dist_vect_3d, axis = 1)
  # cacluate 
  min_delay = distance_3D/cs.speed_of_light
  # calculate free space path loss
  fspl = 20*np.log10(distance_3D) + 20*np.log10(fc) -147.55

  def get_feature_value(key):
      feature_pd = pd.DataFrame()
      for i in range(25):
          feature_pd = pd.concat([feature_pd, data[key+'_%d'%(i+1)]], axis =1)
      return feature_pd
  
  data_all = np.zeros([data.shape[0], 8, 25])
  data_all[:,0,:] = get_feature_value('path_loss') 
  data_all[:,1,:] = get_feature_value('delay')
  #  the range of elevation angles from 0 to 180
  data_all[:,2,:] = get_feature_value('zoa')
  #  the range of azimuth angles from -180 to 180
  data_all[:,3,:] = get_feature_value('aoa')
  data_all[:,4,:] = get_feature_value('zod')
  data_all[:,5,:] = get_feature_value('aod') 
  data_all[:,6,:] = get_feature_value('phase')

  link_state = data['link state'].to_numpy().copy()
  link_state = link_state.astype(float)
  
  # NLOS state
  link_state[link_state==2] = np.random.uniform(-1.0,-0.9,len(link_state[link_state==2]))
  link_state[np.isnan(link_state)] = np.random.uniform(-1.0,-0.9,sum(np.isnan(link_state)))
  # LOS state
  link_state[link_state==1] =  np.random.uniform(0.9,1.0,len(link_state[link_state==1]))

  data_all[:,7,:] = np.repeat(link_state[:,None], 25, axis = -1)
  
 
  data_all[:,1,:] = data_all[:,1,:] *1e7
  data_all[:,1,:] -=  min_delay[:,None]*1e7
  
  #fspl = fspl[~np.isnan(data_all[:,0,0])]
  #distance_2D = distance_2D[~np.isnan(data_all[:,0,0])]
  #heights = heights[~np.isnan(data_all[:,0,0])]
  #data_all = data_all[~np.isnan(data_all[:,0,0])]
  
  print(data_all.shape)
  # fill in the virtual values to make the size consistent
  b_index = np.isnan(data_all[:,0,:])
  data_all[:,0,:][np.isnan(data_all[:,0,:])] =  np.random.uniform(low = 200, high= 210, size = (np.sum(b_index),))
  data_all[:,0,:] -=fspl[:,None]
  data_all[:,1,:][np.isnan(data_all[:,1,:])] = np.random.uniform(low = 0.01, high =110 , size = (np.sum(b_index),))  
  data_all[:,2,:][np.isnan(data_all[:,2,:])] = np.random.uniform(low = 0, high= 180, size = (np.sum(b_index),))
  data_all[:,3,:][np.isnan(data_all[:,3,:])] = np.random.uniform(low = -180, high =180, size = (np.sum(b_index),))
  data_all[:,4,:][np.isnan(data_all[:,4,:])] = np.random.uniform(low = 0, high = 180, size = (np.sum(b_index),))
  data_all[:,5,:][np.isnan(data_all[:,5,:])] = np.random.uniform(low = -180, high = 180, size = (np.sum(b_index),))
  data_all[:,6,:][np.isnan(data_all[:,6,:])] = np.random.uniform(low = -180, high = 180, size = (np.sum(b_index),))
  keys = ['path_loss', 'delay', 'zoa', 'aoa', 'zod', 'aod', 'phase']

  def max_min_scalor(data, key):
      max_ = max_min[f'{key}_max']
      min_ = max_min[f'{key}_min']
      if key == 'delay':
          max_  = max_*1e7
          min_ = min_*1e7
      data = (2*data - max_ - min_)/(max_ - min_)
      
      return data
  # scale data so that the range of data becomes from -1 to 1
  for i, key in enumerate(keys):
      data_all[:,i,:] = max_min_scalor(data_all[:,i,:], key)
      
  if freq_cond is True:
      return data_all, np.column_stack((distance_2D,  10*freqs/1e9))
  else:
      return data_all, np.column_stack((distance_2D,  heights))



data_total = []
cond_total = []
dir_ = 'Herald_square_data'
fc_list = [12e9, 12e9, 12e9, 12e9, 12e9]
csv_files = ['bs_1_6m.csv', 'bs_30m.csv', 'bs_60m.csv', 'bs_90m.csv', 'bs_120m.csv']
heights = [1.6, 30, 60, 90, 120] # this is condition values

#dir_ = 'Herald_square_data_freq'
#fc_list = [6e9, 12e9, 18e9, 24e9]
#csv_files = ['bs_6GHz.csv', 'bs_12GHz.csv', 'bs_18GHz.csv', 'bs_24GHz.csv']

with open(f'{dir_}/max_min.json', 'r') as out:
    max_min = json.load(out)
freq_cond = False

data_total = None
cond_total = None
s =0
for i,  (file, fc) in enumerate(zip(csv_files, fc_list)):
    df = pd.read_csv(f'{dir_}/{file}')
    s += len(df)
    processed_data, conditions = data_processing(df, heights[i], max_min, fc = fc, freq_cond=freq_cond)
    I = np.arange(conditions.shape[0])
    I = np.random.permutation(I)
    N = int(len(I)*1.0)
    processed_data, conditions = processed_data[:N], conditions[:N]
    if i ==0:
        data_total = processed_data
        cond_total = conditions
    else:
        data_total = np.append(data_total, processed_data, axis =0)
        cond_total = np.append(cond_total, conditions, axis = 0)
    
print(f'total data shape is {data_total.shape}')
print(f'total cond data shape is {cond_total.shape}')
print(s)
for i in range(8):
    print(i,np.min(data_total[:,i,:]), np.max(data_total[:,i,:]))


with open(f'{dir_}/data_total.pickle', 'wb') as handle:
    pickle.dump(data_total, handle)
with open(f'{dir_}/cond_total.pickle', 'wb') as handle:
    pickle.dump(cond_total, handle)


if 0:
    img0 = data_total[28002]
    #ax = plt.gca()
    #ax.get_yaxis().set_visible(False)
    plt.yticks(range(img0.shape[0]), np.arange(1,img0.shape[0]+1))
    plt.xticks(np.arange(img0.shape[1]), np.arange(1, img0.shape[1]+1))
    plt.imshow(img0)
    plt.tight_layout()
    plt.savefig('img0.png', dpi = 600, bbox_inches='tight',pad_inches=0)
    
    img0_enlarged = np.repeat(img0, 8, axis =0)
    img0_enlarged  = np.repeat(img0_enlarged , 2, axis =1)
    #ax = plt.gca()
    #ax.get_yaxis().set_visible(False)
    #plt.xticks(np.arange(img0_enlarged.shape[1]), np.arange(1, img0_enlarged.shape[1]+1))
    plt.imshow(img0_enlarged)
    plt.tight_layout()
    plt.colorbar()
    plt.savefig('img0_enlarged.png', dpi = 600, bbox_inches='tight',pad_inches=0)
    
    img1 = data_total[18020]
    #ax = plt.gca()
    #ax.get_yaxis().set_visible(False)
    plt.yticks(range(img0.shape[0]), np.arange(1,img0.shape[0]+1))
    plt.xticks(np.arange(img1.shape[1]), np.arange(1, img1.shape[1]+1))
    plt.imshow(img1)
    plt.tight_layout()
    plt.savefig('img1.png', dpi = 600, bbox_inches='tight',pad_inches=0)
    
    img1_enlarged = np.repeat(img1, 8, axis =0)
    img1_enlarged  = np.repeat(img1_enlarged , 2, axis =1)
    #ax = plt.gca()
    #ax.get_yaxis().set_visible(False)
    #plt.xticks(np.arange(img0_enlarged.shape[1]), np.arange(1, img0_enlarged.shape[1]+1))
    plt.imshow(img1_enlarged)
    plt.tight_layout()
    plt.savefig('img1_enlarged.png', dpi = 600,bbox_inches='tight',pad_inches=0)
    
    img2 = data_total[180000]
    #ax.get_yaxis().set_visible(True)
    plt.yticks(range(img1.shape[0]), np.arange(1,img1.shape[0]+1))
    plt.xticks(np.arange(img1.shape[1]), np.arange(1, img1.shape[1]+1))
    plt.imshow(img2)
    plt.tight_layout()
    plt.savefig('img2.png', dpi = 600,bbox_inches='tight',pad_inches=0)
    
    img2_enlarged = np.repeat(img2, 8, axis =0)
    img2_enlarged  = np.repeat(img2_enlarged , 2, axis =1)
    #ax = plt.gca()
    #ax.get_yaxis().set_visible(False)
    #plt.xticks(np.arange(img0_enlarged.shape[1]), np.arange(1, img0_enlarged.shape[1]+1))
    plt.imshow(img2_enlarged)
    plt.tight_layout()
    plt.savefig('img2_enlarged.png', dpi = 600,bbox_inches='tight',pad_inches=0)
    
    #img2 = data_total[180020]
    
    #img1_enlarged = np.repeat(img1, 8, axis =0)
    #img1_enlarged  = np.repeat(img1_enlarged , 2, axis =1)
    
    #img2_enlarged = np.repeat(img2, 8, axis =0)
    #img2_enlarged  = np.repeat(img2_enlarged , 2, axis =1)
    #plt.imshow(img2_enlarged)
