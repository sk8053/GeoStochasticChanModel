# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 19:51:13 2023

@author: seongjoon kang
"""
# recover data from GAN
# evaluate model with normalized data

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
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_feature_value(data, key):
    feature_pd = pd.DataFrame()
    for i in range(25):
        feature_pd = pd.concat([feature_pd, data[key+'_%d'%(i+1)]], axis =1)
    return feature_pd

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

height = 120
feature = 'aoa'

file_name_dict = {1.6:'Boston_data/bs_1_6m.csv', 30:'Boston_data/bs_30m.csv',
                  60:'Boston_data/bs_60m.csv', 90:'Boston_data/bs_90m.csv',
                  120:'Boston_data/bs_120m.csv'}

bs_data = pd.read_csv(file_name_dict[height])
feature_dict = {'path_loss':0, 'delay':1, 'zoa':2, 
                'aoa':3, 'zod':4, 'aod':5, 'phase':6, 'link state':7}
pl = get_feature_value(bs_data, key = 'path_loss')
pl = pl.to_numpy().reshape(-1)
pl = pl[~np.isnan(pl)]
I = pl<=180

if feature == 'link state':
    d2 = bs_data[feature]
    print(np.sum(d2==1))
    d2[d2==2] = -1
    d2 = d2.to_numpy().reshape(-1)
    f_data = d2[~np.isnan(d2)]

else:
    d2 = get_feature_value(bs_data, key = feature)  
    d2 = d2.to_numpy().reshape(-1)
    f_data = d2[~np.isnan(d2)][I]

# read data
with open('Boston_data/data_total.pickle','rb') as f:
    dataset = pickle.load(f)
    
with open('Boston_data/cond_total.pickle','rb') as f:
    cond = pickle.load(f)
I = cond[:,1]==height
dataset = dataset[I]

print ('conditions is loaded, shape is ', cond.shape)

z_dim = 20
img_channel = 1
generator_features = 32
n_cond = 2

batch_size = 5000
dist2d_max = 1400
dist2d_min = 100


model = torch.load(f'save_model/cwgan_z_{z_dim}.pt')

gen_state_dict = model['gen']
gen = Generator(z_dim, img_channel, generator_features, n_cond = n_cond).to(device)
gen.load_state_dict(gen_state_dict)
gen.eval()

height_test = torch.repeat_interleave(torch.tensor([height]), batch_size, dim = 0)
height_test = height_test[:,None]
dist2d = (dist2d_max - dist2d_min)*torch.rand(height_test.shape)+ dist2d_min
dist3d = torch.sqrt(dist2d**2 + height_test**2).squeeze().numpy()

cond_test = torch.concat([dist2d, height_test], dim =1).to(device)
noise = torch.randn(batch_size, z_dim, 1, 1).to(device)

fake = gen(noise, cond_test)
fake = fake.detach().cpu().numpy().squeeze()


# sample fake
sampled_fake1 = np.zeros((batch_size, 8,50))
for i in range(8):
    sampled_fake1[:,i] = fake[:,i*8+5]
sampled_fake2 = np.zeros((batch_size, 8,25))
for j in range(25):
    sampled_fake2[:,:,j] = sampled_fake1[:,:,j*2+1]

with open('Boston_data/max_min.json', 'r') as out:
    max_min = json.load(out)

I = np.random.permutation(range(len(sampled_fake2)))
f_data_sampled = dataset[I]

print(sampled_fake2.shape, f_data_sampled.shape)
for j in range(8):
    f_m = sampled_fake2[:,j].reshape(-1)
    f_d = f_data_sampled[:,j].reshape(-1)
    print(st.wasserstein_distance(f_m, f_d))
    
'''
data_test = reverse_processing(sampled_fake2, dist3d, max_min)
pl = data_test[:,feature_dict['path_loss']].reshape(-1)
I = pl <= 180
f_model = data_test[:,feature_dict[feature]].reshape(-1)

if feature != 'link state':
    f_model = f_model[I]
else:
    f_model[f_model>0] = 1.0
    f_model[f_model<=0] = -1.0

plt.plot(np.sort(f_model), np.linspace(0,1,len(f_model)), label = 'model');
plt.plot(np.sort(f_data), np.linspace(0,1,len(f_data)), label = 'data');
plt.legend()
f_data_sampled = np.random.choice(f_data, len(f_model))
print(st.wasserstein_distance(f_model, f_data_sampled))
'''