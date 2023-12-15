# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:43:12 2023

@author: seongjoon kang
"""
# sampling channel parameters from trained WGAN
# this is the example code on how to sample channel parameters

from channel_params import Channel_Params
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 5000 # batch size to sample from trained model
height = 120 # height of receiver
height_i = np.repeat([height], batch_size)
test = True
pathloss_threshold = 180 # pathloss value to determine link outage
max_n_path = 25 # maximum number of multipaths
#dist2d_max = 1250
#dist2d_min = 10
freq = 12e9 # carrier frequency
z_dim = 25 # dimension of latent variable
n_cond = 4 # dimension of conditional variable
max_min_dir = 'Herald_square_data' # directory to take data statistics, especially max and min values

saved_model_name = f'cwgan_z_{z_dim}.pt' # model name to upload and use
print(f'height = {height}')

# setup all the necessary parameters to sampel channel data
chan_param_model = Channel_Params(device = device, 
                                  path_loss_threshold=pathloss_threshold,
                                  z_dim = z_dim,
                                  n_cond = n_cond,
                                  save_file_name = saved_model_name,
                                  data_dir = max_min_dir,
                                  fc = freq)

feature_dict = chan_param_model.feature_dict
# list of channel parameter dictionaries
chan_dict_list_model = []

dist3d_list = []

for _ in tqdm(range(3)):
    # randomly sample distances between Tx and Rx in x and y directions
    dx = np.random.uniform(low=10,high =800, size =(batch_size,))
    dy = np.random.uniform(low=10,high =800, size =(batch_size,))
    # sample channel parameters by batchsize 
    _chan_dict_list_model = chan_param_model.get_chan_params(dist_vec = np.column_stack((dx, dy, height_i)))
    
    # append channel parameters 
    chan_dict_list_model += _chan_dict_list_model  
    
    dist3d = np.sqrt(dx**2 + dy**2 + height**2)
    dist3d_list += list(dist3d)
    


if test == True:
    # check the conditionality between 3D distance and pathloss
    path_loss_list = []
    dist3d_list_new = []
    for i, chan in enumerate(chan_dict_list_model):
        if len(chan) !=0:
            path_loss_list+= chan['path_loss']
            dist3d_list_new += list(np.repeat([dist3d_list[i]], len(chan['path_loss'])))
    
    fspl = 20*np.log10(dist3d_list_new) + 20*np.log10(freq) -147.55
    plt.figure()        
    plt.scatter(dist3d_list_new, path_loss_list, label = 'pathloss from model', s= 5)
    plt.scatter(dist3d_list_new, fspl, label = 'FSPL',s = 5)
    plt.xlabel('3D distance [m]', fontsize = 13)
    plt.ylabel('Pathloss [dB]', fontsize = 13)
    plt.title (f'height = {height} m', fontsize = 13)
    plt.legend()
    plt.grid()
    plt.savefig(f'evaluation_data/test_images/distance_vs_pathloss at {height}m.png', dpi = 600)

