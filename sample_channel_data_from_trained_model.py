# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:43:12 2023

@author: seongjoon kang
"""
# sampling channel parameters from trained WGAN
# this is the example code on how to sample channel parameters

from channel_params import Channel_Params
#from utill import get_feature_value, compute_linkstate_prob
#from utill import  get_feature_value, get_pdfs, compute_rms_spread
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
#import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 5000 # batch size to sample from trained model
height = 30 # height of receiver
height_i = np.repeat([height], batch_size)

pathloss_threshold = 180 # pathloss value to determine link outage
max_n_path = 25 # maximum number of multipaths
#dist2d_max = 1250
#dist2d_min = 10
freq = 12e9 # carrier frequency
z_dim = 25 # dimension of latent variable
n_cond = 4 # dimension of conditional variable
max_min_dir = 'Herald_square_data' # directory to take data statistics, especially max and min values

saved_model_name = f'cwgan_z_{z_dim}.pt'
print(f'height = {height}')

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

for iter in tqdm(range(3)):
    dx = np.random.uniform(low=10,high =800, size =(batch_size,))
    dy = np.random.uniform(low=10,high =800, size =(batch_size,))
    # sample channel parameters by batchsize 
    _chan_dict_list_model = chan_param_model.get_chan_params(dist_vec = np.column_stack((dx, dy, height_i)))
    # append channel parameters 
    chan_dict_list_model += _chan_dict_list_model  
    
print(chan_dict_list_model[10])


