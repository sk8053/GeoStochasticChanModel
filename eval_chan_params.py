# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:34:32 2023

@author: seongjoon kang
"""

from channel_params import Channel_Params
from utill import get_feature_value, compute_LOS_prob
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 5000
height = 120
pathloss_threshold = 180
max_n_path = 25
dist2d_max = 1250
dist2d_min = 10

chan_param_model = Channel_Params(device = device, 
                                  path_loss_threshold=pathloss_threshold)

feature_dict = chan_param_model.feature_dict

file_name_dict = {1.6:'Boston_data/bs_1_6m.csv', 30:'Boston_data/bs_30m.csv',
                  60:'Boston_data/bs_60m.csv', 90:'Boston_data/bs_90m.csv',
                  120:'Boston_data/bs_120m.csv'}

bs_data_df_original = pd.read_csv(file_name_dict[height])
bs_data_df_original['link state'][bs_data_df_original['link state']==2] = -1


tx_x, tx_y = bs_data_df_original['tx_x'], bs_data_df_original['tx_y']
rx_x, rx_y = bs_data_df_original['rx_x'], bs_data_df_original['rx_y']
dist2d_data_original = np.sqrt((tx_x - rx_x)**2 + (tx_y- rx_y)**2).to_numpy()

chan_dict_list_data, chan_dict_list_model = [], []
dist2d_list =[]
for iter in tqdm(range(5)):
    I_ = np.random.permutation(len(bs_data_df_original))[:batch_size]
    
    bs_data_df = bs_data_df_original.iloc[I_,:]
    dist2d = dist2d_data_original[I_]
    
    T = bs_data_df.shape[0]
    bs_data = np.zeros((T, len(feature_dict), max_n_path))
    
    for i, feature in enumerate(feature_dict.keys()):
        df = get_feature_value(bs_data_df, key=feature, max_n_path=max_n_path)
        bs_data[:, i] = df.to_numpy()
        
    _chan_dict_list_data = chan_param_model.get_channel_dict(bs_data, 
                                                            pathloss_threshold, 
                                                            feature_dict)
    
    _chan_dict_list_model = chan_param_model.get_chan_params(dist2d, height)
    
    chan_dict_list_data += _chan_dict_list_data
    chan_dict_list_model += _chan_dict_list_model
    dist2d_list += list(dist2d)

pathloss_list_model, pathloss_list_data = [], []
delay_list_model, delay_list_data = [], []
linkstate_list_model, linkstate_list_data = [], []


for chan_model, chan_data in zip(chan_dict_list_model, chan_dict_list_data):
    pathloss_list_model+= chan_model['path_loss']
    pathloss_list_data += chan_data['path_loss']
    
    delay_list_model += chan_model['delay']
    delay_list_data += chan_data['delay']
    
    linkstate_list_model += chan_model['link state']
    linkstate_list_data += chan_data['link state']


dist2d_array= np.array(dist2d_list)
linkstate_list_model = np.array(linkstate_list_model)[dist2d_array<dist2d_max]
linkstate_list_data = np.array(linkstate_list_data)[dist2d_array<dist2d_max]
dist2d_array = dist2d_array[dist2d_array<dist2d_max]

los_prob_model = compute_LOS_prob(linkstate_list_model, dist2d_array)
los_prob_data = compute_LOS_prob(linkstate_list_data, dist2d_array)

np.savetxt(f'data/path_loss/path_loss_{height}_model.txt', pathloss_list_model)
np.savetxt(f'data/path_loss/path_loss_{height}_data.txt', pathloss_list_data)

np.savetxt(f'data/delay/delay_{height}_model.txt', delay_list_model)
np.savetxt(f'data/delay/delay_{height}_data.txt', delay_list_data)

np.savetxt(f'data/link state/link state_{height}_model.txt', los_prob_model)
np.savetxt(f'data/link state/link state_{height}_data.txt', los_prob_data)


