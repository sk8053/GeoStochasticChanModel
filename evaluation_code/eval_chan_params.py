# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:34:32 2023

@author: seongjoon kang
"""

from channel_params import Channel_Params
from utill import get_feature_value, compute_linkstate_prob
from utill import  get_feature_value, get_pdfs, compute_rms_spread
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 5000
height = 1.6
pathloss_threshold = 180
max_n_path = 25
#dist2d_max = 1250
#dist2d_min = 10
z_dim = 35
n_cond = 4
dir_ = 'Herald_square_data'

saved_model_name = f'cwgan_z_{z_dim}.pt'
print(f'height = {height}')

chan_param_model = Channel_Params(device = device, 
                                  path_loss_threshold=pathloss_threshold,
                                  z_dim = z_dim,
                                  n_cond = n_cond,
                                  save_file_name = saved_model_name,
                                  data_dir = dir_,
                                  fc = 12e9)

feature_dict = chan_param_model.feature_dict

file_name_dict = {1.6:f'{dir_}/bs_1_6m.csv', 30:f'{dir_}/bs_30m.csv',
                  60:f'{dir_}/bs_60m.csv', 90:f'{dir_}/bs_90m.csv',
                  120:f'{dir_}/bs_120m.csv'}

bs_data_df_original = pd.read_csv(file_name_dict[height])
bs_data_df_original['link state'][bs_data_df_original['link state']==2] = -1


tx_x, tx_y, tx_z = bs_data_df_original['tx_x'], bs_data_df_original['tx_y'], bs_data_df_original['tx_z']
rx_x, rx_y, rx_z = bs_data_df_original['rx_x'], bs_data_df_original['rx_y'], bs_data_df_original['rx_z'] 
#dist2d_data_original = np.sqrt((tx_x - rx_x)**2 + (tx_y- rx_y)**2).to_numpy()

chan_dict_list_data, chan_dict_list_model = [], []
dist2d_list =[]
los_zoa, los_zod = [], []
height_i = np.repeat([height], batch_size)

for iter in tqdm(range(20)):
    I_ = np.random.permutation(len(bs_data_df_original))[:batch_size]
    
    bs_data_df = bs_data_df_original.iloc[I_,:]
    
    tx_x_i, tx_y_i, tx_z_i = tx_x.iloc[I_], tx_y.iloc[I_], tx_z.iloc[I_]
    rx_x_i, rx_y_i, rx_z_i = rx_x.iloc[I_], rx_y.iloc[I_], rx_z.iloc[I_]
    
    dx, dy = (rx_x_i - tx_x_i).to_numpy(), (rx_y_i - tx_y_i).to_numpy()
    dz = (rx_z_i - tx_z_i).to_numpy()
    
    #dist3d = np.sqrt(dx**2+dy**2+dz**2)
    LOS_comp = chan_param_model.compute_LOS_components(dist_vec=np.column_stack((dx, dy, rx_z_i - tx_z_i)))
    
    _los_zoa = LOS_comp['zoa']
    _los_zod = LOS_comp['zod']
    
    bs_data = np.zeros((batch_size, len(feature_dict), max_n_path))
    
    for i, feature in enumerate(feature_dict.keys()):
        df = get_feature_value(bs_data_df, key=feature, max_n_path=max_n_path)
        bs_data[:, i] = df.to_numpy()
        
    _chan_dict_list_data = chan_param_model.get_channel_dict(bs_data,
                                                             dist_vec=None,
                                                            path_loss_threshold=pathloss_threshold, 
                                                            feature_dict=feature_dict)
    
    _chan_dict_list_model = chan_param_model.get_chan_params(dist_vec = np.column_stack((dx, dy, height_i)))
    
    chan_dict_list_data += _chan_dict_list_data
    chan_dict_list_model += _chan_dict_list_model
    
    dist2d = np.sqrt(dx**2 + dy**2)
    dist2d_list += list(dist2d)
    
    los_zoa += list(_los_zoa)
    los_zod += list(_los_zod)


pathloss_list_model, pathloss_list_data = [], []
delay_list_model, delay_list_data = [], []
linkstate_list_model, linkstate_list_data = [], []

outage_state_model = np.zeros((len(dist2d_list, )))
outage_state_data = np.zeros((len(dist2d_list, )))

ZOA_data_mat = np.ones((len(dist2d_list), max_n_path))*np.nan
ZOD_data_mat = np.ones((len(dist2d_list), max_n_path))*np.nan
ZOA_model_mat = np.ones((len(dist2d_list), max_n_path))*np.nan
ZOD_model_mat = np.ones((len(dist2d_list), max_n_path))*np.nan

aod_model, aod_data = [], []
aoa_model, aoa_data = [], []
phase_model, phase_data = [],[]
dly_rms_model, dly_rms_data = [],[]
aod_rms_model, aod_rms_data = [], []
aoa_rms_model, aoa_rms_data = [], []

#pl_1path_model, pl_1path_data = [], []
#dly_1path_model, dly_1path_data = [], []

for i, (chan_model, chan_data) in enumerate(zip(chan_dict_list_model, chan_dict_list_data)):
    
    if len(chan_model['path_loss']) ==0:
        outage_state_model[i] =1
    else:
        _dly_rms_model = compute_rms_spread(chan_model, feature='delay')
        dly_rms_model.append(_dly_rms_model)
        
        _aoa_rms_model = compute_rms_spread(chan_model, feature='aoa')
        aoa_rms_model.append(_aoa_rms_model)
        
        _aod_rms_model = compute_rms_spread(chan_model, feature='aod')
        aod_rms_model.append(_aod_rms_model)
        
        #if len(chan_model['path_loss']) >=2:
        #pl_1path_model.append(chan_model['path_loss'][0])
        #dly_1path_model.append(chan_model['delay'][0])
    
    if len(chan_data['path_loss']) ==0:
        outage_state_data[i] =1
    else:
        _dly_rms_data = compute_rms_spread(chan_data, feature='delay')
        dly_rms_data.append(_dly_rms_data)
        
        _aoa_rms_data = compute_rms_spread(chan_model, feature='aoa')
        aoa_rms_data.append(_aoa_rms_data)
        
        _aod_rms_data = compute_rms_spread(chan_model, feature='aod')
        aod_rms_data.append(_aod_rms_data)
        
        #if len(chan_data['path_loss']) >= 2:
        #pl_1path_data.append(chan_data['path_loss'][0])
        #dly_1path_data.append(chan_data['delay'][0])
    
        
    pathloss_list_model+= chan_model['path_loss']
    pathloss_list_data += chan_data['path_loss']
    
    delay_list_model += chan_model['delay']
    delay_list_data += chan_data['delay']
    
    linkstate_list_model += chan_model['link state']
    linkstate_list_data += chan_data['link state']
    
    ZOA_model_mat[i, :len(chan_model['zoa'])] = chan_model['zoa']
    ZOD_model_mat[i, :len(chan_model['zod'])] = chan_model['zod']
    
    ZOA_data_mat[i, :len(chan_data['zoa'])] = chan_data['zoa']
    ZOD_data_mat[i, :len(chan_data['zod'])] = chan_data['zod']
    
    aod_model +=chan_model['aod']
    aod_data += chan_data['aod']
    
    aoa_model +=chan_model['aoa']
    aoa_data += chan_data['aoa']
    
    phase_model +=chan_model['phase']
    phase_data += chan_data['phase']
    
    
    
dist2d_array= np.array(dist2d_list)

los_prob_model = compute_linkstate_prob(linkstate_list_model, dist2d_array)
los_prob_data = compute_linkstate_prob(linkstate_list_data, dist2d_array)

# outage probability 
outage_prob_model = compute_linkstate_prob(outage_state_model, dist2d_array, enable_first_prob=False)
outage_prob_data = compute_linkstate_prob(outage_state_data, dist2d_array, enable_first_prob=False)

'''
plt.figure()
plt.plot(np.sort(pl_1path_data), np.linspace(0,1,len(pl_1path_data)), label = 'data')
plt.plot(np.sort(pl_1path_model), np.linspace(0,1,len(pl_1path_model)), label = 'model')
plt.title(f'Pathloss, height = {height}m')
plt.legend()

plt.figure()
plt.plot(np.sort(dly_1path_data), np.linspace(0,1,len(dly_1path_data)), label = 'data')
plt.plot(np.sort(dly_1path_model), np.linspace(0,1,len(dly_1path_model)), label = 'model')
plt.legend()
plt.title(f'Delay, height = {height}m')


plt.figure()
plt.plot(np.sort(dly_rms_data), np.linspace(0,1,len(dly_rms_data)), label = 'data')
plt.plot(np.sort(dly_rms_model), np.linspace(0,1,len(dly_rms_model)), label = 'model')
plt.title(f'Delay, height = {height}m')
plt.legend()

plt.figure()
plt.plot(np.sort(aoa_rms_data), np.linspace(0,1,len(dly_rms_data)), label = 'data')
plt.plot(np.sort(aoa_rms_model), np.linspace(0,1,len(dly_rms_model)), label = 'model')
plt.title(f'AOA, height = {height}m')
plt.legend()
'''
plt.figure()
plt.plot(np.sort(pathloss_list_data), np.linspace(0,1,len(pathloss_list_data)), label = 'data')
plt.plot(np.sort(pathloss_list_model), np.linspace(0,1,len(pathloss_list_model)), label = 'model')
plt.title(f'Pathloss, height = {height}m')
plt.legend()


np.savetxt(f'data/path_loss/path_loss_{height}_model.txt', pathloss_list_model)
np.savetxt(f'data/path_loss/path_loss_{height}_data.txt', pathloss_list_data)

#np.savetxt(f'data/path_loss/first_pl_{height}_model.txt', pl_1path_model)
#np.savetxt(f'data/path_loss/first_pl_{height}_data.txt', pl_1path_data)


np.savetxt(f'data/delay/delay_{height}_model.txt', delay_list_model)
np.savetxt(f'data/delay/delay_{height}_data.txt', delay_list_data)

#np.savetxt(f'data/delay/first_delay_{height}_model.txt', dly_1path_model)
#np.savetxt(f'data/delay/first_delay_{height}_data.txt', dly_1path_data)


np.savetxt(f'data/link state/link state_{height}_model.txt', los_prob_model)
np.savetxt(f'data/link state/link state_{height}_data.txt', los_prob_data)

np.savetxt(f'data/link state/outage state_{height}_model.txt', outage_prob_model)
np.savetxt(f'data/link state/outage state_{height}_data.txt', outage_prob_data)


zoa_img_model = get_pdfs(dist2d_array, ZOA_model_mat-np.array(los_zoa)[:,None], feature='zoa')
zoa_img_data = get_pdfs(dist2d_array, ZOA_data_mat-np.array(los_zoa)[:,None], feature='zoa')
zod_img_model = get_pdfs(dist2d_array, ZOD_model_mat-np.array(los_zod)[:,None], feature='zod')
zod_img_data = get_pdfs(dist2d_array, ZOD_data_mat-np.array(los_zod)[:,None], feature='zod')

np.savetxt(f'data/zenith_angles/zod_{height}_model.txt', zod_img_model)
np.savetxt(f'data/zenith_angles/zod_{height}_data.txt', zod_img_data)
np.savetxt(f'data/zenith_angles/zoa_{height}_model.txt', zoa_img_model)
np.savetxt(f'data/zenith_angles/zoa_{height}_data.txt', zoa_img_data)

np.savetxt(f'data/azimuth_angles/aod_{height}_model.txt', aod_model)
np.savetxt(f'data/azimuth_angles/aod_{height}_data.txt', aod_data)
np.savetxt(f'data/azimuth_angles/aoa_{height}_model.txt', aoa_model)
np.savetxt(f'data/azimuth_angles/aoa_{height}_data.txt', aoa_data)
np.savetxt(f'data/azimuth_angles/phase_{height}_model.txt', phase_model)
np.savetxt(f'data/azimuth_angles/phase_{height}_data.txt', phase_data)

np.savetxt(f'data/RMS/delay_rms_{height}_data.txt', dly_rms_data)
np.savetxt(f'data/RMS/delay_rms_{height}_model.txt', dly_rms_model)

np.savetxt(f'data/RMS/aod_rms_{height}_data.txt', aod_rms_data)
np.savetxt(f'data/RMS/aod_rms_{height}_model.txt', aod_rms_model)

np.savetxt(f'data/RMS/aoa_rms_{height}_data.txt', aoa_rms_data)
np.savetxt(f'data/RMS/aoa_rms_{height}_model.txt', aoa_rms_model)
