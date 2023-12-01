# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:10:44 2023

@author: seongjoon kang
"""
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
import json
import scipy.constants as cs

files = glob.glob('*.csv')
total_df = None
for i, file in enumerate(files):
    df = pd.read_csv(file)
    if i ==0:
        total_df = df
    else:
        total_df = pd.concat([total_df, df], axis = 0, ignore_index = True)
        
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
    angle_dict = {'aod':LOS_AOD_phi,'zod':LOS_ZOD_theta, 'aoa':LOS_AOA_phi, 'zoa':LOS_ZOA_theta}
    return angle_dict

def get_feature_value(data, key, fc = 12e9):
        
    feature_pd = pd.DataFrame()
    for i in range(25):
        feature_pd = pd.concat([feature_pd, data[key+'_%d'%(i+1)]], axis =1)
    tx_x, tx_y, tx_z = data['tx_x'], data['tx_y'], data['tx_z']
    rx_x, rx_y, rx_z = data['rx_x'], data['rx_y'], data['rx_z']
    
    if key == 'path_loss' or key == 'delay':
        
        dist_vect_3d = np.column_stack((tx_x-rx_x, tx_y-rx_y, tx_z-rx_z))
        distance_3D = np.linalg.norm(dist_vect_3d, axis = 1)
        
        fspl = 20*np.log10(distance_3D) + 20*np.log10(fc) -147.55
        min_delay = distance_3D/cs.speed_of_light
        
        if key == 'path_loss':
            feature_pd -= fspl[:,None]
        elif key == 'delay':
            feature_pd -= min_delay[:,None]
    '''
    elif key == 'zod' or key =='zoa' or key =='aoa' or key == 'aod':
        
        tx_loc = np.column_stack((tx_x.to_numpy(), tx_y.to_numpy(), tx_z.to_numpy()))
        rx_loc = np.column_stack((rx_x.to_numpy(), rx_y.to_numpy(), rx_z.to_numpy()))
        angel_dict = compute_LOS_angle(tx_loc, rx_loc)
        feature_pd -= angel_dict[key][:,None]
        print(key)
    '''    
    return feature_pd

max_min_dict = dict()
keys = ['path_loss', 'delay', 'zoa', 'zod','aoa','aod', 'phase']
fc = 12e9
for key in tqdm(keys):    
    f_pd = get_feature_value(total_df, key, fc = fc)
    f_pd = f_pd.to_numpy().reshape(-1)
    max_, min_ = np.nanmax(f_pd),np.nanmin(f_pd)
    max_min_dict[f'{key}_max'] = max_
    max_min_dict[f'{key}_min'] = min_

with open('max_min.json', 'w') as out:
    json.dump(max_min_dict, out)



    
    
