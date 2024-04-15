# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 15:36:41 2023

@author: seongjoon kang
"""

import torch
import numpy as np
import pickle
import pandas as pd
import scipy.constants as cs

def gradient_penalty(critic,real, fake,cond, device = 'cpu'):
    BATCH_SIZE, C, H, W= real.shape
    epsilon = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    #BATCH_SIZE, input_size = real.shape
    #epsilon = torch.rand((BATCH_SIZE,1)).repeat(1, input_size).to(device)

    interpolated_images = real*epsilon + fake * (1-epsilon)
    
    # caculate critic scores 
    mixed_scores = critic(interpolated_images, cond)
    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
        )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        # Load data and get label    
        y = self.labels[index]

        return X, y


def process_data(img_file, dist2d_file, WINDOW_SIZE=5):
       
    with open(img_file,'rb') as f:
        img_data = pickle.load(f)
        img_data = np.array(img_data)[:,None]
        img_data = np.repeat(img_data, 8, -2)
        img_data = np.repeat(img_data, 2, -1)
  
    with open(dist2d_file,'rb') as f:
        dist2d_data = pickle.load(f)
        dist2d_data = np.array(dist2d_data)/100
        L = len(dist2d_data)
        dist2d_data = np.append(np.zeros(WINDOW_SIZE-1), dist2d_data)
        
    new_dist2d_data  = []
    for k in range(L):
        new_dist2d_data.append(dist2d_data[k:k+WINDOW_SIZE])
    return torch.tensor(img_data, dtype = torch.float32), torch.tensor(new_dist2d_data, dtype = torch.float32)

def get_feature_value(data, key, max_n_path = 25):
    feature_pd = pd.DataFrame()
    for i in range(max_n_path):
        
        if key == 'link state':
            df = data[key]
            feature_pd = pd.concat([feature_pd, df], axis =1)
        else:
            feature_pd = pd.concat([feature_pd, data[key+'_%d'%(i+1)]], axis =1)
    return feature_pd

def compute_linkstate_prob(_prob, dist2d, bin_size = 10, enable_first_prob = True):
    
    dist2d = np.array(dist2d)
    
    p_count = np.zeros(int(max(dist2d))+2)
    n_count = np.zeros(int(max(dist2d))+2)
    dist_2d = np.array(dist2d/bin_size, dtype = int)
    for j, d_2d_i in enumerate(dist_2d):
        if _prob[j]>0:
            p_count[d_2d_i] +=1
        else:
            n_count[d_2d_i] +=1
    _prob = p_count/(p_count + n_count)
   
    if enable_first_prob is True:
        _prob[0] = 1.0
    
    return _prob

def get_pdfs(dist2d, angle,  feature, min_dist = 30):
    #angle_list = []
    dist_bin_size = 10
    distance_bins = np.arange(min_dist, 900,step = dist_bin_size)
    
    if feature == 'zod' or feature =='zoa':
        angle_bin_size = 3
        angle_bins = np.arange(-90,93, step = angle_bin_size)
        #v_max = 0.4
    else:
        angle_bin_size = 5
        angle_bins = np.arange(-80, 80, step =angle_bin_size)
       # v_max = 0.2
    
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


def compute_rms_spread(chan_params, feature = 'delay'):
    # extract the feature data 
    d = np.array(chan_params[feature])
    if feature != 'delay':
        # convert from degrees to radians
        d = np.deg2rad(d)
    else:
        # compute 'excess delay'
        d = d - np.min(d)
            
    # derive path gains of multipaths
    path_loss = np.array(chan_params['path_loss'])
    path_gain_lin = 10**(-0.1*path_loss)
    # normalize path gains
    path_gain_normlzed = path_gain_lin/path_gain_lin.sum()
    # computer first order average with normalized path gains
    d_avg_first_order = (d*path_gain_normlzed).sum()
    # compute the second order average in the same way
    d_avg_second_order = ((d**2)*path_gain_normlzed).sum()
    # compute RMS
    rms = np.sqrt(d_avg_second_order - d_avg_first_order**2)
    return rms

'''
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        # Load data and get label    
        y = self.labels[index]

        return X, y
'''
