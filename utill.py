# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 15:36:41 2023

@author: gangs
"""

import torch
import numpy as np
import pickle
import pandas as pd

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

def compute_LOS_prob(los_prob, dist2d, bin_size = 10):
    
    dist2d = np.array(dist2d)
    
    LOS_count = np.zeros(int(max(dist2d))+2)
    NLOS_count = np.zeros(int(max(dist2d))+2)
    dist_2d = np.array(dist2d/bin_size, dtype = int)
    for j, d_2d_i in enumerate(dist_2d):
        if los_prob[j]>0:
            LOS_count[d_2d_i] +=1
        else:
            NLOS_count[d_2d_i] +=1
    los_prob = LOS_count/(LOS_count + NLOS_count)
    los_prob[0] = 1.0
    return los_prob

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