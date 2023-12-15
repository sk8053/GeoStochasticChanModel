# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:03:00 2023

@author: seongjoon kang
"""

import matplotlib

matplotlib.rc('xtick', labelsize=11) 
matplotlib.rc('ytick', labelsize=11) 
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import numpy as np

fig, axs = plt.subplots(2,5, figsize = (10,2.5))
feature = 'zod'
for i, height in enumerate([1.6, 30, 60, 90, 120]):#, 30, 60, 90, 120]:
    _data = np.loadtxt(f'evaluation_data/zenith_angles/{feature}_{height}_data.txt')
    _model = np.loadtxt(f'evaluation_data/zenith_angles/{feature}_{height}_model.txt')
    
    _data = _data[10:-10][:,:60]
    _model = _model[10:-10][:,:60]
    
    
    #fig.suptitle('Vertically stacked subplots')
    
    pcm = axs[0,i].imshow(_data, cmap='jet', vmax = 0.1)
    
    axs[0,i].set_xticks([])#(np.arange(zoa_data.shape[1]+1, step=20), np.arange(1300, step=200)) 
    axs[0,i].set_title(f'Data at {height}m', fontsize = 12)
    
    pcm = axs[1,i].imshow(_model, cmap='jet', vmax = 0.1)
    
    axs[0,i].set_yticks(np.array([ 0, 9, 19, 29, 39]), np.arange(-60,60+30, step=30))
    axs[1,i].set_yticks(np.array([ 0, 9, 19, 29, 39]), np.arange(-60,60+30, step=30))
    if height !=1.6:
        axs[0,i].set_yticklabels([])
        axs[1,i].set_yticklabels([])
        
    axs[1,i].set_xticks(np.array([ 0, 20, 40, 59]), np.arange(700, step=200))
    axs[1,i].set_title(f'Model at {height}m', fontsize = 12)
    #axs[1,i].set_xlabel('2D distance [m]', fontsize = 13)
    
    #plt.subplots_adjust(wspace =0, hspace = 0)
    #plt.tight_layout()
 
    if height == 120:
        cb =plt.colorbar(pcm, ax = axs.ravel().tolist(), shrink = 0.78,pad = 0.02, 
                     location = 'right',cmap = 'jet')
        #cb.set_label(label = 'PDF', size=10, weight='bold',x=0.8, y =1.15, rotation =0)
        cb.ax.set_title('PDF', size = 10, weight = 'bold')
        cb.ax.tick_params(labelsize=10)
        fig.text(0.4, 0.01, r'2D distance [m]', 
                 va='center',rotation = 'horizontal', fontsize = 12, weight = 'bold')
    elif height == 1.6:
        fig.text(0.075, 0.5, r'Relative angle [$\circ$]', 
                 va='center',rotation = 'vertical', fontsize = 12, weight = 'bold')


plt.savefig(f'evaluation_data/figures/all_{feature}.png', dpi = 800, bbox_inches='tight')
