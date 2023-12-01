# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:52:24 2023

@author: seongjoon kang
"""
import matplotlib

matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import numpy as np

height = 90
feature = 'zoa'
for height in [1.6, 30, 60, 90, 120]:#, 30, 60, 90, 120]:
    zoa_data = np.loadtxt(f'data/zenith_angles/{feature}_{height}_data.txt')
    zoa_model = np.loadtxt(f'data/zenith_angles/{feature}_{height}_model.txt')
    
    fig, axs = plt.subplots(2)
    #fig.suptitle('Vertically stacked subplots')
    
    pcm = axs[0].imshow(zoa_data, cmap='jet', vmax = 0.1)
    axs[0].set_yticks(np.arange(10, zoa_data.shape[0], step=10), np.arange(-60,60+30, step=30))
    axs[0].set_xticks([])#(np.arange(zoa_data.shape[1]+1, step=20), np.arange(1300, step=200)) 
    axs[0].set_title('Data', fontsize = 13)
    
    pcm = axs[1].imshow(zoa_model, cmap='jet', vmax = 0.1)
    axs[1].set_yticks(np.arange(10, zoa_data.shape[0], step=10), np.arange(-60,60+30, step=30))
    axs[1].set_xticks(np.arange(zoa_data.shape[1], step=20), np.arange(1200, step=200))
    axs[1].set_title('Model', fontsize = 13)
    axs[1].set_xlabel('2D distance [m]', fontsize = 13)
    
    #plt.subplots_adjust(wspace =0, hspace = 0)
    plt.tight_layout()
    if height == 120:
        cb =plt.colorbar(pcm, ax = axs.ravel().tolist(), shrink = 0.78,pad = 0.02, 
                     location = 'right',cmap = 'jet')
        cb.set_label(label = 'PDF', size=10,weight='bold')
        cb.ax.tick_params(labelsize=10)
    elif height == 1.6:
        fig.text(0.195, 0.5, r'Angle relative to LOS direction [$\circ$]', 
                 va='center',rotation = 'vertical', fontsize = 13)
    
    #plt.tight_layout()
    plt.savefig(f'data/figures/{feature}_{height}.png', dpi = 800, bbox_inches='tight')
