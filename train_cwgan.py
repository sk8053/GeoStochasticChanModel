# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:14:57 2023

@author: seongjoon kang
"""
import torch
import torch.optim as optim
import torchvision
#import torchvision.datasets as datasets
#import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from c_wgan import Generator, Critic, initialize_weight #,embedder
from utill import gradient_penalty, Dataset
import numpy as np
import pickle

import os
import shutil
#from torch.optim.lr_scheduler import ExponentialLR

if os.path.exists('logs'):
    shutil.rmtree('logs')
# Hyperparameters etc
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

load = True
lr = 1e-4 # learning rate

img_size = [64, 50]
img_channel = 1
z_dim = 25
n_epoch = 12
critic_features = 32+3
generator_features = 32
critic_iteration = 5
lambda_gp= 5
batch_size = 256
vertical_repeats = 8
horizontal_repeats = 2

data_dir = 'Herald_square_data'
saved_loc = f'save_model/cwgan_z_{z_dim}.pt'

# read data
with open(f'{data_dir}/data_total.pickle','rb') as f:
    dataset = pickle.load(f)
img_size = [dataset.shape[-2]*vertical_repeats, 
            dataset.shape[-1]*horizontal_repeats]

with open(f'{data_dir}/cond_total.pickle','rb') as f:
    cond = pickle.load(f)

print ('data is loaded, shape is ', dataset.shape)


dist3d = np.sqrt(cond[:,0]**2 + cond[:,1]**2)
ang = np.arccos(cond[:,1]/dist3d)
ang = np.rad2deg(ang)
cond = np.column_stack((cond[:,0]/1000,cond[:,1]/120, dist3d[:,None]/1000, ang/90))
cond_vec = torch.tensor(cond)

print ('conditions is loaded, shape is ', cond_vec.shape)

dataset= torch.tensor(dataset[:,None])
n_cond = cond_vec.shape[1] 

data_set_instances =  Dataset(dataset, cond_vec)
loader = DataLoader(data_set_instances, batch_size=batch_size, shuffle=True)

# initialize gen and disc/critic
gen = Generator(z_dim, img_channel, generator_features, n_cond = n_cond).to(device)
critic = Critic(img_channel, critic_features, 
                img_size =img_size , 
                n_cond = n_cond).to(device)

print("Num params of generator: ", sum(p.numel() for p in gen.parameters()))
print("Num params of critic: ", sum(p.numel() for p in critic.parameters()))


opt_gen = optim.Adam(gen.parameters(), lr=lr, betas = (0.5,0.9))
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas = (0.5,0.9))
# schedulers are optional
#scheduler1 = ExponentialLR(opt_gen, gamma=0.95)
#scheduler2 = ExponentialLR(opt_critic, gamma=0.95)

if load == True:
    PATH = saved_loc
    gen.load_state_dict(torch.load(PATH)['gen'])
    critic.load_state_dict(torch.load(PATH)['critic'])
    opt_gen.load_state_dict(torch.load(PATH)['opt_gen'])
    opt_critic.load_state_dict(torch.load(PATH)['opt_critic'])
else:
    initialize_weight(gen)
    initialize_weight(critic)
# for tensorboard plotting
#fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
#writer_cond_ch = SummaryWriter(f"logs/cond_check")
step = 0

gen.train()
critic.train()

best_loss_critic = -1e5
best_loss_gen = 1e5

loss_gen_list, loss_critic_list = [], []          
for epoch in range(n_epoch):
    
    if epoch!=0 and epoch % 1 ==0:
        print("=============== model is saved ====================")
        torch.save({
            'gen':gen.state_dict(),
            'critic':critic.state_dict(),
             'opt_gen':opt_gen.state_dict(),
             'opt_critic':opt_critic.state_dict(),
             'epoch':epoch,
             'loss_critc':loss_critic_list,
             'loss_gen':loss_gen_list,
            }, saved_loc)
        
        
    loss_gen_sum, loss_critic_sum = 0,0
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, cond) in enumerate(loader):
        cond = cond.to(device, dtype = torch.float)
        real = real.to(device, dtype = torch.float)
        real = torch.repeat_interleave(real, vertical_repeats, dim = -2)
        real = torch.repeat_interleave(real, horizontal_repeats, dim = -1) # 64*50
        batch_size_now = real.shape[0]
       
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(critic_iteration):
            noise = torch.randn(batch_size_now, z_dim, 1, 1).to(device)
            fake = gen(noise, cond)
            
            critic_real = critic(real, cond).reshape(-1)
            critic_fake = critic(fake, cond).reshape(-1)
            gp = gradient_penalty(critic, real, fake,cond, device = device)
            loss_critic =( -(torch.mean(critic_real) - torch.mean(critic_fake))
            + lambda_gp*gp)
            opt_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
           

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, cond).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        loss_gen_sum += loss_gen.item()
        loss_critic_sum += loss_critic.item()
        # Print losses occasionally and print to tensorboard
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            #critic.eval()
            print(
                f"Epoch [{epoch}/{n_epoch}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")
    
            with torch.no_grad():
                fake = gen(noise, cond)
                
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            gen.train()
            #critic.train()
    
    loss_gen_list.append(loss_gen_sum/len(loader))
    loss_critic_list.append(loss_critic_sum/len(loader))
    # scheudulers are optional
    #scheduler1.step()
    #scheduler2.step()

torch.save({
    'gen':gen.state_dict(),
    'critic':critic.state_dict(),
     'opt_gen':opt_gen.state_dict(),
     'opt_critic':opt_critic.state_dict(),
     'epoch':epoch,
     'loss_critc':loss_critic_list,
     'loss_gen':loss_gen_list,
    }, saved_loc)
      
      
               
