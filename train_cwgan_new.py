# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:14:57 2023

@author: seongjoon kang
"""
import torch
import torch.optim as optim
import torchvision
import subprocess, sys
#import torchvision.datasets as datasets
#import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from c_wgan import Generator, Critic, initialize_weight, upsample #,embedder
from utill import gradient_penalty, Dataset
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import shutil
from Herald_square_data.download_data import download_data
import argparse
from datetime import datetime
from scipy.stats import wasserstein_distance
from evaluate_cwgan import evaluate_cwgan
init_run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
print(f'############### run name : {init_run_name} ###############')
#from torch.optim.lr_scheduler import ExponentialLR

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", action="store_true")
parser.add_argument("--num_bins", type=int, default=32,
                    help="number of uniform bins N over the 2D distance range (must be even)")

parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size for training model")
parser.add_argument("--n_epoch", type=int, default=100,
                    help="Number of epochs for training model")

parser.add_argument("--lr_g", type=float, default=1e-4,
                    help="Leraning rate of generator for training model")

parser.add_argument("--lr_c", type=float, default=1e-4,
                    help="Leraning rate of critic for training model")

parser.add_argument("--split_ratio", type=float, default=0.5,
                    help="fraction p of each bin width allocated to training (0 < p < 1); "
                         "the lower p-fraction of every bin is used for training, the rest is held out")

parser.add_argument("--run_name", type=str, default=None,
                    help="name for this run (used in log dir and checkpoint); auto-generated if not given")

parser.add_argument("--seed", type=int, default=333,
                    help="random seed for reproducibility (numpy + torch); disabled if not given")

parser.add_argument("--num_workers", type=int, default=4, help="number of DataLoader worker processes")

parser.add_argument("--start_epochs", type=int, default=0, help="number of epochs to contnue training")

parser.add_argument ("--enable_tensorboard", action = "store_true", help = "record the logs to show them in tensorboard")

parser.add_argument("--disable_test_train_split", action ="store_true", help = "if it is True, the data is not split into test and train data")

args = parser.parse_args()

if args.run_name is not None:
    run_name = args.run_name
else:
    run_name = f"N{args.num_bins}_p{args.split_ratio}_" + datetime.now().strftime("%Y%m%d-%H%M%S")+'_ebin'

is_load = args.load_model
N_bins = args.num_bins
p_split = args.split_ratio
batch_size = args.batch_size
lr_g = args.lr_g
lr_c = args.lr_c  
n_epoch = args.n_epoch
start_epochs = args.start_epochs
enable_tensorboard = args.enable_tensorboard
disable_test_train_split = args.disable_test_train_split


assert 0.0 < p_split <= 1.0, "split_ratio must be in (0, 1)"
print(f'loading the saved model is {is_load}')
print(f'N = {N_bins} bins, split ratio p = {p_split}')

#if (is_load == False) and os.path.exists('logs'):
#    shutil.rmtree('logs')

# ----------------------------- Reproducibility -----------------------------
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f'random seed fixed to {args.seed}')

device = "cuda" if torch.cuda.is_available() else "cpu"
# Input sizes are fixed -> let cuDNN autotune the fastest conv algorithms.
torch.backends.cudnn.benchmark = True


#lr = 1e-4 # learning rate or you can set it as 5*1e-5. 
img_channel = 3
vertical_repeats=8
horizontal_repeats=2

z_dim = 25
critic_features = 32+5
generator_features = 32
critic_iteration = 5
lambda_gp= 5


print (f'LR of critic: {lr_c}, LR of generator:{lr_g}')
print(f'image channel={img_channel},  Batch size = {batch_size}, the device is {device}, n_epoch={n_epoch}')

data_dir = 'Herald_square_data'

if start_epochs ==0:
  saved_loc = f'save_model/cwgan_z_{z_dim}_dist2D_N{N_bins}_p{p_split}_ebin.pt'
else:
  saved_loc = f'save_model/cwgan_z_{z_dim}_dist2D_N{N_bins}_p{p_split}_{start_epochs}_ebin.pt'

os.makedirs('save_model', exist_ok=True)
file_path = f'{data_dir}/data_total.pickle'

if os.path.isfile(file_path) is False:
    download_data(pre_processed_only = True)
# read data
with open(file_path,'rb') as f:
    dataset_0 = pickle.load(f)
    
img_size = [dataset_0.shape[-2]*vertical_repeats, 
            dataset_0.shape[-1]*horizontal_repeats]

with open(f'{data_dir}/cond_total.pickle','rb') as f:
    cond_data_0 = pickle.load(f)

print(f'shape of uploaded condition and training data = {cond_data_0.shape} and {dataset_0.shape}')

dist3d = np.sqrt(cond_data_0[:,0]**2 + cond_data_0[:,1]**2)
ang = np.arccos(cond_data_0[:,1]/dist3d)
ang = np.rad2deg(ang)
cond = np.column_stack((cond_data_0[:,0]/1000,cond_data_0[:,1]/120, dist3d[:,None]/1000, ang/90))
cond_vec = torch.tensor(cond)
dataset= torch.tensor(dataset_0[:,None])

if disable_test_train_split is False:
  # ---- Quantile-based (equal-mass) train/held-out split over 2D distance ----
  # Paper notation: bin boundaries are set at empirical quantiles of S,
  # b_i = F^{-1}(i/N), so that every bin contains the same number of links.
  # Mapping each sample to its normalized rank u = rank/M in [0, 1) turns
  # equal-mass binning in distance into uniform binning in quantile space.
  # Within each bin, the lower p-fraction OF SAMPLES (not of bin width)
  # is used for training; train if rho = frac(u * N) < p.
  dist2d = cond_data_0[:, 0]
  M = len(dist2d)

  order = np.argsort(dist2d, kind='stable')
  ranks = np.empty(M, dtype=np.float64)
  ranks[order] = np.arange(M)
  u = ranks / M                      # empirical CDF value, in [0, 1)

  rho = u * N_bins - np.floor(u * N_bins)
  I = (rho < p_split)

  # quantile bin edges in meters (for logging / checkpoint / paper table)
  bin_edges = np.quantile(dist2d, np.linspace(0.0, 1.0, N_bins + 1))

  d_min, d_max = dist2d.min(), dist2d.max()
  print(f'### 2D distance range = [{d_min:.1f}, {d_max:.1f}] m')
  print(f'### quantile bin edges [m]: {np.round(bin_edges, 1)}')
  print(f'### samples per bin: {M // N_bins} (+/-1)')
  print(f'### training samples: {I.sum()} / {M} ({100 * I.sum() / M:.1f}%)')

  cond_vec_heldout = cond_vec[~I]
  cond_vec = cond_vec[I]

  dataset_heldout = dataset[~I]
  dataset = dataset[I]


print (f'### shape of conditions and data after splitting are {cond_vec.shape} and {dataset.shape}')
n_cond = cond_vec.shape[1] # the number of conditions
# parameters to evaluate model during training
eval_params = {
        'seed_list': [333],
        'z_dim': z_dim,
        'n_cond': n_cond,          # used only for constructing the generator
        'num_bins': N_bins,
        'split_ratio': p_split,
        'batch_size': 3000,
        'height_list': [1.6, 30, 60, 90, 120],
        'n_iterations': 5,
        'device': device,
        'verbose': False,
        'save_path': 'evaluation_data/wasserstein_dist.txt',
        'disable_test_train_split': disable_test_train_split    }

data_set_instances =  Dataset(dataset, cond_vec)
#loader = DataLoader(data_set_instances, batch_size=batch_size, shuffle=True)
loader = DataLoader(
    data_set_instances,
    batch_size=batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(args.num_workers > 0),
)

# initialize gen and disc/critic
gen = Generator(z_dim, img_channel, generator_features, n_cond = n_cond).to(device)
critic = Critic(img_channel, critic_features, 
                img_size =img_size , 
                n_cond = n_cond).to(device)

print("Num params of generator: ", sum(p.numel() for p in gen.parameters()))
print("Num params of critic: ", sum(p.numel() for p in critic.parameters()))


opt_gen = optim.Adam(gen.parameters(), lr=lr_g, betas = (0.5,0.9))
opt_critic = optim.Adam(critic.parameters(), lr=lr_c, betas = (0.5,0.9))
# schedulers are optional
scheduler1 = ExponentialLR(opt_gen, gamma=0.999)
scheduler2 = ExponentialLR(opt_critic, gamma=0.999)
#print(f'Exponential LR scheduler is employed {}')

# ---- Default values (used when no checkpoint exists or keys are missing) ----
loss_gen_list, loss_critic_list = [], []
ws_dist_list = []
ws_dist_test_list = []
corr_dist_test_list = []
best_test_loss = 100
step = 0
epoch_saved = 0

if is_load == True:
    PATH = saved_loc
    check_points = torch.load(PATH, weights_only=False)

    # ---- Required keys: fail immediately if missing, since resuming
    #      training without model weights is meaningless ----
    gen.load_state_dict(check_points['gen'])
    critic.load_state_dict(check_points['critic'])

    # ---- Optimizer/scheduler states: load if present, otherwise start fresh ----
    if 'opt_gen' in check_points:
        opt_gen.load_state_dict(check_points['opt_gen'])
    else:
        print('!!! opt_gen state not found; optimizer starts fresh')

    if 'opt_critic' in check_points:
        opt_critic.load_state_dict(check_points['opt_critic'])
    else:
        print('!!! opt_critic state not found; optimizer starts fresh')

    if 'scheduler_gen' in check_points:
        scheduler1.load_state_dict(check_points['scheduler_gen'])
    if 'scheduler_critic' in check_points:
        scheduler2.load_state_dict(check_points['scheduler_critic'])

    # ---- Optional keys: keep the defaults defined above if missing ----
    loss_gen_list       = check_points.get('loss_gen', loss_gen_list)
    loss_critic_list    = check_points.get('loss_critic', loss_critic_list)
    ws_dist_list        = check_points.get('ws_dist', ws_dist_list)
    ws_dist_test_list   = check_points.get('ws_dist_test', ws_dist_test_list)
    corr_dist_test_list = check_points.get('corr_dist_test', corr_dist_test_list)
    best_test_loss      = check_points.get('best_test_loss', best_test_loss)
    epoch_saved               = check_points.get('epoch', epoch_saved)
    step                = check_points.get('global_step', step)

    # ---- Log which keys were missing from the checkpoint ----
    expected_keys = ['gen', 'critic', 'opt_gen', 'opt_critic',
                     'scheduler_gen', 'scheduler_critic',
                     'loss_gen', 'loss_critic', 'ws_dist',
                     'ws_dist_test', 'corr_dist_test',
                     'best_test_loss', 'epoch', 'global_step']
    missing = [k for k in expected_keys if k not in check_points]
    if missing:
        print(f'!!! missing keys in checkpoint (defaults used): {missing}')

    print('-----------------------------------------------------')
    print(f'################### the saved model is loaded from {PATH}, previous best loss:{best_test_loss} ##################')
    print('------------------------------------------------------')
    best_test_loss = 0.205
else:
    # Weight initialization must run ONLY when training from scratch;
    # otherwise it would overwrite the loaded checkpoint weights.
    initialize_weight(gen)
    initialize_weight(critic)

# for tensorboard plotting
if enable_tensorboard:
  writer_real = SummaryWriter(f"logs/{run_name}/real")
  writer_fake = SummaryWriter(f"logs/{run_name}/fake")
  writer_loss = SummaryWriter(f"logs/{run_name}/loss")

for epoch in range(epoch_saved +1, n_epoch):
          
      loss_gen_sum, loss_critic_sum, w_dist_sum = 0,0, 0
      # Target labels not needed! <3 unsupervised
      _ws_test_list, _corr_test_list = [], []

      for batch_idx, (real, cond) in enumerate(loader):
        gen.train()
        critic.train()
        cond = cond.to(device, dtype = torch.float, non_blocking = True)
        real = real.to(device, dtype = torch.float, non_blocking = True)
        real = upsample(real, vertical_repeats=8, horizontal_repeats=2, img_channel =3)
        
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
            loss_critic.backward(retain_graph= True)
            opt_critic.step()
   
        w_dist = -(torch.mean(critic_real) - torch.mean(critic_fake))
        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, cond).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        loss_gen_sum += loss_gen.item()
        loss_critic_sum += loss_critic.item()
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            lr_gen = round(opt_gen.param_groups[0]["lr"],5)
            lr_critic =round(opt_critic.param_groups[0]["lr"], 5)

            print(f"Epoch [{epoch}/{n_epoch}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}, LR_G:{lr_gen}, LR_D:{lr_critic}")
    
            with torch.no_grad():
                #eval_params['save_path'] = f'save_model/ws_dist/wasserstein_dist_N{N_bins}_p{p_split}_ep{epoch}.txt'
                #eval_params['save_path'] = f'save_model/wasserstein_dist_{epoch}.txt',
                w_dist_test, corr_dist_test = evaluate_cwgan(gen, cond_data_0, dataset_0, eval_params)
              
                _ws_test_list.append(w_dist_test.item()) 
                _corr_test_list.append(corr_dist_test.item())
                #w_dist_test = (torch.mean(critic_real_test) - torch.mean(critic_fake_test))
                
                print(f'### WS-distance from TEST is {round(w_dist_test.item(), 5)}, ### corr-distance is {round(corr_dist_test.item(),5)}')
                ws_dist_test_list.append(w_dist_test.item())
                corr_dist_test_list.append(corr_dist_test.item())

                if enable_tensorboard:
                  img_grid_real = torchvision.utils.make_grid(
                      real[:32, 1:2,:,:], normalize=True)
                  img_grid_fake = torchvision.utils.make_grid(
                      fake[:32, 1:2,:,:], normalize=True)

                  writer_loss.add_scalar("Critic Loss",  np.abs(loss_critic.item()),  global_step=step)
                  writer_loss.add_scalar("Generator Loss", loss_gen.item(), global_step=step)
                  writer_loss.add_scalar("Wasserstein distance", (critic_real.mean() - critic_fake.mean()).item(), global_step = step)
                  writer_loss.add_scalar("Correlation distance", corr_dist_test.item(), global_step = step)
                  writer_loss.add_scalar("Wasserstein distance for Test", w_dist_test.item(), global_step = step)
                  writer_loss.add_scalar('epoch', epoch, global_step = step)
                  writer_real.add_image("Real", img_grid_real, global_step=step)
                  writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                  writer_loss.flush()
                  step += 1            

        loss_gen_list.append(loss_gen.item())
        loss_critic_list.append(loss_critic.item())
        ws_dist_list.append(w_dist.item())
        
      scheduler1.step()
      scheduler2.step()
      
      avg_gen_loss = np.round(loss_gen_sum/len(loader),3)
      avg_critic_loss = np.round(loss_critic_sum/len(loader),3)
      avg_ws_dist = np.round(np.mean(_ws_test_list),5)
      print(f'--- average LOSS after one epoch: gen:{avg_gen_loss},  critic:{avg_critic_loss}, average W-dist for test data: {avg_ws_dist}')
      print(f'---------- best loss from TEST is {best_test_loss} -------')

      mean_loss = np.round(np.mean(_ws_test_list), 6)
      
      #if  (best_test_loss >mean_loss) or ((best_test_loss < mean_loss) and np.abs(best_test_loss-mean_loss)) < 7e-4 or (best_test_loss == 100):
      if epoch>12:
        if mean_loss < best_test_loss:
          best_test_loss = mean_loss
          print(f"the updated best loss ={np.round(best_test_loss,4)} ")
 
        if epoch >=12:
          print(f"************************************* New model is saved: best loss ={np.round(best_test_loss,4)} ***************************************")
          saved_loc = f'save_model/cwgan_z_{z_dim}_dist2D_N{N_bins}_p{p_split}_{epoch}_ebin.pt'
          torch.save({
              'gen': gen.state_dict(),
              'critic': critic.state_dict(),
              'opt_gen': opt_gen.state_dict(),
              'opt_critic': opt_critic.state_dict(),
              'scheduler_gen':scheduler1.state_dict(),
              'scheduler_critic':scheduler2.state_dict(),
              'epoch': epoch,
              'best_test_loss':best_test_loss,
              'loss_critic': loss_critic_list,
              'loss_gen': loss_gen_list,
              'ws_dist': ws_dist_list,
              'ws_dist_test':ws_dist_test_list,
              'corr_dist_test':corr_dist_test_list,
              # ---- train/held-out split metadata ----
              'd_min': float(d_min),
              'd_max': float(d_max),
              'bin_edges': bin_edges.tolist(),
              'split_type': 'quantile',
              'N_bins': N_bins,
              'p_split': p_split,
              'global_step':step,
          }, saved_loc)

    #loss_gen_list.append(loss_gen_sum/len(loader))
    #loss_critic_list.append(loss_critic_sum/len(loader))   

if enable_tensorboard:
  writer_loss.close()
  writer_real.close()
  writer_fake.close()

if disable_test_train_split is False:
  
  # ----------------------- Automatic evaluation after training -----------------------
  eval_cmd = [
      sys.executable, "EM_distance_eval_v2.py",
      "--seed", "300", "10", "900",
      "--n_epoch", "0",              # this parameter is useful only when is_all is False
      "--split_ratio", str(p_split),   
      "--num_bins", str(N_bins),       
      "--is_all",
  ]

  print("=" * 60)
  print("Training finished. Starting evaluation:")
  print(" ".join(eval_cmd))
  print("=" * 60)


  del gen, critic
  torch.cuda.empty_cache()
  result = subprocess.run(eval_cmd)

  if result.returncode != 0:
      print(f"### Evaluation failed with return code {result.returncode}")
  else:
      print("### Evaluation completed successfully.")

      
      
               
