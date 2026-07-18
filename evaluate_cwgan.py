# -*- coding: utf-8 -*-
"""
Evaluation function for conditional WGAN-GP channel model.
Refactored from the original script: model loading / argparse removed,
everything is passed in as arguments.

@author: seongjoon kang
"""
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
from scipy.stats import wasserstein_distance as w_dist


def evaluate_cwgan(gen, cond_data_0, dataset_0, params: dict):
    """
    Evaluate a trained conditional WGAN generator on the held-out portion
    of quantile (equal-mass) bins and return the average Wasserstein distance.

    Parameters
    ----------
    gen : torch.nn.Module
        Trained generator (already constructed; PATH loading is not needed).
    cond_data_0 : np.ndarray, shape (M, >=2)
        Condition data; column 0 = 2D distance [m], column 1 = RX height [m].
    dataset_0 : np.ndarray
        Channel image data aligned with cond_data_0.
    params : dict
        Required / optional keys (defaults in parentheses):
            'seed_list'    : list of int, random seeds ([333])
            'z_dim'        : latent dimension (25)
            'num_bins'     : number of equal-mass bins N (8)
            'split_ratio'  : training fraction p per bin (0.5)
            'batch_size'   : samples per evaluation batch (3000)
            'height_list'  : RX heights to evaluate ([1.6, 30, 60, 90, 120])
            'n_iterations' : generation repetitions per height (10)
            'device'       : 'cuda' / 'cpu' (auto-detect)
            'verbose'      : print intermediate results (True)
            'save_path'    : if given, save wdist table with np.savetxt (None)
        Note: 'n_cond' is only needed when constructing the generator,
        so it is not used inside this function.

    Returns
    -------
    float
        np.mean(wdist_list_h): Wasserstein distance averaged over all
        seeds, heights, and channel parameters.
    """
    # ---------------- extract parameters ----------------
    seed_list = params.get('seed_list', [333])
    if isinstance(seed_list, int):
        seed_list = [seed_list]
    z_dim = params.get('z_dim', 25)
    N_bins = params.get('num_bins', 8)
    split_ratio = params.get('split_ratio', 0.5)
    batch_size = params.get('batch_size', 3000)
    height_list = params.get('height_list', [1.6, 30, 60, 90, 120])
    n_iterations = params.get('n_iterations', 10)
    device = params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    verbose = params.get('verbose', True)
    save_path = params.get('save_path','evaluation_data/wasserstein_dist.txt' )
    disable_test_train_split = params.get('disable_test_train_split', False)

    gen = gen.to(device)
    gen.eval()

    wdist_list_h = []
    mean_corr_dist = []

    keys = ['path_loss', 'delay', 'zoa', 'aoa', 'zod', 'aod', 'phase']
    table_keys = ['path_loss', 'delay', 'aod', 'zod', 'aoa', 'zoa', 'phase']

    for seed_i, seed in enumerate(seed_list):
        if verbose:
            print(f' seed = {seed}')
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        for height in height_list:

            I_h = (cond_data_0[:, 1] == height)
            cond_data = cond_data_0[I_h]
            dataset = dataset_0[I_h]

            dist3d = np.sqrt(cond_data[:, 0]**2 + cond_data[:, 1]**2)
            ang = np.rad2deg(np.arccos(cond_data[:, 1] / dist3d))
            cond = np.column_stack((cond_data[:, 0] / 1000,
                                    cond_data[:, 1] / 120,
                                    dist3d[:, None] / 1000,
                                    ang / 90))

            cond_vec2 = torch.tensor(cond)
            dataset2 = torch.tensor(dataset[:, None])

            if disable_test_train_split is False:
              # -------- quantile (equal-mass) bin split --------
              dist2d = cond_data[:, 0]
              M = len(dist2d)

              order = np.argsort(dist2d, kind='stable')
              ranks = np.empty(M, dtype=np.float64)
              ranks[order] = np.arange(M)
              u = ranks / M                       # empirical CDF value in [0, 1)

              rho = u * N_bins - np.floor(u * N_bins)
              I = (rho < split_ratio)

              # quantile bin edges in meters (logging / paper table)
              bin_edges = np.quantile(dist2d, np.linspace(0.0, 1.0, N_bins + 1))
              d_min, d_max = dist2d.min(), dist2d.max()

              if verbose and height == height_list[0] and seed_i == 0:
                  print(f'### 2D distance range = [{d_min:.1f}, {d_max:.1f}] m')
                  print(f'### quantile bin edges [m]: {np.round(bin_edges, 1)}')
                  print(f'### samples per bin: {M // N_bins} (+/-1)')
                  print(f'### training samples: {I.sum()} / {M} '
                        f'({100 * I.sum() / M:.1f}%)')
              
              # held-out (test) data
              cond_vec2 = cond_vec2[~I]
              dataset2 = dataset2[~I]

            #else:
            #  indices = torch.randperm(cond_data_0.shape[0])[:1000]
            #  cond_vec2 = cond_vec2[indices]
            #  dataset2 = dataset2[indices]

            if height == height_list[0]:
                indices = torch.randperm(cond_vec2.shape[0])[:batch_size]
            
            model_data_all, origin_data_all = [], []

            with torch.no_grad():
                for _ in range(n_iterations):
                    noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                    cond_vec_test = cond_vec2[indices].to(device)

                    model_data_0 = gen(noise.float(), cond_vec_test.float())
                    model_data_0 = model_data_0.detach().cpu().numpy()

                    _fake1 = np.zeros((batch_size, 8, 50))
                    for i in range(8):
                        _fake1[:, i] = model_data_0[:, 0, i * 8 + 5, :]

                    model_data = np.zeros((batch_size, 8, 25))
                    for j in range(25):
                        model_data[:, :, j] = _fake1[:, :, j * 2 + 1]

                    original_data = torch.squeeze(dataset2[indices].to(device))
                    original_data = original_data.detach().cpu().numpy()

                    model_data_all.append(model_data)
                    origin_data_all.append(original_data)

            model_data_all = np.array(model_data_all).reshape(-1, 8, 25)
            origin_data_all = np.array(origin_data_all).reshape(-1, 8, 25)

            # -------- correlation distance --------
            corr_model = np.corrcoef(model_data_all.reshape(-1, 8 * 25).T)
            corr_data = np.corrcoef(origin_data_all.reshape(-1, 8 * 25).T)
            corr_dist = 1 - np.trace(corr_data.dot(corr_model)) / (
                np.linalg.norm(corr_data, ord='fro')
                * np.linalg.norm(corr_model, ord='fro'))

            # -------- Wasserstein distances --------
            wdist_dict = dict()
            for i, key in enumerate(keys):
                dist = w_dist(model_data_all[:, i, :].reshape(-1),
                              origin_data_all[:, i, :].reshape(-1))
                wdist_dict[table_keys[i]] = round(float(dist), 5)

            if verbose:
                print(f'height = {height}')
                print(wdist_dict)

            wdist_list_h.append(list(wdist_dict.values()))
            mean_corr_dist.append(corr_dist)

    wdist_list_h = np.array(wdist_list_h).reshape(
        len(seed_list) * len(height_list), -1)

    if verbose:
        print(f'mean correlation distance: {np.mean(mean_corr_dist)}')
        #print(f'average EM distance per height: {np.mean(wdist_list_h, axis=1)}')
        print(f'average EM distance: {np.mean(wdist_list_h)}')

    #if save_path is not None:
    #    np.savetxt(save_path, wdist_list_h, fmt='%.3f')

    return np.mean(wdist_list_h), np.mean(mean_corr_dist)


# ---------------- example usage ----------------
if __name__ == '__main__':
    import pickle
    from c_wgan import Generator

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params = {
        'seed_list': [333, 10, 900],
        'z_dim': 25,
        'n_cond': 4,          # used only for constructing the generator
        'num_bins': 8,
        'split_ratio': 0.75,
        'batch_size': 3000,
        'height_list': [1.6, 30, 60, 90, 120],
        'n_iterations': 10,
        'device': device,
        'verbose': True,
        'save_path': 'evaluation_data/wasserstein_dist.txt',
    }

    PATH = 'save_model/cwgan_z_25_dist2D_N8_p0.75_22_ebin.pt'
    gen = Generator(params['z_dim'], 3, 32, n_cond=params['n_cond']).to(device)
    ckpt = torch.load(PATH, weights_only=False)
    gen.load_state_dict(ckpt['gen'])
    print(f"===== epoch {ckpt['epoch']} =======")

    data_dir = 'Herald_square_data'
    with open(f'{data_dir}/data_total.pickle', 'rb') as f:
        dataset_0 = pickle.load(f)
    with open(f'{data_dir}/cond_total.pickle', 'rb') as f:
        cond_data_0 = pickle.load(f)

    mean_wdist = evaluate_cwgan(gen, cond_data_0, dataset_0, params)
    print(f'returned mean EM distance = {mean_wdist}')
