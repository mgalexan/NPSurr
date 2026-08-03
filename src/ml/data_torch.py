import torch as t
from torch.utils.data import Dataset
import numpy as np

from constants import MLParameters

"""
Dataset handling code for surrogate model training and evaluation.
"""

def log_transform(data, scale: float):
    """
    Log-transform the data with a given scale.
    """
    if isinstance(data, np.ndarray):
        return np.log1p(np.clip(data, 0.0, None) / scale)
    elif isinstance(data, t.Tensor):
        return t.log1p(t.clip(data, 0.0, None) / scale)

def inverse_log_transform(data, scale: float):
    """
    Inverse log-transform the data with a given scale.
    """
    if isinstance(data, np.ndarray):
        return np.expm1(data) * scale
    elif isinstance(data, t.Tensor):
        return t.expm1(data) * scale

def normalize(data, mean: float, std: float):
    """
    Normalize the data with a given mean and standard deviation.
    """
    return (data - mean) / std

def inverse_normalize(data, mean: float, std: float):
    """
    Inverse normalize the data with a given mean and standard deviation.
    """
    return data * std + mean

def stratified_sample_with_weights(CI_grid, n_unif, n_strat, rng):
    """
    Stratified sampling with logarithmic weighting based on CI values.
    """
    Nt, Nr = CI_grid.shape
    N_grid = Nt * Nr
    
    t_grid = np.arange(Nt)
    r_grid = np.arange(Nr)
    T_g, R_g = np.meshgrid(t_grid, r_grid, indexing='ij')
    
    t_flat = T_g.ravel()
    r_flat = R_g.ravel()
    ci_flat = CI_grid.ravel()
    
    idx_unif = rng.choice(N_grid, size=min(n_unif, N_grid), replace=False)
    
    ci64 = ci_flat.astype(np.float64)
    ci_max = ci64.max()
    
    if ci_max > 0.0:
        raw_w = np.log1p(np.clip(ci64, 0.0, None) / ci_max)
    else:
        raw_w = np.ones(N_grid, dtype=np.float64)
    
    w_sum = raw_w.sum()
    if w_sum <= 0.0:
        weights = np.ones(N_grid, dtype=np.float64) / N_grid
    else:
        weights = raw_w / w_sum
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()
    
    idx_strat = rng.choice(N_grid, size=min(n_strat, N_grid), replace=False, p=weights)
    
    indices = np.unique(np.concatenate([idx_unif, idx_strat]))
    
    return indices, r_flat[indices], t_flat[indices]

class SimulationDataset(Dataset):
    """
    PyTorch Dataset for the simulation data with intelligent stratified sampling.
    
    Uses logarithmic weighting to oversample high-concentration regions.
    """
    def __init__(self, params: MLParameters, data: dict):
        self.data = data
        self.r = self.data['r']
        self.t_out = self.data['t_out']
        self.CN = self.data['CN']
        self.CF = self.data['CF']
        self.CIN = self.data['CIN']
        self.CI = self.data['CI']
        self.d_val_m = self.data['d_val_m']
        self.k_rel = self.data['k_rel']
        self.k_rel_log = log_transform(self.k_rel, params.k_rel_scale)
        self.C_I_log = log_transform(self.CI, params.CI_scale)

        self.params = params

        self.d_mean = np.mean(self.d_val_m)
        self.d_std = np.std(self.d_val_m) if len(self.d_val_m) > 1 else 1.0
        self.k_rel_mean = np.mean(self.k_rel)
        self.k_rel_std = np.std(self.k_rel_log) if len(self.k_rel_log) > 1 else 1.0
        self.r_mean = np.mean(self.r)
        self.r_std = np.std(self.r) + 1e-30
        self.t_out_mean = np.mean(self.t_out)
        self.t_out_std = np.std(self.t_out) + 1e-30
        self.CI_mean = np.mean(self.C_I_log)
        self.CI_std = np.std(self.C_I_log) + 1e-30

        self.norm = {
            "d": [self.d_mean, self.d_std],
            "k_rel": [self.k_rel_mean, self.k_rel_std],
            "t": [self.t_out_mean, self.t_out_std],
            "r": [self.r_mean, self.r_std],
            "CI": [self.CI_mean, self.CI_std]
        }
        
        self._generate_stratified_indices()
    
    def _generate_stratified_indices(self):
        """
        Generate stratified sampling indices for all (d, k_rel) pairs.
        Stores per-parameter-pair indices for intelligent sampling.
        """
        self.indices_by_pair = {}
        self.flat_indices = []
        self.pair_mapping = {}  #
        
        rng = np.random.default_rng(42)
        flat_idx = 0
        
        for d_idx in range(len(self.d_val_m)):
            for k_rel_idx in range(len(self.k_rel)):
                CI_grid = self.CI[d_idx, k_rel_idx, :, :]
                
                indices, _, _ = stratified_sample_with_weights(
                    CI_grid, self.params.n_unif, self.params.n_strat, rng
                )
                
                self.indices_by_pair[(d_idx, k_rel_idx)] = indices
                
                for grid_idx in indices:
                    self.pair_mapping[flat_idx] = (d_idx, k_rel_idx, grid_idx)
                    flat_idx += 1
        
        self.sampled_size = flat_idx
    
    def __len__(self):
        return self.sampled_size
    
    def __getitem__(self, idx):
        d_idx, k_rel_idx, grid_idx = self.pair_mapping[idx]
        
        # Decode grid index to (t_idx, r_idx)
        Nr = len(self.r)
        t_idx = grid_idx // Nr
        r_idx = grid_idx % Nr
        
        d_val_m = self.d_val_m[d_idx]
        k_rel = self.k_rel_log[k_rel_idx]
        r = self.r[r_idx]
        t_out = self.t_out[t_idx]
        C_I_log = self.C_I_log[d_idx, k_rel_idx, t_idx, r_idx]

        # Normalize
        d_val_m_norm = normalize(d_val_m, self.norm["d"][0], self.norm["d"][1])
        k_rel_norm = normalize(k_rel, self.norm["k_rel"][0], self.norm["k_rel"][1])
        t_out_norm = normalize(t_out, self.norm["t"][0], self.norm["t"][1])
        r_norm = normalize(r, self.norm["r"][0], self.norm["r"][1])
        C_I_log_norm = normalize(C_I_log, self.norm["CI"][0], self.norm["CI"][1])

        X = t.stack([
            t.tensor(d_val_m_norm, dtype=t.float32), 
            t.tensor(k_rel_norm, dtype=t.float32),
            t.tensor(t_out_norm, dtype=t.float32),
            t.tensor(r_norm, dtype=t.float32)
        ], dim=0)
        
        y = t.tensor(C_I_log_norm, dtype=t.float32).unsqueeze(0)

        return X, y


def prepare_torch_datasets(params: MLParameters):
    """
    Perform train/val/test split and ready datasets
    """
    data = np.load(params.data_filepath)

    train_d_idx = np.where(np.isin(data["d_val_m"], params.train_d))
    train_k_rel_idx = np.where(np.isin(data["k_rel"], params.train_k_rel)) 
    train_idx_cart = np.stack(np.meshgrid(train_d_idx, train_k_rel_idx, indexing='ij'), axis=-1)
    
    train_data = {
        "r": data["r"],
        "t_out": data["t_out"],
        "d_val_m": params.train_d,
        "k_rel": params.train_k_rel,
        "CN": data["CN"][train_idx_cart[:,:,0], train_idx_cart[:,:,1]],
        "CF": data["CF"][train_idx_cart[:,:,0], train_idx_cart[:,:,1]],
        "CIN": data["CIN"][train_idx_cart[:,:,0], train_idx_cart[:,:,1]],
        "CI": data["CI"][train_idx_cart[:,:,0], train_idx_cart[:,:,1]],
    }

    val_d_idx = np.where(np.isin(data["d_val_m"], params.val_d))
    val_k_rel_idx = np.where(np.isin(data["k_rel"], params.val_k_rel)) 
    val_idx_cart = np.stack(np.meshgrid(val_d_idx, val_k_rel_idx, indexing='ij'), axis=-1)
    
    val_data = {
        "r": data["r"],
        "t_out": data["t_out"],
        "d_val_m": params.val_d,
        "k_rel": params.val_k_rel,
        "CN": data["CN"][val_idx_cart[:,:,0], val_idx_cart[:,:,1]],
        "CF": data["CF"][val_idx_cart[:,:,0], val_idx_cart[:,:,1]],
        "CIN": data["CIN"][val_idx_cart[:,:,0], val_idx_cart[:,:,1]],
        "CI": data["CI"][val_idx_cart[:,:,0], val_idx_cart[:,:,1]],
    }

    test_d_idx = np.where(np.isin(data["d_val_m"], params.test_d))
    test_k_rel_idx = np.where(np.isin(data["k_rel"], params.test_k_rel)) 
    test_idx_cart = np.stack(np.meshgrid(test_d_idx, test_k_rel_idx, indexing='ij'), axis=-1)
    
    test_data = {
        "r": data["r"],
        "t_out": data["t_out"],
        "d_val_m": params.test_d,
        "k_rel": params.test_k_rel,
        "CN": data["CN"][test_idx_cart[:,:,0], test_idx_cart[:,:,1]],
        "CF": data["CF"][test_idx_cart[:,:,0], test_idx_cart[:,:,1]],
        "CIN": data["CIN"][test_idx_cart[:,:,0], test_idx_cart[:,:,1]],
        "CI": data["CI"][test_idx_cart[:,:,0], test_idx_cart[:,:,1]],
    }
    
    train_dataset = SimulationDataset(params, train_data)

    val_dataset = SimulationDataset(params, val_data)
    val_dataset.norm = train_dataset.norm # same normalization, computed without val/test data to avoid leakage

    test_dataset = SimulationDataset(params, test_data)
    test_dataset.norm = train_dataset.norm

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    params = MLParameters(data_filepath="/u/mgalexan/NPSurr/data/sim_cin.npz")
    
    train, val, test = prepare_torch_datasets(params)
    print(len(train), len(val), len(test))
    print(train[len(train) - 1], val[0])




    