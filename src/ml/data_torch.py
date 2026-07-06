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
        return np.log(np.clip(data, 1e-30, None) / scale)
    elif isinstance(data, t.Tensor):
        return t.log(t.clip(data, 1e-30, None) / scale)

def inverse_log_transform(data, scale: float):
    """
    Inverse log-transform the data with a given scale.
    """
    if isinstance(data, np.ndarray):
        return np.exp(data) * scale
    elif isinstance(data, t.Tensor):
        return t.log(t.clip(data, 1e-30, None) / scale)

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

class SimulationDataset(Dataset):
    """
    PyTorch Dataset for the simulation data.
    """
    def __init__(self, params: MLParameters, data: dict):
        self.data = data
        self.r = self.data['r']
        self.t_out = self.data['t_out']
        self.CN = self.data['CN']
        self.CF = self.data['CF']
        self.CI = self.data['CI']
        self.d_val_m = self.data['d_val_m']
        self.tau = self.data['tau']
        self.C_I_log = log_transform(self.CI, params.CI_scale)

        self.params = params

        self.d_mean = np.mean(self.d_val_m); self.d_std = np.std(self.d_val_m) if len(self.d_val_m) > 1 else 1
        self.tau_mean = np.mean(self.tau); self.tau_std = np.std(self.tau) if len(self.tau) > 1 else 1
        self.r_mean = np.mean(self.r); self.r_std = np.std(self.r)
        self.t_out_mean = np.mean(self.t_out); self.t_out_std = np.std(self.t_out)
        self.CI_mean = np.mean(self.C_I_log); self.CI_std = np.std(self.C_I_log)

        self.norm = {
            "d": [self.d_mean, self.d_std],
            "tau" : [self.tau_mean, self.tau_std],
            "t" : [self.t_out_mean, self.t_out_std],
            "r" : [self.r_mean, self.r_std],
            "CI" : [self.CI_mean, self.CI_std]
                     }
    
    def __len__(self):
        return len(self.d_val_m) * len(self.tau) * len(self.r) * len(self.t_out)
    
    def __getitem__(self, idx):
        d_val_m_idx = idx // (len(self.tau) * len(self.r) * len(self.t_out))
        tau_idx = (idx // (len(self.r) * len(self.t_out))) % len(self.tau)
        t_out_idx = (idx // len(self.r)) % len(self.t_out)
        r_idx = idx % len(self.r)

        d_val_m = self.d_val_m[d_val_m_idx]
        tau = self.tau[tau_idx]
        r = self.r[r_idx]
        t_out = self.t_out[t_out_idx]
        C_I_log = self.C_I_log[d_val_m_idx, tau_idx, t_out_idx, r_idx]

        

        d_val_m_norm = normalize(d_val_m, self.d_mean, self.d_std)
        tau_norm = normalize(tau, self.tau_mean, self.tau_std)
        t_out_norm = normalize(t_out, self.t_out_mean, self.t_out_std)
        r_norm = normalize(r, self.r_mean, self.r_std)
        C_I_log_norm = normalize(C_I_log, self.CI_mean, self.CI_std)

        X = t.stack([
            t.tensor(d_val_m_norm, dtype=t.float32), 
            t.tensor(tau_norm, dtype=t.float32),
            t.tensor(r_norm, dtype=t.float32),
            t.tensor(t_out_norm, dtype=t.float32)
            ], dim=0)
        
        y = t.tensor(C_I_log_norm, dtype=t.float32).unsqueeze(0)

        return X, y

def prepare_torch_datasets(params: MLParameters):
    """
    Perform train/val/test split and ready datasets
    """
    data = np.load(params.data_filepath)

    train_d_idx = np.where(np.isin(data["d_val_m"], params.train_d))
    train_tau_idx = np.where(np.isin(data["tau"], params.train_tau)) 
    train_idx_cart = np.stack(np.meshgrid(train_d_idx, train_tau_idx, indexing='ij'), axis=-1)
    
    train_data = {
        "r": data["r"],
        "t_out": data["t_out"],
        "d_val_m": params.train_d,
        "tau": params.train_tau,
        "CN": data["CN"][train_idx_cart[:,:,0], train_idx_cart[:,:,1]],
        "CF": data["CF"][train_idx_cart[:,:,0], train_idx_cart[:,:,1]],
        "CI": data["CI"][train_idx_cart[:,:,0], train_idx_cart[:,:,1]],
    }

    val_d_idx = np.where(np.isin(data["d_val_m"], params.val_d))
    val_tau_idx = np.where(np.isin(data["tau"], params.val_tau)) 
    val_idx_cart = np.stack(np.meshgrid(val_d_idx, val_tau_idx, indexing='ij'), axis=-1)
    
    val_data = {
        "r": data["r"],
        "t_out": data["t_out"],
        "d_val_m": params.val_d,
        "tau": params.val_tau,
        "CN": data["CN"][val_idx_cart[:,:,0], val_idx_cart[:,:,1]],
        "CF": data["CF"][val_idx_cart[:,:,0], val_idx_cart[:,:,1]],
        "CI": data["CI"][val_idx_cart[:,:,0], val_idx_cart[:,:,1]],
    }

    test_d_idx = np.where(np.isin(data["d_val_m"], params.test_d))
    test_tau_idx = np.where(np.isin(data["tau"], params.test_tau)) 
    test_idx_cart = np.stack(np.meshgrid(test_d_idx, test_tau_idx, indexing='ij'), axis=-1)
    
    test_data = {
        "r": data["r"],
        "t_out": data["t_out"],
        "d_val_m": params.test_d,
        "tau": params.test_tau,
        "CN": data["CN"][test_idx_cart[:,:,0], test_idx_cart[:,:,1]],
        "CF": data["CF"][test_idx_cart[:,:,0], test_idx_cart[:,:,1]],
        "CI": data["CI"][test_idx_cart[:,:,0], test_idx_cart[:,:,1]],
    }
    
    train_dataset = SimulationDataset(params, train_data)

    val_dataset = SimulationDataset(params, val_data)
    val_dataset.norm = train_dataset.norm # same normalization, computed without val/test data to avoid leakage

    test_dataset = SimulationDataset(params, test_data)
    test_dataset.norm = train_dataset.norm

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    params = MLParameters()
    
    train, val, test = prepare_torch_datasets(params)
    print(len(train), len(val), len(test))
    print(train[len(train) - 1], val[0])




    