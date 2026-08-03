import numpy as np
from tqdm import tqdm
from copy import copy
import os

from simulation.simulator import *
from constants import DatasetParameters, PhysicsConstants, SimulationParameters

def add_noise(data: np.ndarray, noise_level: float):
    """
    Add Gaussian noise to the data.
    """
    rng = np.random.default_rng()
    sigma = np.max(np.abs(data)) * noise_level
    noise = rng.normal(0, sigma, size=data.shape)
    return data + noise

def make_dataset(constants: PhysicsConstants, params: SimulationParameters, dataset_params: DatasetParameters, dim: bool = True):
    '''
    Generate simulations on a grid of parameters and save the results to a .npz file.
    '''
    C = copy(constants)
    r_vals, t_out_vals, CN_vals, CF_vals, CIN_vals, CI_vals = [], [], [], [], [], []
    for d_val_m in tqdm(dataset_params.d_val_m, desc="d_val_m"):
        for k_rel in dataset_params.k_rel:
            C.d_val_m = d_val_m
            C.k_rel = k_rel
            if dim:
                DN = D_N_from_dim(C)
                PN = P_N_from_dim(C)
                alpha = alpha_from_dim(C)
            else:
                DN = D_N_from_dimless(C)
                PN = P_N_from_dimless(C)
                alpha = alpha_from_dim(C)
            r, t_out, CN, CF, CIN, CI = forward_solver(PN, DN, alpha, C, params)
            if dataset_params.noise_level > 0.0:
                CN = add_noise(CN, dataset_params.noise_level)
                CF = add_noise(CF, dataset_params.noise_level)
                CIN = add_noise(CIN, dataset_params.noise_level)
                CI = add_noise(CI, dataset_params.noise_level)
            CN_vals.append(CN)
            CF_vals.append(CF)
            CIN_vals.append(CIN)
            CI_vals.append(CI)
    
    d_vals = dataset_params.d_val_m
    k_rel_vals = dataset_params.k_rel
    r_vals = r
    t_out_vals = t_out

    CN_vals = np.array(CN_vals).reshape(len(d_vals), len(k_rel_vals), len(t_out_vals), len(r_vals))
    CF_vals = np.array(CF_vals).reshape(len(d_vals), len(k_rel_vals), len(t_out_vals), len(r_vals))
    CIN_vals = np.array(CIN_vals).reshape(len(d_vals), len(k_rel_vals), len(t_out_vals), len(r_vals))
    CI_vals = np.array(CI_vals).reshape(len(d_vals), len(k_rel_vals), len(t_out_vals), len(r_vals))

    if dataset_params.save:
        filename = dataset_params.filepath + "/" + dataset_params.filename
        os.makedirs(dataset_params.filepath, exist_ok=True)
        np.savez(filename, r=r_vals, t_out=t_out_vals, CN=CN_vals, CF=CF_vals, CIN=CIN_vals, CI=CI_vals, d_val_m=d_vals, k_rel=k_rel_vals)
    else:      
        return { "r": r_vals, "t_out": t_out_vals, "CN": CN_vals, "CF": CF_vals, "CIN": CIN_vals, "CI": CI_vals, "d_val_m": d_vals, "k_rel": k_rel_vals}

if __name__ == "__main__":
    constants = PhysicsConstants()
    params = SimulationParameters()
    dataset_params = DatasetParameters(filename="sim_cin.npz")
    make_dataset(constants, params, dataset_params)


