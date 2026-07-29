from dataclasses import dataclass
import numpy as np
import torch as t

@dataclass
class PhysicsConstants:
    C_P0: float = 1.0e-3
    tau: float = 12.0*3600
    k_B: float = 1.380649e-23
    T_K: float = 310.0
    eta: float = 1.0e-3
    chi_D: float = 0.5
    xi_ECM: float = 100.0e-9
    m_exp: float = 2.0
    P0: float = 2.0e-8
    chi_P: float = 1.0
    d_pore: float = 100.0e-9
    n_exp: float = 2.0
    k_rel: float = 1.0e-5
    k_up: float = 5.0e-6
    D_F: float = 1.0e-11
    k_int: float = 2.0e-5
    k_clr: float = 1.0e-5
    k_deg: float = 1.0e-6
    d_val_m: float = 40.0e-9
    P_F: float = 1.27e-8
    alpha_0: int = 300
    d_0: float = 20e-9

    # Learned values
    a_D : float = chi_D / (xi_ECM ** m_exp)
    a_P : float = chi_P / (d_pore ** n_exp)

@dataclass
class SimulationParameters:
    R_T: float = 5.0e-3
    t_f: float = 24 * 2 *3600
    Nr: int = 100
    Nt_out: int = 200

@dataclass
class DatasetParameters:
    d_val_m: np.ndarray = np.arange(20, 101, 1) * 1.0e-9
    k_rel: np.ndarray = np.logspace(-6, -3, 19)
    save: bool = True
    filepath: str = "data/"
    filename: str = "simulation_dataset.npz"
    noise_level: float = 0.0

@dataclass
class MLParameters:

    # ML dataset creation
    data_filepath: str = "/u/mgalexan/NPSurr/data/simulation_dataset.npz"
    val_d: np.ndarray = np.arange(20, 101, 20) * 1.0e-9
    val_k_rel: np.ndarray = np.logspace(-6, -3, 4)
    test_d: np.ndarray = np.array([35, 45, 55, 65]) * 1.0e-9
    test_k_rel: np.ndarray = np.array([1e-6])
    train_d: np.ndarray = np.setdiff1d(np.arange(20, 101, 1) * 1.0e-9, np.concatenate((val_d, test_d)))
    train_k_rel: np.ndarray = np.setdiff1d(np.logspace(-6, -3, 19), np.concatenate((val_k_rel, test_k_rel)))
    CI_scale: float = 1e-8
    k_rel_scale: float= 1e-6
    n_unif: int = 200
    n_strat: int = 200

    # ML Surrogate creation
    input_dim: int = 4
    hidden: int = 128
    n_hidden: int = 3

    # Surrogate Training
    LR_INIT: int = 3e-3
    LR_MIN: int = 1e-5
    N_EPOCHS: int = 150
    BATCH_SIZE: int = 2048
    save_path: str = "/u/mgalexan/NPSurr/data/surrogate_model.pt"

    DEVICE: str = "cuda" if t.cuda.is_available() else "cpu"


@dataclass
class InversionParameters:
    num_grid: int = 7
    A_low: float = 1e-2
    A_high: float = 1
    G_low: float = 0.1
    G_high: float = 50

    d_low: float = 20e-9
    d_high: float = 80e-9
    k_rel_low: float = 1e-6
    k_rel_high: float = 1e-3

