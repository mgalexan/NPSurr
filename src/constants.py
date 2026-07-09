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
    alpha: float = 1.0
    D_F: float = 1.0e-11
    k_int: float = 2.0e-5
    k_clr: float = 1.0e-5
    k_deg: float = 1.0e-6
    d_val_m: float = 40.0e-9

    # Learned values
    a_D : float = chi_D / (xi_ECM ** m_exp)
    a_P : float = chi_P / (d_pore ** n_exp)

@dataclass
class SimulationParameters:
    R_T: float = 5.0e-3
    t_f: float = 48.0*3600
    Nr: int = 100
    Nt_out: int = 200

@dataclass
class DatasetParameters:
    d_val_m: np.ndarray = np.arange(20, 101, 1) * 1.0e-9
    tau: np.ndarray = np.linspace(6.0*3600, 24.0*3600, 19)
    save: bool = True
    filepath: str = "data/"
    filename: str = "simulation_dataset.npz"
    noise_level: float = 0.0

@dataclass
class MLParameters:

    # ML dataset creation
    data_filepath: str = "/u/mgalexan/NPSurr/data/simulation_dataset.npz"
    val_d: np.ndarray = np.arange(20, 101, 20) * 1.0e-9
    val_tau: np.ndarray = np.array([36000, 86400.])
    test_d: np.ndarray = np.array([35, 45, 55, 65]) * 1.0e-9
    test_tau: np.ndarray = np.array([72000.])
    train_d: np.ndarray = np.setdiff1d(np.arange(20, 101, 1) * 1.0e-9, np.concatenate((val_d, test_d)))
    train_tau: np.ndarray = np.setdiff1d(np.linspace(6.0*3600, 24.0*3600, 19), np.concatenate((val_tau, test_tau)))
    CI_scale: float = 1e-8
    n_unif: int = 150
    n_strat: int = 150

    # ML Surrogate creation
    input_dim: int = 4
    hidden: int = 64
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
    tau_low: float = 6*3600.
    tau_high: float = 24*3600.

