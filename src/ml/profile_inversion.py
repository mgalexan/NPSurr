import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from constants import PhysicsConstants, SimulationParameters, InversionParameters
from util import pde_misfit
from simulation.simulator import *

def pde_misfit(data: np.ndarray, t_obs_idx: np.ndarray, r_obs_idx: np.ndarray, D_N: float, P_N: float, params: SimulationParameters, phys: PhysicsConstants):
    """
    Compute the difference between a given dataset and the forward simulation of the PDE
    """
    r, t_out, CN, CF, CI = forward_solver(P_N, D_N, phys, params)

    CN = CN[np.ix_(t_obs_idx, r_obs_idx)]

    if CN.shape != data["CN"].shape:
        raise ValueError(
            f"Shape mismatch after indexing: simulation CN {CN.shape} vs data CN {data['CN'].shape}. "
            "Check that obs['t_out'] and obs['r'] correspond to the same subset of CN."
        )

    diff = CN - data["CN"]
    err = np.sum(diff**2)
    
    return err



def invert_profile(obs: dict, phys: PhysicsConstants, params: SimulationParameters, inv: InversionParameters):
    """
    Obtain an estimate for dimensionless A and Gamma from a noisy observation
    """

    # Ensure simulation profiles line up with observation data
    r_sim = np.linspace(0, params.R_T, params.Nr)
    t_sim = np.linspace(0, params.t_f, params.Nt_out)

    r_obs_idx = np.where(np.isclose(r_sim[:, None], obs["r"][None, :]).any(axis=1))[0]
    t_obs_idx = np.where(np.isclose(t_sim[:, None], obs["t_out"][None, :]).any(axis=1))[0]

    
    def _loss(X):
        DN = (np.exp(X[0]) * params.R_T ** 2) / params.t_f
        PN = (np.exp(X[1]) * DN) / params.R_T
        return pde_misfit(obs, t_obs_idx, r_obs_idx, DN, PN, params, phys)
    
    # Coarse grid initial search

    log_grid_A = np.linspace(np.log(inv.A_low), np.log(inv.A_high), inv.num_grid)
    log_grid_G = np.linspace(np.log(inv.G_low), np.log(inv.G_high), inv.num_grid)

    x_0 = np.log(np.array([(inv.A_low + inv.A_high) / 2, (inv.G_low + inv.G_high) / 2]))
    best_loss = np.inf

    for A in log_grid_A:
        for G in log_grid_G:
            L = _loss(np.array([A, G]))
            if best_loss > L:
                best_loss = L
                x_0 = np.array([A,G])

    bounds = [(np.log(inv.A_low), np.log(inv.A_high)), (np.log(inv.G_low), np.log(inv.G_high))]

    result = minimize(
            _loss,
            x_0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 600, "ftol": 1e-20, "gtol": 1e-12},
        )
    
    A_est = float(np.exp(result.x[0]))
    G_est = float(np.exp(result.x[1]))

    return {
            "A_est": A_est,
            "G_est": G_est,
            "loss": float(result.fun),
            "success": bool(result.success),
            "n_eval": result.nfev + inv.num_grid ** 2,
            "result": result,
        }

def estimate_laws(d_arr, A_est_arr, G_est_arr, phys: PhysicsConstants, params: SimulationParameters):
    """
    Aggregate results from several profile inversions into an estimate for the fitted functional versions of P_N(d) and D_N(d)
    """
    A0 = phys.k_B * phys.T_K * params.t_f / (3.0 * np.pi * phys.eta * params.R_T ** 2)
    G0 = phys.P0 * 3 * np.pi * phys.eta * params.R_T / (phys.k_B * phys.T_K)

    log_A_d = -np.log(d_arr * A_est_arr / A0) # = a_D * d^m
    log_Gamma_d = -np.log(G_est_arr / (G0 * d_arr)) # =  a_P d^n - a_D d^m 

    a_D_est = log_A_d / (d_arr ** phys.m_exp)
    a_P_est = (log_A_d  + log_Gamma_d) / (d_arr ** phys.n_exp)

    a_D_mean = np.mean(a_D_est); a_P_mean = np.mean(a_P_est)
    a_D_std = np.std(a_D_est); a_P_std = np.std(a_P_est)

    return {
        "a_D_est": a_D_est,
        "a_P_est": a_P_est,
        "a_D_mean": a_D_mean,
        "a_P_mean": a_P_mean,
        "a_D_std": a_D_std,
        "a_P_std": a_P_std
    }

if __name__ == "__main__":
    from constants import PhysicsConstants, SimulationParameters, InversionParameters
    import numpy as n
    data = n.load("data/simulation_dataset.npz")
    d_vals = [data["d_val_m"][i] for i in range(0, 41, 10)]
    data_obs = [{"r": data["r"], "t_out": data["t_out"][::100], "CN": data["CN"][i,0][::100]} for i in range(0, 41, 10)]
    phys = PhysicsConstants(d_val_m=2e-8, tau=21600.0)
    params = SimulationParameters()
    A_list = []
    G_list = []
    for d_val, single_data in tqdm(zip(d_vals, data_obs)):
        phys.d_val_m = d_val
        res = invert_profile(single_data, phys, params, InversionParameters())
        D_est = (res["A_est"] * params.R_T ** 2) / params.t_f
        P_est = (res["G_est"] * D_est) / params.R_T
        D_gt = D_N_from_dim(phys)
        P_gt = P_N_from_dim(phys)
        print(D_est, D_gt)
        print(P_est, P_gt)

        A_list.append(res["A_est"])
        G_list.append(res["G_est"])

    print(estimate_laws(np.array(d_vals), np.array(A_list), np.array(G_list), PhysicsConstants(), SimulationParameters()))



    