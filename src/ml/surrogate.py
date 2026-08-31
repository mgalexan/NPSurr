import torch as t
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from scipy.optimize import minimize, differential_evolution
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from ml.data_torch import *
from simulation.simulator import *
from constants import MLParameters, InversionParameters, PhysicsConstants, SimulationParameters
import time


"""
Surrogate Model for the inverse problem
"""

class Surrogate(nn.Module):
    def __init__(self, params: MLParameters):
        super().__init__()
        self.P = {**vars(params)}
        gain = nn.init.calculate_gain("tanh")
        layers = [nn.Linear(self.P["input_dim"], self.P["hidden"]), nn.Tanh()]
        for _ in range(self.P["n_hidden"]-1):
            layers += [nn.Linear(self.P["hidden"], self.P["hidden"]), nn.Tanh()]
        layers.append(nn.Linear(self.P["hidden"], 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)
        
        self.norm = None
    
    def load_norm(self, data: SimulationDataset):
        """
        Load normalization data
        """
        self.norm = data.norm

    def forward(self, x):
        if self.P["input_dim"] == 3: # Only estimating d
            return self.net(x[:, [0,2,3]])
        else:
            return self.net(x)
    
    def forward_physical(self, x):
        """
        Network with physical values
        """
        n = self.norm
        x[:, 0] = normalize(x[:,0], n["d"][0], n["d"][1])
        x[:, 1] = normalize(log_transform(x[:,1], self.P["k_rel_scale"]), n["k_rel"][0], n["k_rel"][1])
        x[:, 2] = normalize(x[:,2], n["t"][0], n["t"][1])
        x[:, 3] = normalize(x[:,3], n["r"][0], n["r"][1])

        y_norm = self.forward(x)
        y_log = inverse_normalize(y_norm, n["CI"][0], n["CI"][1])
        y = inverse_log_transform(y_log, self.P["CI_scale"])
        return y


def train_surrogate(params: MLParameters):
    """
    train/val/test pipeline for the surrogate model
    """
    model = Surrogate(params)

    train_data, val_data, test_data = prepare_torch_datasets(params)
    train_loader = DataLoader(train_data, params.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, params.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, params.BATCH_SIZE, shuffle=True, num_workers=0)

    model.load_norm(train_data)

    n_params = sum(p.numel() for p in model.parameters())

    opt = t.optim.Adam(model.parameters(), lr=params.LR_INIT, weight_decay=1e-5)
    sch = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params.N_EPOCHS, eta_min=params.LR_MIN)
    crit = nn.MSELoss()

    print(f"Parameters: {n_params:,}")

    # Select a validation sample for animation tracking
    val_dataset_raw = val_data.dataset if hasattr(val_data, 'dataset') else val_data
    # Pick the first (d, k_rel) pair for consistent animation
    sample_d_idx, sample_k_rel_idx = 0, 0
    sample_d = val_dataset_raw.d_val_m[sample_d_idx]
    sample_k_rel = val_dataset_raw.k_rel[sample_k_rel_idx]
    sample_r = val_dataset_raw.r
    sample_t = val_dataset_raw.t_out
    
    # Find time index closest to 12 hours
    target_t = 12 * 3600.0
    t_idx_12h = np.argmin(np.abs(sample_t - target_t))
    
    # Compute true trajectory using forward_solver
    from constants import PhysicsConstants, SimulationParameters
    phys_const = PhysicsConstants()
    phys_const.d_val_m = sample_d
    phys_const.k_rel = sample_k_rel
    sim_params = SimulationParameters()
    sample_P_N = P_N_from_dim(phys_const)
    sample_D_N = D_N_from_dim(phys_const)
    sample_alpha = alpha_from_dim(phys_const)
    r_true, t_true, CN_true, CF_true, CIN_true, CI_true = forward_solver(
        sample_P_N, sample_D_N, sample_alpha, phys_const, sim_params, verbose=False
    )
    true_spatial_profile = CI_true[t_idx_12h, :]
    
    animation_frames = []
    
    best_val, best_state = float("inf"), None
    trn_hist, val_hist = [], []
    t0_tr = time.time()

    for epoch in tqdm(range(1, params.N_EPOCHS+1)):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(params.DEVICE), yb.to(params.DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(xb)
        trn_loss = running / len(train_data)
        sch.step()

        model.eval()
        v_run = 0.0
        with t.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(params.DEVICE), yb.to(params.DEVICE)
                v_run += crit(model(xb), yb).item() * len(xb)
        val_loss = v_run / len(val_data)
        trn_hist.append(trn_loss)
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}


        if epoch % 25 == 0 or epoch == 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{params.N_EPOCHS}  trn={trn_loss:.4e}  val={val_loss:.4e}  lr={lr_now:.2e}  ({time.time()-t0_tr:.0f}s)")

    model.load_state_dict(best_state)
    print(f"\nBest val MSE = {best_val:.4e}  ({time.time()-t0_tr:.0f}s total)")
    t.save(model, params.save_path)
        
    return trn_hist, val_hist


class SurrogateInversion:
    """
    Provide estimates of d and k_rel for a desired trajectory of CI
    """
    def __init__(self, model: Surrogate, inv: InversionParameters, phys: PhysicsConstants, params: SimulationParameters, n_fit: int = 2, CI_thresh: float = 1e-4):
        self.model = model
        self.inv = inv
        self.phys = phys
        self.params = params
        self.n_fit = 2
        self.dr = 0
        self.dt = 0
        self.CI_thresh = CI_thresh
    
    def _load_data(self, data: dict):
        self.obs = data
        data_target = data["CI"].flatten()
        self.dr = data["r"][1] - data["r"][0]
        self.dt = data["t_out"][1] - data["t_out"][0]
        t_grid, r_grid = np.meshgrid(data["t_out"], data["r"], indexing="ij")
        fixed_coords = np.stack([t_grid.ravel(), r_grid.ravel()], axis=-1)

        if self.n_fit == 2:
            padded_coords = np.concat([np.ones_like(fixed_coords), fixed_coords], axis = 1)
        elif self.n_fit == 1:
            padded_coords = np.concat([np.ones((len(data_target), 1)), fixed_coords], axis = 1)

        self.data_tensor = t.tensor(data_target, requires_grad=False).float().unsqueeze(1)
        self.coord_tensor = t.tensor(padded_coords).float()
    
    
    def _pred(self, x: np.ndarray):
        coords = self.coord_tensor.clone().detach()
        if self.n_fit == 2:
            coords[:, 0] *= x[0]
            coords[:, 1] *= np.exp(x[1])
        elif self.n_fit == 1:
            coords[:, 0] *= x[0]
        with t.no_grad():
            res = self.model.forward_physical(coords)
        return res
    
    def _loss(self, x: np.ndarray):
            err = f.mse_loss(self._pred(x), self.data_tensor)
            return np.log(err.item())

    def _loss_sim(self, x: np.ndarray):
        self.phys.d_val_m = x[0]; self.phys.k_rel = np.exp(x[1])
        P_N = P_N_from_dim(self.phys); D_N = D_N_from_dimless(self.phys); alpha = alpha_from_dim(self.phys)
        r, _, _, _, _, CI = forward_solver(P_N, D_N, alpha, self.phys, self.params)
        return np.log(np.mean((CI - self.obs["CI"]) ** 2))

    def _CI(self, x: np.ndarray): 
        res = t.sum(self._pred(x)[:,0] * self.coord_tensor[:, 3] ** 2)
        integral = res.detach().numpy() * self.dt * self.dr * 4 * np.pi
        return -np.log(integral)

    def _CI_sim(self, x: np.ndarray):
        self.phys.d_val_m = x[0]; self.phys.k_rel = np.exp(x[1])
        P_N = P_N_from_dim(self.phys); D_N = D_N_from_dimless(self.phys); alpha = alpha_from_dim(self.phys)
        r, _, _, _, _, CI = forward_solver(P_N, D_N, alpha, self.phys, self.params)
        res = np.sum(CI * (r) ** 2)
        integral = res * self.dt * self.dr * 4 * np.pi
        return -np.log(integral)
    
    def _CI_center(self, x: np.ndarray): 
            pred = self._pred(x)[:,0].detach().numpy()
            rad = self.coord_tensor[:, 3].detach().numpy()
            rad = np.where(rad < self.params.R_T * 0.5, rad, np.zeros_like(rad))   
            integral = np.sum(pred * (rad ** 2)) * self.dt * self.dr * 4 * np.pi
            return -np.log(integral)

    def _uptake(self, x: np.ndarray):
        res = t.sum(self._pred(x)[:,0] * self.coord_tensor[:, 3] ** 2)
        integral_I = res.detach().numpy() * self.dt * self.dr * 4 * np.pi

        tau = self.phys.tau
        integral_P = (self.phys.C_P0 * tau / np.log(2)) * (1 - np.exp(- np.log(2) * self.params.t_f / tau))

        return -np.log(integral_I / integral_P)
        



    def invert(self, data: dict, loss_type= "mse", timing = False):

        self._load_data(data)
        if loss_type == "mse":
            loss = self._loss
        elif loss_type == "mse_sim":
                    loss = self._loss_sim
        elif loss_type == "CI":
            loss = self._CI
        elif loss_type == "CI_center":
            loss = self._CI_center
        elif loss_type == "clearance":
            loss = self._clearance
        elif loss_type == "CI_sim":
            loss = self._CI_sim

        if self.n_fit == 2:   
            bounds = [(self.inv.d_low, self.inv.d_high), (np.log(self.inv.k_rel_low), np.log(self.inv.k_rel_high))]
        elif self.n_fit == 1:
            bounds = [(self.inv.d_low, self.inv.d_high)]

        start_time = time.perf_counter()
        res = differential_evolution(loss, bounds= bounds)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        if timing:
            return execution_time
        else:
            opt_vals = res.x
            opt_vals[1] = np.exp(opt_vals[1])
        
        return opt_vals

    def loss_surface(self, obs: dict, d_grid, k_rel_grid, loss_type="mse"):
        self._load_data(obs)
        L_surf = np.empty((len(d_grid), len(k_rel_grid)))
        for i, d in enumerate(tqdm(d_grid)):
            for j, k in enumerate(k_rel_grid):
                k = np.log(k)
                if loss_type == "mse":
                    L_surf[i, j] = self._loss(np.array([d, k]))
                elif loss_type == "CI":
                    L_surf[i, j] = -self._CI(np.array([d, k]))
                elif loss_type == "CI_center":
                                    L_surf[i, j] = -self._CI_center(np.array([d, k]))
                elif loss_type == "uptake":
                    L_surf[i, j] = -self._uptake(np.array([d, k]))
        return L_surf
    
    def pde_misfit_surface(self, obs: dict, d_grid, k_rel_grid, loss_type="mse"):
        self._load_data(obs)
        L_surf = np.empty((len(d_grid), len(k_rel_grid)))
        for i, d in enumerate(tqdm(d_grid)):
            for j, k in enumerate(k_rel_grid):
                self.phys.d_val_m = d; self.phys.k_rel = k
                P_N = P_N_from_dim(self.phys); D_N = D_N_from_dimless(self.phys); alpha = alpha_from_dim(self.phys)
                r, _, _, _, _, CI = forward_solver(P_N, D_N, alpha, self.phys, self.params)
                if loss_type == "mse":
                    loss = np.mean((CI - obs["CI"]) ** 2)
                elif loss_type == "CI": 
                    res = np.sum(CI * (r) ** 2)
                    integral = res * self.dt * self.dr * 4 * np.pi
                    loss = integral
                elif loss_type == "CI_center":
                    rad = np.where(r < self.params.R_T * 0.5, r, np.zeros_like(r))
                    res = np.sum(CI * (rad ** 2))
                    integral = res * self.dt * self.dr * 4 * np.pi
                    loss = integral
                elif loss_type == "uptake":
                    res = np.sum(CI * r ** 2)
                    integral_I = res * self.dt * self.dr * 4 * np.pi
                    integral_P = (self.phys.C_P0 * self.phys.tau / np.log(2)) * (1 - np.exp(- np.log(2) * self.params.t_f / self.phys.tau))
                    loss = integral_I / integral_P
                L_surf[i, j] = loss
        return L_surf


if __name__ == "__main__":
    params = MLParameters(data_filepath= "/u/mgalexan/NPSurr/data/sim_cin.npz", save_path="/u/mgalexan/NPSurr/data/cin_model.pt")
    train, val = train_surrogate(params)
    np.savez("/u/mgalexan/NPSurr/data/trn_val_hist.npz", trn=train, val=val)
    model = t.load("/u/mgalexan/NPSurr/data/cin_model.pt", weights_only= False)
    data = np.load("/u/mgalexan/NPSurr/data/sim_cin.npz")
    print(data["d_val_m"][15], data["k_rel"][15])
    single_traj = {
        "r" : data["r"],
        "t_out" : data["t_out"],
        "CI" : data["CI"][15,15]
    }
    inverter = SurrogateInversion(model, InversionParameters(), PhysicsConstants(), SimulationParameters())
    print(inverter.invert(single_traj))
