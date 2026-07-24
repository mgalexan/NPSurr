import torch as t
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from scipy.optimize import minimize
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from ml.data_torch import *
from simulation.simulator import *
from constants import MLParameters, InversionParameters, PhysicsConstants, SimulationParameters

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
        x[:, 1] = normalize(x[:,1], n["tau"][0], n["tau"][1])
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
    # Pick the first (d, tau) pair for consistent animation
    sample_d_idx, sample_tau_idx = 0, 0
    sample_d = val_dataset_raw.d_val_m[sample_d_idx]
    sample_tau = val_dataset_raw.tau[sample_tau_idx]
    sample_r = val_dataset_raw.r
    sample_t = val_dataset_raw.t_out
    
    # Find time index closest to 12 hours
    target_t = 12 * 3600.0
    t_idx_12h = np.argmin(np.abs(sample_t - target_t))
    
    # Compute true trajectory using forward_solver
    from constants import PhysicsConstants, SimulationParameters
    phys_const = PhysicsConstants()
    phys_const.d_val_m = sample_d
    phys_const.tau = sample_tau
    sim_params = SimulationParameters()
    sample_P_N = P_N_from_dim(phys_const)
    sample_D_N = D_N_from_dim(phys_const)
    r_true, t_true, CN_true, CF_true, CI_true = forward_solver(
        sample_P_N, sample_D_N, phys_const, sim_params, verbose=False
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

        # Generate animation frame: spatial profile at t=12h
        with t.no_grad():
            t_r_grid = np.meshgrid(sample_t[t_idx_12h], sample_r, indexing="ij")
            coords_frame = np.stack([t_r_grid[0].ravel(), t_r_grid[1].ravel()], axis=-1)
            d_tau_ones = np.ones((len(coords_frame), 2)) * np.array([sample_d, sample_tau])
            full_coords = np.concatenate([d_tau_ones, coords_frame], axis=1)
            full_coords_tensor = t.tensor(full_coords).float().to(params.DEVICE)
            
            # Forward pass through network with physical normalization
            pred_frame = model.forward_physical(full_coords_tensor).cpu().numpy().flatten()
            animation_frames.append(pred_frame)

        if epoch % 25 == 0 or epoch == 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{params.N_EPOCHS}  trn={trn_loss:.4e}  val={val_loss:.4e}  lr={lr_now:.2e}  ({time.time()-t0_tr:.0f}s)")

    model.load_state_dict(best_state)
    print(f"\nBest val MSE = {best_val:.4e}  ({time.time()-t0_tr:.0f}s total)")
    t.save(model, params.save_path)
    
    # Create and save animation
    _save_training_animation(animation_frames, sample_r, true_spatial_profile, sample_d, sample_tau, 
                            sample_t[t_idx_12h], params.save_path.replace('.pt', '_training.mp4'))
    
    return trn_hist, val_hist


def _save_training_animation(frames, r_vals, true_profile, d_val, tau_val, t_val, output_path):
    """
    Save spatial concentration profiles across training epochs as an animation.
    
    Parameters:
    - frames: list of 1D arrays, each containing the spatial profile at a time point
    - r_vals: spatial grid points (meters)
    - true_profile: 1D array of true spatial profile at the fixed time
    - d_val: particle diameter (meters)
    - tau_val: half-life (seconds)
    - t_val: time point (seconds)
    - output_path: path to save the mp4 animation
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Precompute axis limits including both true and predicted
    all_frames = np.array(frames)
    all_values = np.concatenate([all_frames.ravel(), true_profile.ravel()])
    vmin, vmax = all_values.min(), all_values.max()
    vmin -= 0.1 * (vmax - vmin)  # Add 10% padding
    vmax += 0.1 * (vmax - vmin)
    
    def animate(frame_idx):
        ax.clear()
        ax.plot(r_vals * 1e6, true_profile, 'r-', linewidth=2.5, label='True (PDE)', alpha=0.8)
        ax.plot(r_vals * 1e6, frames[frame_idx], 'b-', linewidth=2, label='Surrogate (NN)', alpha=0.8)
        ax.set_xlabel(r'Radius $r$ ($\mu$m)', fontsize=11)
        ax.set_ylabel(r'Concentration $C_I$ (mol/m³)', fontsize=11)
        ax.set_ylim([vmin, vmax])
        ax.legend(loc='best', fontsize=10)
        ax.set_title(f"Epoch {frame_idx+1}: Spatial profile at $t$=12h\n$d$={d_val*1e9:.0f}nm, $\\tau$={tau_val/3600:.1f}h", fontsize=12)
        ax.grid(alpha=0.3)
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50, repeat=True)
    anim.save(output_path, writer='ffmpeg', fps=20, dpi=100)
    plt.close(fig)
    print(f"Animation saved to {output_path}")

class SurrogateInversion:
    """
    Provide estimates of d and tau for a desired trajectory of CI
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
            coords[:, 1] *= x[1]
        elif self.n_fit == 1:
            coords[:, 0] *= x[0]
        with t.no_grad():
            res = self.model.forward_physical(coords)
        return res
    
    def _loss(self, x: np.ndarray):
            err = f.mse_loss(self._pred(x), self.data_tensor)
            return np.log(err.item())

    def _fkc(self, x: np.ndarray): 
        res = t.sum(self._pred(x)[:,0] * self.coord_tensor[:, 3] ** 2)
        integral = res.detach().numpy() * self.dt * self.dr * 4 * np.pi ** 2
        fkc = 1 - np.exp(-0.66 * integral)
        return -np.log(fkc)
    
    def _fkc_lim(self, x: np.ndarray):
        fkc = self._fkc(x)
        pred = self._pred(x).numpy()
        count = np.where(pred > self.CI_thresh, np.ones_like(pred), np.zeros_like(pred)).sum()
        return fkc + count

    def _clearance(self, x: np.ndarray):
        res = t.sum(self._pred(x)[:,0] * self.coord_tensor[:, 3] ** 2)
        integral_I = res.detach().numpy() * self.dt * self.dr * 4 * np.pi ** 2

        tau = x[1]
        integral_P = (self.phys.C_P0 * tau / np.log(2)) * (1 - np.exp(- np.log(2) * self.params.t_f / tau))

        return -np.log(integral_I / integral_P)
        



    def invert(self, data: dict, loss_type= "mse"):

        self._load_data(data)
        if loss_type == "mse":
            loss = self._loss
        elif loss_type == "fkc":
            loss = self._fkc
        elif loss_type == "fkc_lim":
            loss = self._fkc_lim
        elif loss_type == "clearance":
            loss = self._clearance

        if self.n_fit == 2:   
            bounds = [(self.inv.d_low, self.inv.d_high), (self.inv.tau_low, self.inv.tau_high)]
            x_0 = np.array([(self.inv.d_low + self.inv.d_high) / 2, (self.inv.tau_low + self.inv.tau_high) / 2])
        elif self.n_fit == 1:
            bounds = [(self.inv.d_low, self.inv.d_high)]
            x_0 = np.array([(self.inv.d_low + self.inv.d_high) / 2])

        res = minimize(loss, x_0, bounds= bounds, method='Nelder-Mead')
        

        return res

    def loss_surface(self, obs: dict, d_grid, tau_grid, loss_type="mse"):
        self._load_data(obs)
        L_surf = np.empty((len(d_grid), len(tau_grid)))
        for i, d in enumerate(tqdm(d_grid)):
            for j, tau in enumerate(tau_grid):
                if loss_type == "mse":
                    L_surf[i, j] = self._loss(np.array([d, tau]))
                elif loss_type == "fkc":
                    L_surf[i, j] = -self._fkc(np.array([d, tau]))
                elif loss_type == "fkc_lim":
                    L_surf[i, j] = -self._fkc_lim(np.array([d, tau]))
                elif loss_type == "clearance":
                    L_surf[i, j] = -self._clearance(np.array([d, tau]))
        return L_surf
    
    def pde_misfit_surface(self, obs: dict, d_grid, tau_grid, loss_type="mse"):
        self._load_data(obs)
        L_surf = np.empty((len(d_grid), len(tau_grid)))
        for i, d in enumerate(tqdm(d_grid)):
            for j, tau in enumerate(tau_grid):
                self.phys.d_val_m = d; self.phys.tau = tau
                P_N = P_N_from_dim(self.phys); D_N = D_N_from_dimless(self.phys)
                r, _, _, _, CI = forward_solver(P_N, D_N, self.phys, self.params)
                if loss_type == "mse":
                    loss = np.mean((CI - obs["CI"]) ** 2)
                elif loss_type == "fkc": 
                    res = np.sum(CI * r ** 2)
                    integral = res * self.dt * self.dr * 4 * np.pi**2
                    loss = 1 - np.exp(-0.66 * integral)
                elif loss_type == "fkc_lim": 
                    res = np.sum(CI * r ** 2)
                    integral = res * self.dt * self.dr * 4 * np.pi**2
                    fkc = 1 - np.exp(-0.66 * integral)
                    pen = np.where(CI > self.CI_thresh, np.ones_like(CI), np.zeros_like(CI)).sum()
                    loss = fkc / np.exp(pen)
                elif loss_type == "clearance":
                    res = np.sum(CI * r ** 2)
                    integral_I = res * self.dt * self.dr * 4 * np.pi**2
                    integral_P = (self.phys.C_P0 * tau / np.log(2)) * (1 - np.exp(- np.log(2) * self.params.t_f / tau))
                    loss = integral_I / integral_P
                L_surf[i, j] = loss
        return L_surf


if __name__ == "__main__":
    '''
    params = MLParameters(
        input_dim=3, 
        val_tau = np.array([12 * 3600.]),
        test_tau=  np.array([12 * 3600.]),
        train_tau=  np.array([12 * 3600.]),
        data_filepath="/u/mgalexan/NPSurr/data/one_dim_data.npy.npz"
        )
    train_surrogate(params)
    '''
    #params = MLParameters()
    #train_surrogate(params)
    model = t.load("/u/mgalexan/NPSurr/data/surrogate_model.pt", weights_only= False)
    data = np.load("/u/mgalexan/NPSurr/data/simulation_dataset.npz")
    print(data["d_val_m"][15], data["tau"][15])
    single_traj = {
        "r" : data["r"],
        "t_out" : data["t_out"],
        "CI" : data["CI"][15,15]
    }
    inverter = SurrogateInversion(model, InversionParameters(), 2)
    print(inverter.invert(single_traj, True))
