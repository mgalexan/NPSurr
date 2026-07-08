import torch as t
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from scipy.optimize import minimize
import time
from tqdm import tqdm

from ml.data_torch import *
from constants import MLParameters, InversionParameters

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

class  SurrogateInversion:
    """
    Provide estimates of d and tau for a desired trajectory of CI
    """
    def __init__(self, model: Surrogate, inv: InversionParameters, n_fit: int = 2):
        self.model = model
        self.inv = InversionParameters
        self.n_fit = 2
    
    def _load_data(self, data: dict):
        data_target = data["CI"].flatten()
        t_grid, r_grid = np.meshgrid(data["t_out"], data["r"], indexing="ij")
        fixed_coords = np.stack([t_grid.ravel(), r_grid.ravel()], axis=-1)

        if self.n_fit == 2:
            padded_coords = np.concat([np.ones_like(fixed_coords), fixed_coords], axis = 1)
        elif self.n_fit == 1:
            padded_coords = np.concat([np.ones((len(data_target), 1)), fixed_coords], axis = 1)

        self.data_tensor = t.tensor(data_target, requires_grad=False).float().unsqueeze(1)
        self.coord_tensor = t.tensor(padded_coords).float()
    

    def _loss(self, x: np.ndarray):
        coords = self.coord_tensor.clone().detach()
        if self.n_fit == 2:
            coords[:, 0] *= x[0]
            coords[:, 1] *= x[1]
        elif self.n_fit == 1:
            coords[:, 0] *= x[0]
        
        res = self.model.forward_physical(coords)
        err = f.mse_loss(res, self.data_tensor)
        return np.log(err.item())

    def invert(self, data: dict):

        self._load_data(data)

        if self.n_fit == 2:   
            bounds = [(self.inv.d_low, self.inv.d_high), (self.inv.tau_low, self.inv.tau_high)]
            x_0 = np.array([(self.inv.d_low + self.inv.d_high) / 2, (self.inv.tau_low + self.inv.tau_high) / 2])
        elif self.n_fit == 1:
            bounds = [(self.inv.d_low, self.inv.d_high)]
            x_0 = np.array([(self.inv.d_low + self.inv.d_high) / 2])
        
        res = minimize(self._loss, x_0, bounds= bounds, method='Nelder-Mead')

        return res



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
    print(inverter.invert(single_traj))
