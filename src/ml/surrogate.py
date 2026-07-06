import torch as t
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm

from ml.data_torch import *
from constants import MLParameters

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
        if self.P["input_dim"] == 3: # Only estimating r
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
        trn_loss = running / len(test_data)
        sch.step()

        model.eval()
        v_run = 0.0
        with t.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(params.DEVICE), yb.to(params.DEVICE)
                v_run += crit(model(xb), yb).item() * len(xb)
        val_loss = v_run / len(test_data)
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


if __name__ == "__main__":
    params = MLParameters()
    train_surrogate(params)