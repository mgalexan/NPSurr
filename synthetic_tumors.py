"""
Generate data for and train three different models for different tumour types
"""
from constants import *
from simulation.make_dataset import make_dataset
from ml.surrogate import train_surrogate
from p_tqdm import p_map

params = SimulationParameters()
dataset_maker = lambda x, y: make_dataset(x, params, y)


sigma_d_vals = [10e-9, 20e-9, 40e-9]
d_opt_vals = [30e-9, 50e-9, 70e-9]

P0_vals = [1e-10, 1e-9, 1e-8]
d_pore_vals = [80e-9, 150e-9, 300e-9]
xi_ECM_vals = [25e-9, 60e-9, 120e-9]


phys = []
dataset = []
ml = []
for d_opt in d_opt_vals:
    for sigma_d in sigma_d_vals:
        for i in range(len(d_pore_vals)):
            d_pore = d_pore_vals[i]
            xi_ECM = xi_ECM_vals[i]
            P0 = P0_vals[i]
            phys.append(PhysicsConstants(P0= P0, d_pore=d_pore, xi_ECM=xi_ECM, d_opt=d_opt, sigma_d=sigma_d))
            dataset.append(DatasetParameters(filename= f"d_opt={d_opt:.2g}_sigma_d={sigma_d:.2g}_d_pore={d_pore:.2g}_dataset.npz"))
            ml.append(MLParameters(data_filepath= f"/u/mgalexan/NPSurr/data/d_opt={d_opt:.2g}_sigma_d={sigma_d:.2g}_d_pore={d_pore:.2g}_dataset.npz", save_path=f"/u/mgalexan/NPSurr/data/d_opt={d_opt:.2g}_sigma_d={sigma_d:.2g}_d_pore={d_pore:.2g}_model.pt"))

_ = p_map(dataset_maker, phys, dataset)
hist = p_map(train_surrogate, ml)