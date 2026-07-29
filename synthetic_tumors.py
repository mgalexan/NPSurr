"""
Generate data for and train three different models for different tumour types
"""
from constants import *
from simulation.make_dataset import make_dataset
from ml.surrogate import train_surrogate
from p_tqdm import p_map

params = SimulationParameters()
dataset_maker = lambda x, y: make_dataset(x, params, y)

P0_vals = [1e-10, 1e-9, 1e-8]

d_pore_vals = [80e-9, 150e-9, 300e-9]
xi_ECM_vals = [25e-9, 60e-9, 120e-9]


phys = []
dataset = []
ml = []
for P0 in P0_vals:
    for d_pore, xi_ECM in zip(d_pore_vals, xi_ECM_vals):
        phys.append(PhysicsConstants(P0= P0, d_pore=d_pore, xi_ECM=xi_ECM))
        dataset.append(DatasetParameters(filename= f"P0={P0:.2g}_d_pore={d_pore:.2g}_dataset.npz"))
        ml.append(MLParameters(data_filepath= f"/u/mgalexan/NPSurr/data/P0={P0:.2g}_d_pore={d_pore:.2g}_dataset.npz", save_path=f"/u/mgalexan/NPSurr/data/P0={P0:.2g}_d_pore={d_pore:.2g}_model.pt"))

_ = p_map(dataset_maker, phys, dataset)
hist = p_map(train_surrogate, ml)