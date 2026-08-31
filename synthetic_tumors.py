"""
Generate data for and train models for different tumour types
"""
from constants import *
from simulation.make_dataset import make_dataset
from ml.surrogate import train_surrogate
from p_tqdm import p_map



params = SimulationParameters()
dataset_maker = lambda x, y: make_dataset(x, params, y)

# Simuation/Training Run for the 1200 experiment grid
sigma_d_vals = np.linspace(10e-9, 50e-9, 20)
d_opt_vals = np.linspace(20e-9, 80e-9, 20)

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

#_ = p_map(dataset_maker, phys, dataset, num_cpus=num_cpus)
#hist = p_map(train_surrogate, ml, num_cpus=num_cpus)

# Experiments to examine specific transition regions with a finer grid.
phys2 = []
dataset2 = []
ml2 = []

# Dense ECM normal transfer region
d_opt_vals = np.linspace(30e-9, 50e-9, 100)
sigma_d_vals = np.linspace(15e-9, 30e-9, 100)[::-1]
for d_opt, sigma_d in zip(d_opt_vals, sigma_d_vals):
    d_pore = d_pore_vals[0]
    xi_ECM = xi_ECM_vals[0]
    P0 = P0_vals[0]
    phys2.append(PhysicsConstants(P0= P0, d_pore=d_pore, xi_ECM=xi_ECM, d_opt=d_opt, sigma_d=sigma_d))
    dataset2.append(DatasetParameters(filename= f"d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_dataset.npz"))
    ml2.append(MLParameters(data_filepath= f"/u/mgalexan/NPSurr/data/d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_dataset.npz", save_path=f"/u/mgalexan/NPSurr/data/d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_model.pt"))

# Intermediate ECM normal transfer region
d_opt_vals = np.linspace(35e-9, 55e-9, 100)
sigma_d_vals = np.linspace(30e-9, 45e-9, 100)[::-1]
for d_opt, sigma_d in zip(d_opt_vals, sigma_d_vals):
    d_pore = d_pore_vals[1]
    xi_ECM = xi_ECM_vals[1]
    P0 = P0_vals[1]
    phys2.append(PhysicsConstants(P0= P0, d_pore=d_pore, xi_ECM=xi_ECM, d_opt=d_opt, sigma_d=sigma_d))
    dataset2.append(DatasetParameters(filename= f"d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_dataset.npz"))
    ml2.append(MLParameters(data_filepath= f"/u/mgalexan/NPSurr/data/d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_dataset.npz", save_path=f"/u/mgalexan/NPSurr/data/d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_model.pt"))

# Loose ECM normal transfer region
d_opt_vals = np.linspace(35e-9, 55e-9, 100)
sigma_d_vals = np.linspace(30e-9, 45e-9, 100)[::-1]
for d_opt, sigma_d in zip(d_opt_vals, sigma_d_vals):
    d_pore = d_pore_vals[2]
    xi_ECM = xi_ECM_vals[2]
    P0 = P0_vals[2]
    phys2.append(PhysicsConstants(P0= P0, d_pore=d_pore, xi_ECM=xi_ECM, d_opt=d_opt, sigma_d=sigma_d))
    dataset2.append(DatasetParameters(filename= f"d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_dataset.npz"))
    ml2.append(MLParameters(data_filepath= f"/u/mgalexan/NPSurr/data/d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_dataset.npz", save_path=f"/u/mgalexan/NPSurr/data/d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_model.pt"))

# Dense ECM brach splitting region
d_opt_vals = np.linspace(60e-9, 80e-9, 100)
sigma_d_vals = np.ones(100) * 15e-9
for d_opt, sigma_d in zip(d_opt_vals, sigma_d_vals):
    d_pore = d_pore_vals[0]
    xi_ECM = xi_ECM_vals[0]
    P0 = P0_vals[0]
    phys2.append(PhysicsConstants(P0= P0, d_pore=d_pore, xi_ECM=xi_ECM, d_opt=d_opt, sigma_d=sigma_d))
    dataset2.append(DatasetParameters(filename= f"d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_dataset.npz"))
    ml2.append(MLParameters(data_filepath= f"/u/mgalexan/NPSurr/data/d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_dataset.npz", save_path=f"/u/mgalexan/NPSurr/data/d_opt={d_opt:.4g}_sigma_d={sigma_d:.4g}_d_pore={d_pore:.4g}_model.pt"))

_ = p_map(dataset_maker, phys2, dataset2)
hist = p_map(train_surrogate, ml2)