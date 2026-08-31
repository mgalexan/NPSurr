from constants import *
from simulation.make_dataset import make_dataset
from ml.surrogate import train_surrogate

constants = PhysicsConstants()
params = SimulationParameters()
dataset_params = DatasetParameters(filename="sim_cin.npz")
make_dataset(constants, params, dataset_params)

params = MLParameters(data_filepath= "/u/mgalexan/NPSurr/data/sim_cin.npz", save_path="/u/mgalexan/NPSurr/data/cin_model.pt")
train, val = train_surrogate(params)
np.savez("/u/mgalexan/NPSurr/data/trn_val_hist.npz", trn=train, val=val)