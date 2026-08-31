from constants import *

from simulation.make_dataset import make_dataset
from ml.surrogate import train_surrogate
from p_tqdm import p_map

ablates = ["transport", "loading", "uptake"]

dataset_params = [
    DatasetParameters(filename=f"ablate_{ablate}.npz") for ablate in ablates
]

ml_params = [
    MLParameters(data_filepath=f"/u/mgalexan/NPSurr/data/ablate_{ablate}.npz", save_path=f"/u/mgalexan/NPSurr/data/ablate_{ablate}_model.pt") for ablate in ablates
]

dataset_maker = lambda params, ablate: make_dataset(PhysicsConstants(), SimulationParameters(), params, ablate= ablate)

_ = p_map(dataset_maker, dataset_params, ablates)
hist = p_map(train_surrogate, ml_params)
