import torch as t
import numpy as np
from tqdm import tqdm

from constants import *
from simulation.make_dataset import make_dataset

from ml.profile_inversion import estimate_laws, ProfileInversion
from ml.surrogate import train_surrogate, SurrogateInversion

def main():
    # First create some noisy observations for 4 different diameters
    d_vals = np.array([20, 40, 60, 80]) * 1e-9
    noisy_dataset_params = DatasetParameters(
        d_val_m=  d_vals,
        tau= np.array([12.0*3600]),
        save= False,
        noise_level= 0.02
    )
    data_noisy = make_dataset(PhysicsConstants(), SimulationParameters(), noisy_dataset_params)

    # Reformat the data to contain only a few timesteps
    data_obs = [{"r": data_noisy["r"], "t_out": data_noisy["t_out"][::20], "CN": data_noisy["CN"][i,0][::20]} for i in range(len(d_vals))] 

    # Estimate values of A and Gamma from these observations
    A_est_list = []
    G_est_list = []
    inverter = ProfileInversion(PhysicsConstants(), SimulationParameters(), InversionParameters())

    for d_val, single_data in tqdm(zip(d_vals, data_obs)):
        inverter.phys.d_val_m = d_val
        res = inverter.invert(single_data)
        A_est_list.append(res["A_est"])
        G_est_list.append(res["G_est"])
    
    # Estimate physical laws from these inversions
    law_result = estimate_laws(d_vals, np.array(A_est_list), np.array(G_est_list), PhysicsConstants(), SimulationParameters())

    fitted_phys = PhysicsConstants(a_D = law_result["a_D_mean"], a_P= law_result["a_P_mean"])

    # Make training dataset
    training_data_params = DatasetParameters(
        filepath= "./example",
        filename= "training_dataset.npz"
    )
    make_dataset(fitted_phys, SimulationParameters(), training_data_params, dim=False)

    # Train surrogate model
    ml_params = MLParameters(
        data_filepath= "./example/training_dataset.npz",
        save_path= "./example/surrogate_model.pt"
    )
    
    train_surrogate(ml_params)
    
    # For our purposes our desired trajectory is 35nm nanoparticles with tau=72000
    desired_data_params = DatasetParameters(
        d_val_m = np.array([35]) * 1e-9,
        tau= np.array([72000]),
        save= False
    )
    data_desired = make_dataset(PhysicsConstants(), SimulationParameters(), desired_data_params)
    data_desired["CI"] = data_desired["CI"][0,0]
    
    # Perform surrogate inversion
    surr_inv = SurrogateInversion(t.load("./example/surrogate_model.pt", weights_only= False), InversionParameters())

    inverted_res = surr_inv.invert(data_desired)

    print(inverted_res.x)



if __name__ == "__main__":
    main()
