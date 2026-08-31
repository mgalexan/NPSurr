# Welcome!

This codebase was used to generate all of the numerical results for the paper "From Tumor Transport Phenotype to Optimal Nanoparticle Design through Mechanistic Modeling and Machine Learning". 

# Working with the code
To track all of the Python packages used, I used the package manager `uv`. You can install it following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/). If you find yourself lost, start by trying the following Bash command:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
After `uv` is installed, you can build all of the requirements with the shell command `uv sync`. After that, run Python files with `uv run {filename}.py`. The sync command will also create a `venv` for you to use if you have a preference. To run Jupyter notebooks with all of the packages installed, you 
# Replicating the results
If you're interested in just replicating the results we present in the paper, just follow these steps:

1. Go to `/src/constants.py` and change `ROOT` to the absolute path of this repository on your system. 

2. Navigate your terminal to the root directory of this project and run:
```
uv run basic_model.py
```
```
uv run synthetic_tumors.py
```
```
uv run ablation.py
```
This will generate training data for and train the surrogate models necessary to replicate our results. 

**Fair Warning:** These files are implemented with multithreading to make the job faster, however keep in mind that they were written with access to a HPC cluster in mind. By my count these two commands will attempt to train roughly 1600 different neural networks in parallel, so this will take forever/eat your CPU based on your hardware. It will also save a *lot* of information.

3. Run every cell in the following notebooks: `numerical_figs.ipynb`, `profile_inversion_figs.ipynb` and `surrogate_figs.ipynb`. 

This will generate the figures, and the rerun statstical tests we present, so your results might vary slightly from those reported in the paper. The same warning as above applies, as now you will be *processing* the results of 1600 neural networks. 

4. You're done!

# Rough File Layout

In the case you want to more with the code (e.g. change the model), here is a rough layout of the files to get you started.

`./content`: Contains all the figures from the paper generated in our codebase.

`./data`: The default storage location for the various datasets and neural networks made by the code.

`./notebooks`: Contains the notebooks used to make the figures and obtain statistical results for the project.

`./src`: Contains the source code for the machinery used in the experiments, built as a module for use in the rest of the codebase. Most of the files here have a short script written at the bottom of them to test the main functions/classes of the file.

`./src/constants.py`: Contains the default values for all of the experimental parameters. Change these classes to make tweaks to the model used. Note that the data classes containing the defaults are often called with different arguments in the code to make differences in parameters.

`./src/simulation/simulator.py`: Contains the main FDM simulation and constitutive law logic for the paper. Change the functions here to change the model used throughout the project.

`./src/simulation/make_dataset.py`: Contains a macro used to make datasets from the simulation over grids of $(d, k_{rel})$, expecially for training the neural networks.

`./src/ml/data_torch.py`: Contains the data loading logic for training models with PyTorch.

`./src/ml/profile_inversion.py`: Contains the class used to generate point estimates of tissue parameters from observations, as well as the function used to aggregate those point estimates into constitutive laws.

`./src/ml/surrogate.py`: Contains the definition of the surrogate class and the training loop used to train models (change these if you want to use a different surrogate architecture!), as well as the SurrogateInversion class used to run optimization tasks using surrogate models. 

`./main.py`: Contains a demo of the end-to-end `Observation -> FDM Simulation -> Surrogate -> Optimal` Design pipeline. 

`./basic_model.py`: Generates the default surrogate model

`./ablation.py`: Generates the neural networks used in the ablation study

`./synthetic_tumors.py`: Generates the neural networks used to study the effect of tissue and uptake differences in the optimal design.

