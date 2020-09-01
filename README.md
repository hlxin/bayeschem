# Bayeschem (under development)

## This repository contains the Python module required to find parameters for the d-band theory of chemisorption using Bayesian optimization.


The folder of each model contains two important python scripts and multiple txt files. The txt files are how the surface DOS, adsorbate DOS and DFT adsorption energies are stored. These txt files are required to run the script "model.py", which is how the Newns-Anderson parameters are optimized. This script also shows all the parameters, priors and equations used for each model. Running the model.py with the required libraries will run the MCMC sampling to generate the parameters in a pickle file called "M.pickle". The script "Output.py" can then be run to get the parameters by averaging from the samples generated via MCMC, this script will also generate the model predicted adsorption energies in the text file "E_NA.txt".

In order to speed up the MCMC sampling a cython function was compiled ("chemisorption.so") to calculate the hybridization energy contribution. You will need to recompile it using the setup.py. A python version of this function has also been included in each folder ("chemisorption_py.py") which will return the same hybridization energy but is slower and thus is mainly used if large sampling is not required. 

**Steps to use this approach**
* Compile the cython code of the chemisorption function.
* Prepare the metal density of states and molecular (atomic) orbital DOS of adsorbates.
* Modify model.py code to tailored valence adsorbate orbitals included in chemsorption
* Run model.py to generate posterior distribution of model parameters
* Analysis the trajectory for converged parameters and use posterior parameters for prediction