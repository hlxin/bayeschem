# bayeschem

## This repository contains the python scripts required to find parameters for the Newns-Anderson model of chemisorption using bayesian optimization.


To understand the adsorption of *O we created two different models which are referred to as the simplfied and multiorbital models. In the simplified model all p orbitals (pz, px and py) are treated as degenerate. Whereas in the multiorbital model the pz orbital was treated separately from the degenerate px and py orbitals. In the case of *OH there is only a single multiorbital (3sigma, 1pi and 4sigma) model.

The folder of each model contains two important python scripts and multiple txt files. The txt files are how the surface DOS, adsorbate DOS and DFT adsorption energies are stored. These txt files are required to run the script "model.py", which is how the Newns-Anderson parameters are optimized. This script also shows all the parameters, priors and equations used for each model. Running the model.py with the required libraries will run the MCMC sampling to generate the parameters in a pickle file called "M.pickle". The script "Output.py" can then be run to get the parameters by averaging from the samples generated via MCMC, this script will also generate the model predicted adsorption energies in the text file "E_NA.txt".

In order to speed up the MCMC sampling a cython function was compiled ("chemisorption.so") to calculate the hybridization energy contribution. A python version of this function has also been included in each folder ("chemisorption_py.py") which will return the same hybridization energy but is slower and thus is mainly used if large sampling is not required. 
