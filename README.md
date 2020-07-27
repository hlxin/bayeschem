# bayeschem

In this repository we have the python scripts required to generate the parameters for the Newns-Anderson model of chemisorption for *O and *OH using bayesian optimization.
For *O two different models were generated in the first model which we call the simplified model all p orbitals (pz, px and py) were treated as degenerate. In the multiorbital model
the pz orbital was treated separately from the degenerate px and py orbitals. For *OH we have a single multiorbital (3sigma, 1pi and 4sigma*).
For each model we have a script called "model.py" which has the equations and the priors. Running the model.py with the required libraries will run the MCMC sampling to generate the parameters in a pickle file called "M.pickle".

The script "Output.py" can then be run to get the parameters by averaging from the samples generated via MCMC. "Output.py" will also generate the model predicted adsorption energies in thetext file "E_NA.txt".

In order to speed up the MCMC sampling a cython function was compiled ("chemisorption.so") to calculate the hybrdization energy contribution. A python version of this function has also been included in each folder ("chemisorption_py.py") which will return the same hybridization energy but is slower and thus is mainly used if large sampling is not required. 
