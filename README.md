# PEOC
Parameters Estimation Of Chalara model

Author : Coralie Fritsch

Contact : coralie.fritsch@inria.fr

Version 1.0 - July 8, 2025. 

This software is governed by the CeCILL license

# Software aim

PEOC is Python software which aims to simulate and estimate the parameters of the chalara model of the article 'Mechanistic-statistical model for the expansion of ash dieback' by Coralie Fritsch, Marie Grosdidier, Anne Gégout-Petit and Benoît Marçais. It allows to reproduce the simulations made in this article, which is available at the following url [https://hal.science/hal-04690647](https://hal.science/hal-04690647).

# Python 3 port

This tree is a Python 3 port of PEOC 1.0 (numpy, scipy, matplotlib). With the same numerical libraries it gives the same outputs as the Python 2 original, bit for bit; `verification/` holds the comparison scripts. The original instructions below still apply, using `python3 main_script.py`. The port adds:

- `diffusion_solver` in `parameters_of_the_model.py`: `'exact'` (default; identical outputs) or `'fast'` (sparse-LU Crank–Nicolson, 10–40× faster, log-likelihoods within ~1e-14 relative; valid for D ≤ 1280).
- `save_every` (optional, in the algorithm file): MH saves its arrays and random state every `save_every` iterations instead of every iteration (default 1).
- `run.py`: the menu actions from the command line, for scripted or parallel runs. It follows `main_script.py` step by step and gives the same output files. `python3 run.py -h` lists the options. Examples:

      python3 run.py mh --solver fast --run-dir Simulations/my_chain \
          --config-set N_MH=1000 --config-set "initial_parameters=[1000, 15, 24, 0, 18, 90, 40, 0.5, 0.0085, 0.05]"
      python3 run.py resume my_chain --config-set N_MH=2000     # action 3
      python3 run.py plots my_chain                             # action 4
      python3 run.py compare-mh --config-set "simu=['my_chain', 'other_chain']"   # action 6

  `--set NAME=EXPR` overrides a model parameter and `--config-set NAME=EXPR` a setting of the action's input file; both edit the copies saved in the run folder, never the top-level files.
- `scripts/appendix_a1_chains.sh`: re-runs the ten MH chains of the paper's Appendix A.1 with this (released) model. The shipped `Simulations/MH_initial_condition_*` come from an earlier, unreleased model version and should not be resumed with this code.

# Usage

To run simulations :
- check the Python version of your computer, then, 
- in a terminal, go in the current directory,
- run the script main_script.py in the terminal using the Python 2 call
associated to your computer version
	ex : ````python main_script.py````	or	````python2.7 main_script.py````

The script 'main_script' allows to realize simulations with news parameters (which are to be enter in the global_parameters files) or to realize additional actions for an old set of parameters.

The possible actions are:
- New run of the Metropolis-Hastings algorithm (additional iterations can be added to already
performed run) 
- New run of the AMIS (Adaptive Multiple Importance Sampling) algorithm (additional iterations can be added to already performed run)  
- Realization of graphs of performed simulations (including posterior distributions, dynamics  of model with the set of parameters having the maximal likelihood...)
- Computation of the RMSE for the parameters set of an existing simulation with the highest likelihood
- Plotting graphs comparing the iteration evolution of different Metropolis-Hastings algorithm runs
- Plotting graphs comparing the posterior distributions of different AMIS algorithm runs
- Computation of the likelihood on a regular grid
- Plotting the likelihood computed on a regular grid

Run the script 'main_script' and follow the instruction in the preamble before to select an action. 
In particular, for new runs of the Metropolis-Hastings or the AMIS algorithm, the parameters  have to be enter in the file 'parameters_of_the_model.py' and  '1_or_2_parameters_of_the_AMIS_or_MH_algorithm.py'. 
For graph comparing the iteration evolution of different Metropolis-Hastings algorithm runs,  open the script '6_simulations_MH_to_compare.py' and follow the instruction in the preamble to indicate which runs have to be compared. 
For graphs comparing the posterior distributions of different AMIS algorithm runs, open the script '7_simulations_AMIS_to_compare' and follow the instruction in the preamble to indicate which runs have to be compared.
For new computations of the likelihood on a regular grid, the parameters have to be enter in the  file 'parameters_of_the_model.py' and '8_parameters_for_computation_on_regular_grid'. 
For graphs the likelihood computed on a regular grid, open the script  '9_graph_log_likelihood_on_regular_grid' and follow the instruction in the preamble to indicate which runs have to be plotted.

# Description of the model

The model is described at https://hal.science/hal-04690647.
It is composed of three sub-models:
(1) a reaction-diffusion model describing the expansion of leaf infection by H. fraxineus and the subsequent colonization of leaf debris in the litter (rachis). This model includes effect of the ash density, the impact of the rainfall and an Allee effect.
(2) a stochastic model describing the development of dieback symptoms as a function of leaf spore infection. This model takes into account temperature effects.
(3) a stochastic model describing observation data as a function of symptoms and quadrats visited. 

# Data

Environmental data:
The climatic data (rainfall and temperatures) are given in the file 'climate_data'. Its corresponds to aggregated data of [Safran data from Météo-France](https://meteo.data.gouv.fr).
The density of ashes is given in the file 'density.txt'. These values are derived from 2006 to 2015 data of [IGN – Inventaire forestier national français, Données brutes](https://inventaire-forestier.ign.fr/dataIFN/).

Observation data:
The file 'data_chalara_2008_2023.txt' contains the observation data obtained by the [Département de la Santé des Forêts (DSF)](https://agriculture.gouv.fr/le-departement-de-la-sante-des-forets-role-et-missions).

In both files, the column 'quadrat' contains the extended lambert II coordinates of the center of the quadrat.

# Description of the different actions

## Action 1

Action 1 run a Metropolis-Hastings algorithm for the estimation of the model parameters.

The inputs of action 1 are to be entered in the file parameters_of_the_model.py and in the file 1_or_2_parameters_of_the_AMIS_or_MH_algorithm. The former one contains the model parameter as well as the discretisation parameter of the reaction-diffusion equation. The latter one contains the parameters of the Metropolis-Hastings algorithm (initialization, prior distribution, proposal distribution).

The output of action 1 are 
- sampled_parameters.npy which contains an array of size (N_MH+1,10) where the variable (0,:) is the initial parameter and the variable (i,:) is the (i+1)-th parameter by the algorithm and with N_MH the number of iterations of the algorithm.
- log_likelihood.npy which contains an array of size N_MH +1 where the i-th entry is the log- likelihood of the i-th tested parameter.
- acceptation_rate.npy which contains an array of size N_MH +1 where the i-th entry is the acceptation of the algorithm from iteration 0 to i.

The inputs are saved in the folder in the files simulationParameters.py and  simulationParameters_of_the_AMIS_or_MH_algorithm.py.

##  Action 2

Action 2 run an AMIS algorithm for the estimation of the model parameters.

The inputs of action 2 are to be entered in the file parameters_of_the_model.py and in the file 1_or_2_parameters_of_the_AMIS_or_MH_algorithm. The former one contains the model parameter as well as the discretisation parameter of the reaction-diffusion equation. The latter one contains the parameters of the AMIS algorithm (initialization, prior distribution).

The output of action 2 are 
- sampled_parameters.npy which contains an array of size (N_AMIS,N_L,10) where the variable (i,j,:) is j-th parameter sampled at the i-th iteration of the algorithm and with N_AMIS the number of iterations of the algorithm and N_L the number of parameters sampled at each iteration.
- log_likelihood_times_prior.npy which contains an array of size (N_AMIS,N_L) where the variable (i,j) is the log-lihood * prior of the j-th parameter sampled at the i-th iteration of the algorithm.
- weight.npy which contains the weights of the parameters.
- mean_proposal.npy which contains which contains an array of size (N_AMIS,10) where the variable (i,:) is the mean of the parameter sampled at the i-th iteration of the algorithm.
- var_proposal.npy which contains which contains an array of size (N_AMIS,10, 10) where the variable (i,:,:) is the covariance matrix of the parameter sampled at the i-th iteration of the algorithm.

The inputs are saved in the folder in the files simulationParameters.py and  simulationParameters_of_the_AMIS_or_MH_algorithm.py.

##  Action 3

Action 3 allows to run additional iterations of Metropolis-Hastings or AMIS algorithm run by  action 1 or 2. The final number of iterations of the algorithm (N_MH or N_AMIS depending on the algorithm previously run), including the new ones and the previous ones, has set in in the file 'simulationParameters_of_the_AMIS_or_MH_algorithm.py' of the corresponding simulation. The  algorithm automatically detects the algorithm which was previously run.

The outputs are the same as previously described for action 1 and 2.

##  Action 4

Action 4 plots graphs associated to a Metropolis-Hastings or AMIS algorithm which was previously run.

For a run of the Metropolis-Hastings algorithm, the outputs are
- in the folder algo_iterations: the parameters and the acceptation rate w.r.t iterations as  well as the log-likelihood of the sampled parameters.
- in the folder graph_law: the marginal and 2-dimensional posterior distributions.
- in the folder evolution_best_para: the model dynamics with the parameter with the highest  likelihood among the sampled parameters.

For a run of the AMIS algorithm, the outputs are
- in the folder algo_iterations: the log-likelihood and weights of sampled parameters.
- in the folder graph_law: the marginal and 2-dimensional posterior distributions.
- in the folder evolution_best_para: the model dynamics with the parameter with the highest  likelihood among the sampled parameters.

## Action 5

Action 5 computes and print the RMSE of the model with the parameter with the highest  likelihood among the sampled parameters by a Metropolis-Hastings or AMIS algorithm.

##  Action 6

Action 6 plots the posterior distributions of different Metropolis-Hastings runs (which have to be run by the action 1). This action can be used for example to observe the convergence of parameters for the algorithm started from different initial conditions.

The inputs of action 6 are to be entered in the file 6_simulations_MH_to_compare.
They concern the names of the simulations to plot as well as some graphical parameters.

The output of action 6 are the following graphs which are plotted in a sub-folder of the folder Simulations, named Compare_MH_x.

log_likelihood.png:
The figures show the values of the log-likelihood w.r.t the iterations of the runs of the Metropolis-Hastings algorithms.

p_evolution.png: (with p=beta_0, D, C_init_, S, kappa, gamma,...)
The figures show the values of the parameter p w.r.t the iterations of the different runs of the Metropolis-Hastings algorithms.

beta1_Hmax_beta0_evolution.png
The figures show the values of beta_1*max_{a,i}H_a^i/beta_0 w.r.t the iterations of the  Metropolis-Hastings algorithms, where H_a^i is the rainfall of the year a in quadrat i.

2D_p1_p2_evolution.png : (with p1, p2=beta_0, D, C_init_, S, kappa, gamma,...)
The figures show the values of the parameter p2 w.r.t the parameter p1 for the different runs of  the Metropolis-Hastings algorithms. The figures are plotted in the variable evol2D of the file 6_simulations_MH_to_compare is True.

The inputs are saved in the folder in the file simulations_MH_to_compare.py.

## Action 7

Action 7 plots the posterior distributions of different AMIS runs (which have to be run by the action 2). This action can be used for example to compare the posterior distributions for different climate variables.

The inputs of action 7 are to be entered in the file 7_simulations_AMIS_to_compare. They concern the names of the simulations to plot as well as some graphical parameters.

The output of action 7 are the following graphs which are plotted in a sub-folder of the folder Simulations, named Compare_posterior_x.

law_p.pdf: (with p=beta_0, D, C_init_, S, kappa, gamma,...)
The figures show the posterior distributions of the parameter p for the different runs of the AMIS algorithms.

The inputs are saved in the folder in the file simulations_AMIS_to_compare.py.

## Action 8

Action 8 computes the log-likelihoods of parameters taken on a regular grid. It allows to detect a suitable zone for the initialization of the AMIS algorithm.

The inputs of action 8 are to be entered in the file
8_parameters_for_computation_on_regular_grid.py. They concern the discretization the each variable on the regular grid as well as the number of computation to make in parallel (useful in particular for costly computations made on a cluster).

The output of action 8 are 
- tested_parameters.npy which contains an array of size (N,10) where the (i,:) entry is the i-th  tested parameter.
- log_likelihood.npy which contains an array of size N where the i-th entry is the log- likelihood of the i-th tested parameter.

The inputs are saved in the folder in the file parameters_for_computation_on_regular_grid.py.

## Action 9

Action 9 plots the log-likelihoods of parameters taken on a regular grid, computed by action 8. It is possible to plot the log-likelihoods of several regular grids (computed by several simulations made by action 8). It also plots the argmax of the log-likelihood w.r.t some fixed parameters.

The inputs of action 9 are to be entered in the file 9_graph_log_likelihood_on_regular_grid.py. They concern the names of the simulations to plot as well as some graphical parameters.

The output of action 9 are the following graphs which are plotted in a sub-folder of the folder Simulations, named Likelihoods_on_regular_grids_x.

tested_para.pdf:
The figure shows the values of the parameter (beta_0,D) for which the likelihood has been plotted. The blue cross represent the parameter values for which the likelihood have been  computed for all values of other parameters on the regular grid formed by all simulations. Red crosses represent parameter values for which there are missing values in other parameters.

best_p.pdf: (with p=C_init_, S, kappa, gamma,...)
The figures show, for each pair (beta_0,D), for each other parameter p which is discretized in more than one value, the parameter p which maximizes the likelihood for the pair (beta_0,D) (i.e. the argmax of the log-likelihood).

vrai_p1_p2.pdf: (with p1,p2=beta_0, D, C_init_, S, kappa, gamma,...)
The figures show, for each pair of parameters (p1,p2) the maximal log-likelihood on the sub-regular grid formed by the other parameters.

On the folder p_fixed: (with p=C_init_, S, kappa, gamma,...)
The figures best_p.pdf and vrai_p1_p2.pdf detailed above are plotted for each other parameters than p, for each fixed values of p, for all p which is discretized in more than one value.

The inputs are saved in the folder in the file graph_log_likelihood_on_regular_grid.py.

Note also that the save of the log-likelihood and tested parameters of the different simulations are modified in the respective folders and are saved on files log_likeli_tab.npy and beta_0_values.npy, D_values.npy, beta_1_values.npy...
