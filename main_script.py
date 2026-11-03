# -*- coding:Utf-8 -*-
"""Please, read the instructions of the file READ_ME.txt"""

import time
import sys
import numpy as np
import os
import shutil
# PEOC only saves figures to files and never opens a window.  A GUI backend
# (macOS's default 'macosx') is unsafe to fork after initialisation, and the
# worker pools must fork, so always use the non-interactive Agg backend;
# the saved PNG/PDF files are unchanged.
import matplotlib
matplotlib.use('Agg')

plot_graphs_at_the_end_of_sampling_algo = False
np.random.seed(1) #set the random seed (can be changed for independant runs).

print("""This script allows to run algorithms to estimate parameters of a model
for the expension of the chalara or to make graphs associated to run of these
algorithms. 

The possible actions are:
   1 New run of the Metropolis-Hastings (MH) algorithm.
     -> Enter the parameters in both files 'parameters_of_the_model.py' and 
        '1_or_2_parameters_of_the_AMIS_or_MH_algorithm.py', then select 1.

   2 New run of the AMIS algorithm.
     -> Enter the parameters in both files 'parameters_of_the_model.py' and 
        '1_or_2_parameters_of_the_AMIS_or_MH_algorithm.py', then select 2.

   3 Additional iterations of an existing MH or AMIS simulation.
     -> Increase the number of iterations of the algorithm (N_MH or N_AMIS) to 
        the final number of iterations (including the new ones and the previous
        ones) in the file 'simulationParameters_of_the_AMIS_or_MH_algorithm.py'
        of the corresponding simulation, then select 3. 

   4 Plot graphs of an existing MH or AMIS simulation.

   5 Print RMSE for the best parameters of an existing MH or AMIS simulation.

   6 Plot graphs to compare different runs of the Metropolis-Hastings algorithm.
     -> Enter the names of the folder associated to the simulations to plot in 
        the file '6_simulations_MH_to_compare.py', then select 6. 

   7 Plot graphs to compare the posterior distributions of different AMIS runs.
     -> Enter the names of the folder associated to the simulations to plot in 
        the file '7_simulations_AMIS_to_compare.py', then select 7. 

   8 Compute the log-likelihood of sets of parameters on a regular grid.
     -> Enter the parameters in both files 'parameters_of_the_model.py' and
        '8_parameters_for_computation_on_regular_grid', then select 8.

   9 Plot the log-likelihoods computed for sets of parameters on a regular grid.
     -> Enter the names of the folder associated to the simulations to plot in 
        the file '9_graph_log_likelihood_on_regular_grid.py', then select 9.

   q quit

For more informations about the inputs and outputs of actions, please read the
file 'READ_ME.txt'.
""")

arret = False
action2 = ""

while arret==False:
    action = input("Select an action (1, 2, 3, 4, 5, 6, 7, 8, 9 or q): ")
    if action in ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'Q']:
        arret = True

sys.path.append(os.getcwd()+'/source_files')
        
#New simulation: creation of the folder
if action in ["1","2", "8"]:
    if os.path.exists('Simulations')==False: os.mkdir('Simulations')   
    dossier = time.strftime('Simulations/%y%m%d%H%M%S',time.localtime())
    if action=="1": dossier = dossier + '_MH'
    if action=="2": dossier = dossier + '_AMIS'
    if action=="8": dossier = dossier + '_regular_grid'  
    os.mkdir(dossier)
    shutil.copy("parameters_of_the_model.py",
                    dossier + '/simulationParameters.py')
    sys.path.append(os.getcwd()+'/'+dossier)

    
#Action on an existing simulation: choice of the simulation
if action in ["3","4","5"]:
    if os.path.exists('Simulations')==False:
        liste = []
    else:
        liste = os.listdir('Simulations')

    if liste == []:
        print("No simulation have been done.")
    else:
        liste.sort()
        print("List of the simulations:")
        for i in range(len(liste)):
            print(i+1, ' ', liste[i])
        print("q quit")
        numsimu = input("Enter the number of the simulation:")
        if numsimu not in ["q","Q"]:
            simu = liste[int(numsimu)-1]
            print('Selected simulation : ', simu)
        else: action="q"

algo = ""   
#Action on an existing simulation: determining the type of algorithm
if action in ["3", "4", "5"]:
    if os.path.exists('Simulations/' + simu + '/acceptation_rate.npy'):
        algo = "MH"
    if os.path.exists('Simulations/' + simu +
                          '/log_likelihood_times_prior.npy'):
        algo = "AMIS"

        
if action in ["3","4","5"] :
    dossier = 'Simulations/' + simu
    sys.path.append(os.getcwd()+'/'+dossier)

def save_scripts(list_of_script, original_folder, copy_folder):
    if os.path.exists(copy_folder)==False: os.mkdir(copy_folder)
    for f in list_of_script: shutil.copy(original_folder + f, copy_folder)
    
#Save script for new simulations
save_script = False
if action in ["1", "2", "3", "8"] and save_script:
        if os.path.exists(dossier+'/source_files')==False:
            os.mkdir(dossier+'/source_files')
        if action=="1" or (action=="3" and algo=="MH"):
                algo_file = "metropolis_hastings.py"
        if action=="2" or (action=="3" and algo=="AMIS"):
                algo_file = "amis.py"
        if action=="8" : algo_file = "log_likelihood_on_regular_grid.py"
        save_scripts(['main_script.py', 'density.txt',
                      'data_chalara_2008_2023.txt', 'climate_data',
                      'rachis_initialization.txt'],"", dossier+'/source_files')
        save_scripts(['likelihood.py', 'environmental_data.py',
                    'observations.py',
                    'initialisation_rachis.py', 'plotting_functions.py',
                    'preliminary_computations_binomial_law.py', algo_file,
                    'reaction_diffusion.py', 'py2_compat.py'],'source_files/',
                    dossier+'/source_files')

if action in ["1", "2", "3", "8"]:
    print('Path of the simulation: ', dossier)
    import simulationParameters as para
    if para.plot_curves:
        import plotting_functions
        plotting_functions.plot_data(dossier)

def action_end_of_algo(action):
    if plot_graphs_at_the_end_of_sampling_algo:
        return "graph"
    else:
        print("To plot graphs, run 'main_script.py' and select action 4.")
        return action

if action in ["1", "2"]:
    oldSimu=False
    shutil.copy("1_or_2_parameters_of_the_AMIS_or_MH_algorithm.py", dossier +
                        '/simulationParameters_of_the_AMIS_or_MH_algorithm.py')
if action=="3": oldSimu=True
    
if action=="1" or (action=="3" and algo=="MH"):
    import metropolis_hastings as MH
    MH.metropolis_hastings(dossier, oldSimu)
    action = action_end_of_algo(action)
    algo = 'MH'

if action=="2" or (action=="3" and algo=="AMIS"):
    import amis
    amis.amis(dossier, oldSimu)
    action = action_end_of_algo(action)
    algo = 'AMIS'

        
#Graphs associated to MH or AMIS algorithms
if action=="4": action="graph"
        
if action=="graph":
    if save_script:
        save_scripts(['graphs_sampling_algorithms.py', 'plotting_functions.py'],
                     'source_files/',
                    dossier+'/source_files')
    import graphs_sampling_algorithms as graphs
    if algo == "MH":
        graphs.acceptation_rate(dossier)
        graphs.parameters_evolution(dossier, algo)
    graphs.log_vraisemblance(dossier, algo)
    
    if algo=="AMIS":
        graphs.plot_para_proposal_evolution(dossier)
        graphs.plot_proposal_evolution(dossier)
        graphs.algo_convergence(dossier, algo)
        graphs.weights(dossier, algo)
    graphs.dynamics_best_parameter(dossier, algo, plot_2020=False)
    graphs.posterior_distributions(dossier, algo,
                                          convergence_to_posterior=False)

if action=="5":
    if save_script:
        save_scripts(['RMSE.py'],'source_files/', dossier+'/source_files')
    import RMSE
    RMSE.RMSE(dossier, algo)

if action=="6":
    import graph_comparison_of_MH_iterations

if action=="7":
    import graph_comparison_of_AMIS_posterior_distributions

if action=="8":
    shutil.copy("8_parameters_for_computation_on_regular_grid.py", dossier +
                '/parameters_for_computation_on_regular_grid.py')

    import log_likelihood_on_regular_grid
    log_likelihood_on_regular_grid.compute_log_likelihoods(dossier)

if action=="9":
    import plot_log_likelihood_on_regular_grid
    

