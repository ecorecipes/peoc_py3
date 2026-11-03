# -*- coding:Utf-8 -*-
"""This script compute the log-likelihood of parameters on a regular grid.
The parameters of the regular grid have to be entered in the file 
'8_parameters_for_computation_on_regular_grid' which contains
- the discretization of each parameter, 
- the number of computations that have to been in parallel
"""
import time
import sys
import numpy as np
import multiprocessing
# force fork (the Python 2 behaviour); see the comment in amis.py
Pool = multiprocessing.get_context('fork').Pool
import simulationParameters as para
import likelihood
import initialisation_rachis as init_rachis
import environmental_data as env_data
import reaction_diffusion as RD
RD.set_dense_cache_size(1)  # workers see tasks grouped by D
from parameters_for_computation_on_regular_grid import *
        
infection_1 = init_rachis.infection_1
infection_2 = init_rachis.infection_2
index_temp = para.initialization_year - para.first_year_of_data

def log_likelihood(p):

    (r_symp, r_allee, beta_0, beta_1, D, S_sat, gamma, C_non_recover,
     C_init, puiss) = p
    # PEOC 1.0 inverted once per D in the parent and pickled M_inv and A
    # (332 MB) to the worker with every task.  Each worker now builds them
    # itself, with the identical computation, and keeps the last D's inverse.
    M_inv, A = RD.diffusion_operators(D)

    f_temp1 = para.f_temperature(env_data.T[index_temp],gamma,puiss)
    f_temp2 = para.f_temperature(env_data.T[1+index_temp],gamma,puiss)
    f_temp1[f_temp1==0] = 1.0
    f_temp2[f_temp2==0] = 1.0
    
    relative_fongi_1 = C_init*r_symp*infection_1 /f_temp1
    relative_fongi_2 = C_init*r_symp*((infection_2-C_non_recover*infection_1)/
                        ((1-C_non_recover*infection_1*C_init)*f_temp2))

    fongi = RD.solve_reaction_di(r_allee, beta_0, beta_1, D, S_sat,
                                relative_fongi_1,relative_fongi_2, M_inv, A)

    [logvraisem, terme0] = likelihood.compute_log_likelihood(r_symp,
                                                    C_non_recover,
                                                    gamma, puiss, fongi,
                                                    env_data.T)
    return logvraisem*(1-terme0)

                
def compute_log_likelihoods(folder):
    N_D = np.size(v_D)
    N_by_D = (np.size(v_r_allee)*np.size(v_beta_0)*np.size(v_beta_1)
            *np.size(v_gamma)*np.size(v_S_sat)*np.size(v_C_init)
                *np.size(v_kappa)*np.size(v_r_symp)*np.size(v_C_recover))
    N = N_D*N_by_D
    print("Number of parameters: ", N_D, "*", N_by_D, "=", N)
    print("Log-likelihoods computed on ", num_workers, " cores")
    
    parametre = np.zeros((N, 10))
    log_vraisemblance = np.zeros(N)
    k = 0
    l = 1
    for D in v_D:
        for beta_0 in v_beta_0:
            for beta_1 in v_beta_1:
                for r_allee in v_r_allee:
                    for gamma in v_gamma:
                        for S_sat in v_S_sat:
                            for C_init in v_C_init:
                                for kappa in v_kappa:
                                    for r_symp in v_r_symp:
                                        for C_recover in v_C_recover:
                                            p = [r_symp, r_allee, beta_0,
                                                 beta_1, D, S_sat, gamma,
                                                 C_recover, C_init, kappa]
                                            parametre[k] = p
                                            k = k+1

    np.save(folder + '/infection_1.npy', init_rachis.infection_1)
    np.save(folder + '/infection_2.npy', init_rachis.infection_2)

    k_D = 0
    k = 0

    pool1 = Pool(processes = num_workers)
    while k_D < N_D:
        while k<N_by_D:
            v_range = np.arange(num_workers)+k+k_D*N_by_D
            v_range = v_range[v_range<N_by_D+k_D*N_by_D]
            
            threads = [pool1.apply_async(log_likelihood,
                                    args=(parametre[i],))
                        for i in v_range]
            output = [t.get() for t in threads]

            j = 0
            for i in v_range:
                log_vraisemblance[i] = output[j]
                j+=1
                
            l = (np.max(v_range)+1)
            np.save(folder + '/tested_parameters.npy', parametre[:l])
            np.save(folder + '/log_likelihood.npy', log_vraisemblance[:l])
            sys.stdout.write("\r{0}/{1}={2}%          ".format(l,N,l*100.0/N))
            sys.stdout.flush()
            k += num_workers
        k_D += 1
        k = 0
    print("")

