#-*- coding:Utf-8 -*-
"""
The main function of this file is the "amis" function below.
The enter parameters of this function are 
    N_amis : number of iteration of the AMIS algorithm
    N_L : number of new parameters generated at each iteration step
    folder : the folder in which the results will be saved
The following results are saved in the folder :
    - the sample parameters in the file sampled_parameters.npy
    - the computed weights of the parameters in the file weight.npy
    - the computed log-likelihood times the prior of the parameters in the file 
      log_likelihood_times_prior.npy
    - the means of the proposal distribution at different iterations in the file
      mean_proposal.npy
    - the variances of the proposal distribution at different iterations in the
      file var_proposal.npy
    - the state of the random seed in the file seed.npy
"""

import numpy as np
import likelihood
import simulationParameters_of_the_AMIS_or_MH_algorithm as paraAMIS
import initialisation_rachis as init_rachis
import scipy.stats
import multiprocessing
import sys
from py2_compat import py2_str, save_random_state, load_random_state

# Python 2 on Unix always forked worker processes. Python 3 defaults to 'spawn'
# on macOS (and 'forkserver' on Linux from 3.14), which would re-execute the
# interactive main_script.py in each worker; force 'fork' to keep the original
# semantics (workers inherit sys.path and the loaded simulation modules).
Pool = multiprocessing.get_context('fork').Pool

# Every AMIS sample has its own D, so the dense-inverse cache (meant for
# component-wise MH) could never hit; keep memory as in PEOC 1.0.
import reaction_diffusion as RD
RD.set_dense_cache_size(0)

num_workers = paraAMIS.num_workers

def compute_log_likelihood_times_prior(p):
    prior_p = paraAMIS.prior_distribution(p)
    if prior_p == 0:
        return 0
    else:
        [loglike, is_zero] = likelihood.log_likelihood(p,
                               init_rachis.infection_1, init_rachis.infection_2)
       
        return (loglike + np.log(prior_p))*(1-is_zero)

def compute_mean_variance(weight,p, Np, ND_index):
    mean = np.zeros(Np)
    var = np.zeros((Np,Np))
    mean = np.sum(weight*p[:,:,ND_index].transpose((2,0,1)), axis=(1,2))
    dif_para_mean = p[:,:,ND_index] - mean

    for i in range(Np):
        var[i,i] = np.sum(weight*dif_para_mean[:,:,i]**2)
        for j in np.arange(Np-i-1)+i+1 :
            temp = np.sum(weight*dif_para_mean[:,:,i]*dif_para_mean[:,:,j])
            var[i,j] = temp*1.
            var[j,i] = temp*1.
    return [mean, var]
    
def log_sum_normal(mu, sigma, param):
    iterations_number = mu.shape[0]
    z = np.zeros((param.shape[0],param.shape[1]))
    for i in range(iterations_number):
        z += scipy.stats.multivariate_normal.pdf(param, mu[i], sigma[i])
    return np.log(z/iterations_number)

def compute_weight(para, log_vrai_prior,mean_prop, var_prop):
    log_weight = (log_vrai_prior - log_sum_normal(mean_prop, var_prop, para))
    weight = np.exp(log_weight-np.max(log_weight[log_vrai_prior!=0]))
    weight[log_vrai_prior==0] = 0
    weight /= np.sum(weight)
    return weight

def print_proposal(mean_proposal, var_proposal):
    print("mean of the proposal: ")
    print(mean_proposal)
    print("covariance matrix of the proposal:")
    print(var_proposal)
    print("")
            
def amis(folder, oldSimu=False):
    N_amis = paraAMIS.N_AMIS
    N_L = paraAMIS.N_L
    
    #memory allocation
    para_amis = np.zeros((N_amis, N_L, paraAMIS.number_of_parameters))
    log_vraisemblance_prior = np.zeros((N_amis, N_L))

    ND_index = list(range(paraAMIS.number_of_parameters))   #non-degenerated index
    for i in range(paraAMIS.number_of_parameters):
        if paraAMIS.parameters_ranges[i,0]==paraAMIS.parameters_ranges[i,1]:
            ND_index.remove(i)
            para_amis[:,:,i] = paraAMIS.parameters_ranges[i,0]

    Np = len(ND_index)
    mean_proposal = np.zeros((N_amis+1, Np))
    var_proposal =  np.zeros((N_amis+1, Np, Np))

    if oldSimu==False:
        np.save(folder + '/infection_1.npy',init_rachis.infection_1)
        np.save(folder + '/infection_2.npy',init_rachis.infection_2)
        #initialization
        mean_proposal[0] = paraAMIS.initial_parameters()[ND_index]
        var_proposal[0] = np.diag(paraAMIS.scale_para_AMIS[ND_index])
        k = 0
    else:
        k = np.size(np.load(folder + '/sampled_parameters.npy')[:,0,0])
        print("number of iterations of the previous simulation:", k)

        para_amis[:k] = np.load(folder + '/sampled_parameters.npy')
        weight = np.load(folder + '/weight.npy')
        log_vraisemblance_prior[:k] = np.load(folder
                                    + '/log_likelihood_times_prior.npy')
        mean_proposal[:k+1] = np.load(folder + '/mean_proposal.npy')
        var_proposal[:k+1] = np.load(folder + '/var_proposal.npy')
        load_random_state(folder)
        
    pool1 = Pool(processes = num_workers)           
    while k<N_amis:
        print("Iteration", k+1, "/", N_amis)
        #modification of the parameters:
        para_amis[k][:,
                 ND_index] = np.random.multivariate_normal(mean_proposal[k],
                                                         var_proposal[k],N_L)
        if N_L<= num_workers:
            threads = [pool1.apply_async(compute_log_likelihood_times_prior,
                                    args=(para_amis[k,i],))
                        for i in range(N_L)]
            output = [t.get() for t in threads]
            log_vraisemblance_prior[k] = output
        else:
            l = 0
            while l<N_L:
                v_range = np.arange(num_workers)+l
                v_range = v_range[v_range<N_L]
                threads = [pool1.apply_async(compute_log_likelihood_times_prior,
                                    args=(para_amis[k,i],))
                    for i in v_range]
                output = [t.get() for t in threads]
                j = 0
                for i in v_range:
                    log_vraisemblance_prior[k,i] = output[j]
                    j+=1
                sys.stdout.write("\r{0}/{1}={2}%         ".format(
                    np.max(v_range)+1,N_L,(np.max(v_range)+1)*100.0/N_L))
                sys.stdout.flush()
                l += num_workers
            print("")

        weight = compute_weight(para_amis[:k+1][:,:,ND_index],
                                 log_vraisemblance_prior[:k+1],
                                 mean_proposal[:k+1], var_proposal[:k+1])
        
        [mean,var] = compute_mean_variance(weight,para_amis[:k+1],
                                                Np, ND_index)
        mean_proposal[k+1] = mean
        var_proposal[k+1] = var
                    
        print_proposal(mean_proposal[k+1], var_proposal[k+1])
        print("determinant of the matrix=",np.linalg.det(var_proposal[k+1]))
        k += 1
        print(py2_str(k*100.0/N_amis), "% of the algorithm is done")
        
        np.save(folder + '/sampled_parameters.npy', para_amis[:k])
        np.save(folder + '/weight.npy', weight)
        np.save(folder + '/log_likelihood_times_prior.npy',
                log_vraisemblance_prior[:k])
        np.save(folder + '/mean_proposal.npy', mean_proposal[:k+1])
        np.save(folder + '/var_proposal.npy', var_proposal[:k+1])
        
        save_random_state(folder)





