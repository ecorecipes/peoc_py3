# -*- coding:Utf-8 -*-
"""
Enter in this file the parameters of the different actions you want to run by
the script main_script.py.

The parameters are organized by sections of parameters
- parameters of the model (actions 1, 2, ...)
- parameters for the discretization of the reaction-diffusion equation
- parameters of algorithms for parameters estimation (action 1 or 2)

"""

import numpy as np
import scipy.stats

#for the Metropolis Hastings algorithm (action 1):
N_MH = 60000 #number of iteration of the Metropolis Hastings algorithm 

#for the AMIS algorithm (action 2):
N_AMIS = 0 #number of iteration of the AMIS algorithm 
N_L = 0 #number of parameters sets sampled at each iteration 
num_workers = 1 #number of computation in parallel (number of cores)

#for both algorithms (action 1 or 2):
'''The function initial_parameters below is used for the initial sampled 
parameters : 
- it is the initial parameter for the Metropolis-Hastings algorithm 
- it is the mean of the initial distribution for the AMIS algorithm.'''
def initial_parameters(): #action 1 or 2
    return np.array([1000.0,    #r_symp
                       10.0,    #r_allee
                        0.1,    #beta_0
                        0.1,    #beta_1
                        8.0,    #D
                       40.0,    #S_sat
                       60.0,    #gamma
                        0.2,    #C_non_recover
                      0.005,    #C_init
                        1.2])   #puiss


##################### PRIOR DISTRIBUTION #########################
#ranges of the parameters for the uniform law of the prior distribution
parameters_ranges = np.array([[1000.0,1000.0], #r_symp
                             [0.0, 5000.0], #r_allee:[0,0] si pas d'effet Allee
                             [0.0, 800.0], #beta_0
                             [0.0,5000.0], #beta_1
                             [0.,200.], #D
                             [0.0, 500.0], #S_sat
                             [0.0, 600.0], #gamma
                             [0.0, 1.0], #C_non_recover
                             [0., 200.], #C_init
                             [0.0, 10.0] #puiss
                             ])

number_of_parameters = np.size(parameters_ranges[:,0])

def uniform_density(v,v_min,v_max):
    """probability density function of the uniform law U([v_min,v_max]) 
    evaluated in v"""
    if v_min==v_max:
        return 1*(v==v_min)
    else:
        return scipy.stats.uniform.pdf(v, v_min, v_max-v_min)

def prior_distribution(para):
    z = 1.0
    for i in range(number_of_parameters):
        z *= uniform_density(para[i], parameters_ranges[i,0],
                             parameters_ranges[i,1])
    return z


##################### PROPOSAL DISTRIBUTION #########################
#For the Metropolis Algorithm : set para_bloc = False for a change of
#only one parameter at each iteration
para_bloc = False
plot_dynamics_initial_parameter = False #plot dynamics of the initial parameters

#For the MH algorithm, scale_para_MH is the variance of the proposal
#distribution
scale_para_MH = np.array([100., #r_sym
                          200., #r_allee
                          200., #beta_0
                          200., #beta_1
                          200., #D
                          200, #S_sat
                          200., #gamma
                          200., #C_non_recover
                          200., #C_init
                          200.]) #puiss

#For the AMIS algorithm, scale_para_AMIS is the variance of the gaussian
#distribution for the initial sample parameters set 
scale_para_AMIS = np.array([0.,     #r_sym
                            0.,     #r_allee
                            3.,     #beta_0
                            0.,     #beta_1
                            0.4,    #D
                            1.,     #S_sat
                            0.,     #gamma
                            0.,     #C_non_recover
                            1.0*10**(-8),    #C_init
                            1.5*10**(-5)])   #puiss


#----- Gamma proposal distribution for the Metropolis - Hastings algorithm
def densite_gamma(new_p, old_p, scale0):
    return scipy.stats.gamma.pdf(new_p,scale0,scale=old_p/scale0)

def proposition_dist_gamma(new_para, old_para):
    z = 1.0
    for i in range(number_of_parameters):
        if parameters_ranges[i,0]!=parameters_ranges[i,1]:
            z*= densite_gamma(new_para[i],old_para[i],scale_para_MH[i])
    return z

def proposition_dist(new_para, old_para):
    #this function return the density value of the proposal distribution 
    return proposition_dist_gamma(new_para, old_para)
 
def generer_proposition_dist_gamma(para, index=-1):
    new_p = para*1.
    if index==-1:
        for i in range(number_of_parameters):
            if parameters_ranges[i,0]==parameters_ranges[i,1]:
                new_p[i] = parameters_ranges[i,0]
            else:
                new_p[i] = np.random.gamma(scale_para_MH[i],
                                                para[i]/scale_para_MH[i])
    else:
        new_p[index] = np.random.gamma(scale_para_MH[index],
                                       para[index]/scale_para_MH[index])
    return new_p

def generer_proposition_dist(para, index=-1):
    #this function samples one parameter from the proposal distribution
    return generer_proposition_dist_gamma(para, index)
#----- /Gamma proposal distribution for the Metropolis - Hastings algorithm
