# -*- coding:Utf-8 -*-
import numpy as np
import likelihood
import simulationParameters_of_the_AMIS_or_MH_algorithm as paraMH
import initialisation_rachis as init_rachis
import plotting_functions
import os
from py2_compat import py2_str, py2_round, save_random_state, load_random_state

def print_parameters(text, parameters_set):
    print(text)
    print("   r_symp = ", parameters_set[0])
    print("   r_allee = ", parameters_set[1])
    print("   beta_0 = ", parameters_set[2])
    print("   beta_1 = ", parameters_set[3])
    print("   D = ", parameters_set[4])
    print("   S_sat = ", parameters_set[5])
    print("   gamma = ", parameters_set[6])
    print("   C_non_recover = ", parameters_set[7])
    print("   C_init = ", parameters_set[8])
    print("   kappa = ", parameters_set[9])

            
def metropolis_hastings(dossier, oldSimu=None):
    N_mh = paraMH.N_MH
    #memory allocation
    Np = paraMH.number_of_parameters
    para_mh = np.zeros((N_mh+1, Np))
    log_vraisemblance = np.zeros(N_mh+1)
    acceptation_rate = np.zeros(N_mh+1)

    if paraMH.para_bloc==False:
        nb_varied_para = 0
        for i in range(Np):
            if paraMH.parameters_ranges[i,0]!=paraMH.parameters_ranges[i,1]:
                nb_varied_para +=1
        index_varied_para = list(range(nb_varied_para))
        nb_varied_para = 0
        for i in range(Np):
            if paraMH.parameters_ranges[i,0]!=paraMH.parameters_ranges[i,1]:
                index_varied_para[nb_varied_para] = i
                nb_varied_para +=1
                
    #initialization
    if oldSimu:
        load_random_state(dossier, key_dtype=np.uint32)
        k = np.size(np.load(dossier + '/log_likelihood.npy'))-1
        para_mh[:k+1] = np.load(dossier + '/sampled_parameters.npy')
        log_vraisemblance[:k+1] = np.load(dossier + '/log_likelihood.npy')
        acceptation_rate[:k+1] = np.load(dossier + '/acceptation_rate.npy')
        infection_1 = np.load(dossier + '/infection_1.npy')
        infection_2 = np.load(dossier + '/infection_2.npy')
        accept = py2_round(acceptation_rate[k]*k) #acceptation/rejection counter
        if paraMH.para_bloc==False:
            i_para = k % nb_varied_para
        print_parameters("Last parameters: ", para_mh[k,:Np])
    else:
        k = 0
        i_para = 0
        accept = 0 #acceptation/rejection counter
        infection_1 = init_rachis.infection_1
        infection_2 = init_rachis.infection_2
        np.save(dossier + '/infection_1.npy',infection_1)
        np.save(dossier + '/infection_2.npy',infection_2)
        para_mh[0,:Np] = paraMH.initial_parameters()
        print_parameters("Initial parameters: ", para_mh[0,:Np])
        #enregistrement graphique pour le paramètre initial
        if paraMH.plot_dynamics_initial_parameter:
            os.mkdir(dossier + '/evolution_with_initial_parameter')
            plotting_functions.evolution_chalarose(dossier +
                                    '/evolution_with_initial_parameter',
                                    para_mh[0,:Np], infection_1,infection_2,
                                        'para_bernoulli')
    
    #Computation of the log-likelihood of the initial parameter
    [log_vrai_oldpara,
    terme0_old] = likelihood.log_likelihood(para_mh[k,:Np], infection_1,
                                              infection_2)
    
    prior_oldpara =  paraMH.prior_distribution(para_mh[k,:Np])
    log_vraisemblance[0] = log_vrai_oldpara*(1-terme0_old)
    
    if terme0_old==1:
        print("WARNING : the likelihood of the initial parameter is zero.")

    while k<N_mh:
        #modification des paramètres:
        new_para = para_mh[k,:Np]*1

        print("")
        if paraMH.para_bloc:
            new_para = paraMH.generer_proposition_dist(para_mh[k,:Np])
        else:
            new_para = paraMH.generer_proposition_dist(para_mh[k,:Np],
                                                     index_varied_para[i_para])
            i_para +=1
            if i_para==nb_varied_para:
                i_para = 0
                    
        print_parameters("New set of parameters : ", new_para)
            
        prior_newpara =  paraMH.prior_distribution(new_para)
        pro_dis1 =  paraMH.proposition_dist(para_mh[k,:Np], new_para)
        pro_dis2 =  paraMH.proposition_dist(new_para,para_mh[k,:Np])
        
        if prior_newpara!=0:
            [log_vrai_newpara,
            terme0_new] = likelihood.log_likelihood(new_para, infection_1,
                                                     infection_2)
        else:
            terme0_new = 1.0
            log_vrai_newpara = log_vrai_oldpara*1.0

        delta = (1-terme0_new)*(((prior_newpara*pro_dis1)
                                 /(prior_oldpara*pro_dis2))*       
                 (np.exp(log_vrai_newpara-log_vrai_oldpara)))

        print('Log-likelihood of the new parameter :', end=' ')
        print(log_vrai_newpara*(1-terme0_new))
        print('Prior of the new parameter :',prior_newpara)
        print('Proposal distribution from the old to the new parameter :', end=' ')
        print(pro_dis1)
        print('Log-likelihood of the old parameter :', end=' ')
        print(log_vrai_oldpara*(1-terme0_old))
        print('Prior of the old parameter :', end=' ')
        print(prior_oldpara)
        print('Proposal distribution from the new to the old parameter :', end=' ')
        print(pro_dis2)
        
        if terme0_old!=1:
            if np.random.random()<delta:
                para_mh[k+1,:Np] = new_para
                log_vraisemblance[k+1] = log_vrai_newpara*(1-terme0_new)
                accept += 1
                log_vrai_oldpara = log_vrai_newpara*1
                prior_oldpara = prior_newpara*1
                terme0_old = terme0_new*1
            else:
                para_mh[k+1] = para_mh[k]
                log_vraisemblance[k+1] = log_vrai_oldpara*(1-terme0_old)
        else: #the log-likelihood of the initial parameter is 0
            k =- 1
            para_mh[k+1] = new_para
            log_vraisemblance[k+1] = log_vrai_newpara*(1-terme0_new)
            log_vrai_oldpara = log_vrai_newpara*1
            prior_oldpara = prior_newpara*1
            terme0_old = terme0_new*1

        acceptation_rate[k+1] = accept*1.0/(k+1)
        print("Acceptation rate of the algorithm:", py2_str(accept*1.0/(k+1)))
        print(py2_str((k+1)*100.0/N_mh), "% of the algorithm is done.")
        
        np.save(dossier + '/sampled_parameters.npy', para_mh[:k+2])
        np.save(dossier + '/log_likelihood.npy', log_vraisemblance[:k+2])
        np.save(dossier + '/acceptation_rate.npy', acceptation_rate[:k+2])
        save_random_state(dossier)
        k +=1

    print("")
    print("Acceptation rate of the algorithm:", py2_str(accept*1.0/N_mh))
    print("Results have been saved in " + dossier)

