# -*- coding:Utf-8 -*-
import numpy as np
import time
import simulationParameters as simuPara
import simulationParameters_of_the_AMIS_or_MH_algorithm as algoPara
import environmental_data as env_data
import observations
import likelihood
import reaction_diffusion as RD

index_temp = simuPara.initialization_year - simuPara.first_year_of_data

def get_para_and_likelihood(folder, algo):
    """return the sampled parameters and the log-likelihood by the algorithm of
       simulation folder"""
    if algo=='MH':
        return [np.load(folder + '/sampled_parameters.npy'),
                    np.load(folder + '/log_likelihood.npy')]
    if algo=='AMIS':
        vrai = np.load(folder + '/log_likelihood_times_prior.npy')
        vrai = vrai.reshape(vrai.shape[0]*vrai.shape[1])
        para = np.load(folder + '/sampled_parameters.npy')
        para = para.reshape((para.shape[0]*para.shape[1],para.shape[2]))
        prior = 1.0
        for i in range(para.shape[1]):
            if (algoPara.parameters_ranges[i,1]
                    !=algoPara.parameters_ranges[i,0]):
                prior *= 1/(algoPara.parameters_ranges[i,1]
                            -algoPara.parameters_ranges[i,0])
        vrai[vrai!=0]-=np.log(prior)
        return[para,vrai]


def return_bernoulli_parameter(r_symp, C_non_recover, gamma,kappa, fongi, T,N):
    '''Return the Bernoulli parameter for a given infected quantity fongi
    with fongi[k,i,j] : fongi by tree of the quadrat (i,j) in the year k with
    k=[para.initialization_year-1,...].'''
    
    bernoulli_parameter = []
    q = np.zeros((simuPara.nb_quadrats_x,simuPara.nb_quadrats_y))
    l = 0
    k_annee = 0
    
    for annee in observations.observation[k_annee:N]:
        old_q = q*1.0
        q_temp = simuPara.para_bernoulli_symptom(fongi[l],T[l+index_temp],
                                                      gamma, kappa, r_symp)
        q = q_temp*(1-C_non_recover*old_q)+C_non_recover*old_q
        x_i = np.int_(annee[0,:])
        y_i = np.int_(annee[1,:])
        q_i = q[x_i,y_i]
        bernoulli_parameter.append(q_i)
        l+=1
                    
    return bernoulli_parameter
    
def computeRMSE(dossier, algo, T, N_annees):
    if algo == 'MH':
        [sampled_para,vraisemblance] =  get_para_and_likelihood(dossier, algo)
        i_max_vrai = np.argmax(vraisemblance[vraisemblance!=0])
        bestpara = sampled_para[vraisemblance!=0][i_max_vrai]

    if algo == "AMIS":
        vrais_priori = np.load(dossier + '/log_likelihood_times_prior.npy')
        vrais_priori = vrais_priori.reshape(vrais_priori.shape[0]
                                                *vrais_priori.shape[1])
        sampled_para = np.load(dossier + '/sampled_parameters.npy')
        sampled_para = sampled_para.reshape((sampled_para.shape[0]
                                *sampled_para.shape[1],sampled_para.shape[2]))
        sampled_para = sampled_para[vrais_priori!=0]
        vrais_priori = vrais_priori[vrais_priori!=0]
        i_max_vrai = np.unravel_index(np.argmax(vrais_priori[vrais_priori!=0]),
                                          vrais_priori.shape)
        bestpara = sampled_para[i_max_vrai]
    infection_1 = np.load(dossier + '/infection_1.npy')
    infection_2 = np.load(dossier + '/infection_2.npy')
    
    (r_symp, r_allee, beta_0, beta_1, D, S_sat, gamma, C_recover, C_init,
         kappa) = bestpara
    print("Parameters set with the best likelihood:")
    print("r_symp = ", bestpara[0])
    print("r_allee = ", bestpara[1])
    print("beta_0 = ", bestpara[2])
    print("beta_1 = ", bestpara[3])
    print("D = ", bestpara[4])
    print("S_sat = ", bestpara[5])
    print("gamma = ", bestpara[6])
    print("C_recover = ", bestpara[7])
    print("C_init = ", bestpara[8])
    print("kappa = ", bestpara[9])
    
    f_temp1 = simuPara.f_temperature(T[index_temp],gamma,kappa)
    f_temp2 = simuPara.f_temperature(T[1+index_temp],gamma,kappa)
    f_temp1[f_temp1==0] = 1.0
    f_temp2[f_temp2==0] = 1.0
    
    relative_fongi_1 = C_init*r_symp*infection_1 /f_temp1
    relative_fongi_2 = C_init*r_symp*((infection_2-C_recover*infection_1)/
                        ((1-C_recover*infection_1*C_init)*f_temp2))
    
    fungi_per_ash = RD.reaction_di(r_allee, beta_0, beta_1, D, S_sat,
                                    relative_fongi_1, relative_fongi_2,
                                       False, False, nb_years=N_annees)
    
    bernoulli_para = return_bernoulli_parameter(r_symp, C_recover, gamma, kappa,
                                                fungi_per_ash, T, N_annees)
    
    print("Log-likelihood from 2008 to", 2007+N_annees, ":", end=' ')
    print(likelihood.compute_log_likelihood(r_symp, C_recover, gamma,kappa,
                                            fungi_per_ash, T, N_annees)[0])

    print("Computation of RMSE values: the printed data are")
    print("year a: number of data - RMSE of the year a - ", end=' ')
    print("RMSE from 2008 to the year a")
    bernoull = np.zeros((N_annees, simuPara.nb_quadrats_x,
                             simuPara.nb_quadrats_y))
    q = np.zeros((simuPara.nb_quadrats_x,simuPara.nb_quadrats_y))    
    for a in range(N_annees):
        old_q = q*1.0
        q_temp = simuPara.para_bernoulli_symptom(fungi_per_ash[a],
                                            T[a+index_temp], gamma,kappa,r_symp)
        q = q_temp*(1-C_recover*old_q)+C_recover*old_q
        bernoull[a] = q*1.0            

    data_RMSE = np.zeros((N_annees, 3))
 
    RMSE = 0
    nb_obs = 0
    for a in range(N_annees):
        y = observations.y[observations.y[:,0]==2008+a]
        RMSE_a = 0
        for i in range(np.size(y[:,0])):
            RMSE_a += (y[i,3]*0.01-bernoull[a][int(y[i,1]), int(y[i,2])])**2
        RMSE += RMSE_a
        nb_obs +=  np.size(y[:,0])
        print("year ", str(2008+a), ":", np.size(y[:,0]), end=' ')
        print(np.sqrt(RMSE_a / np.sum(np.size(y[:,0]))), end=' ')
        print(np.sqrt(RMSE / nb_obs))
        data_RMSE[a] = [np.size(y[:,0]),
                        np.sqrt(RMSE_a / np.sum(np.size(y[:,0]))),
                        np.sqrt(RMSE / nb_obs)]
    print("number of observations:", nb_obs)

def RMSE(dossier, algo):
    computeRMSE(dossier, algo, env_data.T, 2022-2007+1)

    
