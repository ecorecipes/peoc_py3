# -*- coding:Utf-8 -*-
import numpy as np
import observations
import simulationParameters as para
import preliminary_computations_binomial_law as PCBL
import environmental_data as env_data
import reaction_diffusion as RD


index_temp = para.initialization_year - para.first_year_of_data

def compute_log_likelihood(r_symp, C_non_recover, gamma,puiss, fongi, T,
                             nb_years = para.nb_years, indexT = index_temp):
    """Compute the log-likelihood given a fongi quantity fongi with
       fongi[k,i,j] = fongi by tree in the quadrat (i,j) the year k."""
    
    #initialization
    log_vrais = 0    #log-likelihood
    terme0 = 0       #terme0=1 if likelihood=0
    l = 0            
    k_annee = 0 
    q = np.zeros((para.nb_quadrats_x,para.nb_quadrats_y))
    for annee in observations.observation[k_annee:k_annee+nb_years]:
        if terme0==0:
            old_q = q*1.0
            q_temp = para.para_bernoulli_symptom(fongi[l],T[l+indexT],
                                                      gamma,puiss, r_symp)
            q = q_temp*(1-C_non_recover*old_q)+C_non_recover*old_q

            x_i = np.int_(annee[0,:]) #abscissa indexes
            y_i = np.int_(annee[1,:]) #ordinate indexes
            q_i = q[x_i,y_i]
            
            log_term = PCBL.clb_likeli[k_annee+l]*1.
            if np.size(log_term[q_i==0])!=0:
                terme0 = (PCBL.clb_likeli[k_annee+l][q_i==0,1]!=0).any()
                log_term = PCBL.clb_likeli[k_annee+l][q_i!=0]
                q_i = q_i[q_i!=0]
            if np.size(log_term[q_i==1])!=0:
                terme0 = (log_term[q_i==1,2]!=0).any()
                log_term = log_term[q_i!=1]
                q_i = q_i[q_i!=1]
            log_vrais += np.sum(log_term[:,0] + log_term[:,1]*np.log(q_i)
                                        + log_term[:,2]*np.log(1-q_i))
            l+=1
                    
    return [log_vrais, terme0]

def log_likelihood(p, infection_1, infection_2):
    (r_symp, r_allee, beta_0, beta_1, D, S_sat, gamma, C_non_recover,
     C_init, puiss) = p
    f_temp1 = para.f_temperature(env_data.T[index_temp],gamma,puiss)
    f_temp2 = para.f_temperature(env_data.T[1+index_temp],gamma,puiss)
    f_temp1[f_temp1==0] = 1.0
    f_temp2[f_temp2==0] = 1.0
    
    relative_fongi_1 = C_init*r_symp*infection_1 /f_temp1
    relative_fongi_2 = C_init*r_symp*((infection_2-C_non_recover*infection_1)/
                        ((1-C_non_recover*infection_1*C_init)*f_temp2))
    fongi = RD.reaction_di(r_allee, beta_0, beta_1, D, S_sat,
                           relative_fongi_1, relative_fongi_2)
   
    return compute_log_likelihood(r_symp, C_non_recover, gamma, puiss,
                                          fongi, env_data.T)
    
