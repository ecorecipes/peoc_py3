# -*- coding:Utf-8 -*-
import numpy as np
import matplotlib.pyplot as pl
import shutil
import os
import time
import sys
import matplotlib.pyplot as plt
import plotting_functions
import plot_graphs
import simulationParameters_of_the_AMIS_or_MH_algorithm as simuPara
import observations
import scipy.stats

#plt.style.use('classic')

item_evol_laws = 1
ext = plot_graphs.ext
legen = plot_graphs.legen
labels = plot_graphs.labels

font_size_x = 20
font_size_y = 20
font_size_z = 20
ticksize = 16

label_dev_meas = [r'$\mathcal{D}(\mathcal{P}_{r_S},k-1,k)$',
                      r'$\mathcal{D}(\mathcal{P}_{r},k-1,k)$',
                      r'$\mathcal{D}(\mathcal{P}_{\beta_0},k-1,k)$',
                      r'$\mathcal{D}(\mathcal{P}_{\beta_1},k-1,k)$',
                      r'$\mathcal{D}(\mathcal{P}_{D},k-1,k)$',
                      r'$\mathcal{D}(\mathcal{P}_{S},k-1,k)$',
                      r'$\mathcal{D}(\mathcal{P}_{\gamma},k-1,k)$',
                      r'$\mathcal{D}(\mathcal{P}_{C_{pers}},k-1,k)$',
                      r'$\mathcal{D}(\mathcal{P}_{C_{init}},k-1,k)$',
                      r'$\mathcal{D}(\mathcal{P}_{\kappa},k-1,k)$']

def plot_laws(dossier, para, weight, plot_index, index_fig=""):
    if os.path.exists(dossier)==False:                           
        os.mkdir(dossier)

    print("The posterior distributions are plotted in the folder:")
    print(dossier)
        
    for i in plot_index:
        path = dossier + '/law_1D_' + labels[i] + index_fig + ext
        pl.clf()
        plot_graphs.draw_histo_1D(para[:,i], legen[i], col='b',
                                      weights_x=weight)
        pl.savefig(path)
        
        for j in plot_index:
            if j>i:
                path = dossier+'/law_2D_'+labels[i]+'_'+labels[j]+index_fig+ ext
                plot_graphs.law_2D(path, para[:,i], para[:,j], i, j, weight)
            

def get_amis_data(folder,getpara=False, getvraiprior=False, getvrai=False,
                    getweights=False,  getmeanProp=False, getvarProp=False,
                    reshape=True):
    
    ND_index = list(range(simuPara.number_of_parameters))
    for i in range(simuPara.number_of_parameters):
        if simuPara.parameters_ranges[i,0]==simuPara.parameters_ranges[i,1]:
            ND_index.remove(i)
    if 1 in ND_index:
        for i in range(len(ND_index)):
            if ND_index[i]==1: index1 = i
    if 2 in ND_index:
        for i in range(len(ND_index)):
            if ND_index[i]==2: index2 = i
    if 3 in ND_index:
        for i in range(len(ND_index)):
            if ND_index[i]==3: index3 = i
 
    z = []
    if getpara:
        para = np.load(folder + '/sampled_parameters.npy')
        if reshape:
            para = para.reshape((para.shape[0]*para.shape[1],para.shape[2]))
        z.append(para)
    if getvraiprior or getvrai:
        vrai = np.load(folder + '/log_likelihood_times_prior.npy')
        prior = 1.0
        vrai += np.log(prior)
        if reshape: vrai = vrai.reshape(vrai.shape[0]*vrai.shape[1])
        if getvraiprior : z.append(vrai)
        if getvrai:
            for i in range(simuPara.number_of_parameters):
                if (simuPara.parameters_ranges[i,1]
                        !=simuPara.parameters_ranges[i,0]):
                    prior *= 1/(simuPara.parameters_ranges[i,1]
                                -simuPara.parameters_ranges[i,0])
            vrai[vrai!=0]-=np.log(prior)
            z.append(vrai)
    if getweights:
        weight = np.load(folder + '/weight.npy')
        if reshape: weight = weight.reshape(weight.shape[0]*weight.shape[1])
        z.append(weight)
    if getmeanProp:
          mean = np.load(folder + '/mean_proposal.npy')
          z.append(mean)
    if getvarProp:
          var = np.load(folder + '/var_proposal.npy')
          z.append(var)
    return z

def number_of_amis_iterations(folder):
    return get_amis_data(folder, getvraiprior=True, reshape=False)[0].shape[0]

def get_para_and_likelihood(folder, algo):
    """return the sampled parameters and the log-likelihood by the algorithm
        of the simulation folder"""
    if algo=='MH':
        return [np.load(folder + '/sampled_parameters.npy'),
                    np.load(folder + '/log_likelihood.npy')]
    if algo=='AMIS':
        return get_amis_data(folder, getpara=True, getvrai=True)

        
def posterior_distributions(dossier, algo, convergence_to_posterior=False):
    if algo=='MH':
        sampled_para =  get_para_and_likelihood(dossier, algo)[0]
        weight = None
    
    if algo=='AMIS':
        [sampled_para,weight] = get_amis_data(dossier, getpara=True,
                                                  getweights=True)

    ND_para = list(range(simuPara.number_of_parameters))
    for i in range(simuPara.number_of_parameters):
        if simuPara.parameters_ranges[i,0]==simuPara.parameters_ranges[i,1]:
            ND_para.remove(i)
            
    plot_laws(dossier + '/graph_law', sampled_para, weight,ND_para)

    if convergence_to_posterior:
        if algo=='MH':
            N = np.size(sampled_para[:,0])
            for k in [10000,15000,20000,22000, N]:
                para = sampled_para[:k]
                plot_laws(dossier + '/graph_law/evol_law', para, weight,
                           ND_para, index_fig='_'+str(k))
        if algo=="AMIS":
            import amis
            N_AMIS = number_of_amis_iterations(dossier)
            for k in np.arange(0,N_AMIS,item_evol_laws)+1:
                sampled_para = get_amis_data(dossier, getpara=True,
                                                 reshape=False)[0][:k]
                var_prop = get_amis_data(dossier, getvarProp=True,
                                             reshape=False)[0][:k]
                mean_prop = get_amis_data(dossier, getmeanProp=True,
                                              reshape=False)[0][:k]
                log_vrai_prior = get_amis_data(dossier, getvraiprior=True,
                                                   reshape=False)[0][:k]
                weight = amis.compute_weight(sampled_para[:,:,ND_para],
                                          log_vrai_prior, mean_prop, var_prop)
                
                sampled_para = sampled_para.reshape((sampled_para.shape[0]*
                                sampled_para.shape[1],sampled_para.shape[2]))
                weight = weight.reshape(weight.shape[0]*weight.shape[1])
                plot_laws(dossier + '/graph_law/evol_law', sampled_para, weight,
                              ND_para, index_fig='_iter_'+str(k-1))

def weighted_sample(dossier,indices, k):
    import amis
    '''return [p,w] where p is the sample and w the associated weights of the 
    k-th iteration of the simulation "dossier"'''
    p = get_amis_data(dossier,getpara=True, reshape=False)[0][:k+1][:,:,indices]
    w = amis.compute_weight(p,
            get_amis_data(dossier, getvraiprior=True, reshape=False)[0][:k+1],
            get_amis_data(dossier, getmeanProp=True, reshape=False)[0][:k+1],
            var_prop = get_amis_data(dossier, getvarProp=True,
                                         reshape=False)[0][:k+1])

    p = p.reshape((p.shape[0]*p.shape[1],p.shape[2]))
    w = w.reshape(w.shape[0]*w.shape[1])
    return [p,w]

def histogrammes(path, p,w, classes, indice, index_fig):
    nb_para = np.size(p[0])
    hist1D = [np.zeros(1)]*nb_para
    hist2D = [np.zeros(1)]*(nb_para*(nb_para-1)//2)
    l = 0
    for i in range(nb_para):
        pl.clf()
        hist1D[i] = pl.hist(p[:,i], classes[i], density=True, weights=w,
                            stacked=True, color='b', histtype='stepfilled')[0]
        pl.xlabel(legen[indice[i]], fontsize=16)
        pl.savefig(path + '/law_' + labels[indice[i]] + index_fig +'bis.pdf')
        for j in range(nb_para):
            if j>i:
                pl.clf() 
                hist2D[l] = np.histogram2d(p[:,i],p[:,j], density=True,
                                            bins=(classes[i], classes[j]),
                                            weights = w)[0]
                l+=1
    return [hist1D, hist2D]


def algo_convergence(dossier, algo):
    ND_para = list(range(simuPara.number_of_parameters))
    for i in range(simuPara.number_of_parameters):
        if simuPara.parameters_ranges[i,0]==simuPara.parameters_ranges[i,1]:
            ND_para.remove(i)
    nb_para = np.size(ND_para)
                
    if algo=='MH':
        print("The function algo_convergence is not coded for the MH algorithm.")
    
    if algo=='AMIS':
        if os.path.exists(dossier+'/graph_law')==False:           
            os.mkdir(dossier+'/graph_law')
        if os.path.exists(dossier+'/graph_law/evol_law')==False:           
            os.mkdir(dossier+'/graph_law/evol_law')
        nb_range = 200
        N_AMIS = number_of_amis_iterations(dossier)
        print("Iterations number of the AMIS algorithm: ", N_AMIS)
        
        sampled_para  = get_amis_data(dossier, getpara=True, reshape=False)[0]
        partition = np.zeros((nb_para, nb_range))
        h_part = np.zeros(nb_para)
        for i in range(nb_para):
            min_p_i = np.min(sampled_para[:,:,ND_para[i]])
            max_p_i = np.max(sampled_para[:,:,ND_para[i]])
            h = ( max_p_i - min_p_i)/30 + 1*(min_p_i==max_p_i)
            partition[i] = np.linspace(min_p_i-h,max_p_i+h, nb_range)
            h_part[i] = (max_p_i-min_p_i+2*h)/(nb_range-1)
        dev_meas = np.zeros((N_AMIS, nb_para, nb_para))

        [p_k,w_k] = weighted_sample(dossier,ND_para, 0)
        [hist1D_k, hist2S_k] = histogrammes(dossier + '/graph_law/evol_law',
                                            p_k,w_k, partition, ND_para,
                                            '_iter_' + str(0))
        for k in np.arange(N_AMIS):
            [p_k1,w_k1] = weighted_sample(dossier,ND_para, k+1)
            [hist1D_k1, hist2S_k1] = histogrammes(dossier + '/graph_law/' +
                                                    'evol_law', p_k1,
                                                w_k1, partition, ND_para,
                                                '_iter_' + str(k+1))
            l = 0                                     
            for i in range(nb_para):
                dev_meas[k,i,i] = np.max(np.abs(hist1D_k1[i]-hist1D_k[i]))
                for j in range(nb_para):
                    if j>i:
                        dev_meas[k,i,j] = np.max(np.abs(hist2S_k1[l]-
                                                        hist2S_k[l]))
                        l += 1
            hist1D_k = hist1D_k1
            hist2S_k = hist2S_k1
        for i in range(nb_para):
            pl.clf()
            pl.plot(np.arange(N_AMIS)+1, dev_meas[:,i,i]*h_part[i])
            pl.xlabel(r'iteration $k$', fontsize=20)
            pl.ylabel(label_dev_meas[ND_para[i]], fontsize=20)
            pl.tick_params(axis='both', labelsize=16)
            pl.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            pl.savefig(dossier + '/graph_law/evol_law/deviation_measure_1D_'
                        + labels[ND_para[i]] + ext)
            pl.clf()
            pl.plot(np.arange(N_AMIS)+1, dev_meas[:,i,i]*h_part[i])
            pl.xlabel(r'iteration $k$', fontsize=24)
            pl.yscale('log')
            pl.ylabel('deviation measure ' + legen[ND_para[i]], fontsize=16)
            pl.savefig(dossier + '/graph_law/evol_law/deviation_measure_log_1D_'
                        + labels[ND_para[i]] + ext)
            for j in range(nb_para):
                if j>i:
                    pl.clf()
                    pl.plot(np.arange(N_AMIS)+1,
                                dev_meas[:,i,j]*h_part[i]*h_part[j])
                    pl.xlabel('iteration', fontsize=16)
                    pl.ylabel('deviation measure ' + legen[ND_para[i]]
                                  + legen[ND_para[j]], fontsize=16)
                    pl.savefig(dossier+'/graph_law/evol_law/' +
                                'deviation_measure_2D_' +labels[ND_para[i]]
                                   + '_' + labels[ND_para[j]] + ext)
                    pl.clf()
                    pl.plot(np.arange(N_AMIS)+1,
                                dev_meas[:,i,j]*h_part[i]*h_part[j])
                    pl.xlabel('iteration', fontsize=16)
                    pl.ylabel('deviation measure ' + legen[ND_para[i]]
                                  + legen[ND_para[j]], fontsize=16)
                    pl.yscale('log')
                    pl.savefig(dossier+'/graph_law/evol_law/' +
                                'deviation_measure_log_2D_' +labels[ND_para[i]]
                                   + '_' + labels[ND_para[j]] + ext)

def plot_para_proposal_evolution(dossier):
    if os.path.exists(dossier+'/graph_law')==False:              
        os.mkdir(dossier+'/graph_law')
    if os.path.exists(dossier+'/graph_law/evol_proposal')==False:              
        os.mkdir(dossier+'/graph_law/evol_proposal')
        
    ND_para = list(range(simuPara.number_of_parameters))
    for i in range(simuPara.number_of_parameters):
        if simuPara.parameters_ranges[i,0]==simuPara.parameters_ranges[i,1]:
            ND_para.remove(i)

    mean_prop = get_amis_data(dossier, getmeanProp=True)[0]
    var_prop = get_amis_data(dossier, getvarProp=True)[0]
    N_iter = np.size(mean_prop[:,0])
    
    for i in range(len(ND_para)):
        pl.clf()
        pl.plot(range(N_iter), mean_prop[:,i])
        pl.xlabel('iteration', fontsize=16)
        pl.ylabel(legen[ND_para[i]], fontsize=16)
        pl.savefig(dossier+'/graph_law/evol_proposal/evol_mean_'+
                           labels[ND_para[i]] + ext)
        pl.clf()
        pl.plot(range(N_iter), var_prop[:,i,i])
        pl.xlabel('iteration', fontsize=16)
        pl.ylabel('variance ' + legen[ND_para[i]], fontsize=16)
        if i==3:
            pl.yscale('log')
        pl.savefig(dossier+'/graph_law/evol_proposal/law_evol_var_'+
                            labels[ND_para[i]] + ext) 
        for j in range(len(ND_para)):
            if j>i:
                pl.clf()
                pl.plot(range(N_iter), var_prop[:,i,j])
                pl.xlabel('iteration', fontsize=16)
                pl.ylabel('covariance ' + legen[ND_para[i]]
                        + legen[ND_para[j]], fontsize=16)
                pl.savefig(dossier+'/graph_law/evol_proposal/law_evol_cov_'+
                            labels[ND_para[i]]+'_'+labels[ND_para[j]] + ext) 
                
def plot_proposal_evolution(dossier):
    if os.path.exists(dossier+'/graph_law')==False:              
        os.mkdir(dossier+'/graph_law')
    if os.path.exists(dossier+'/graph_law/evol_proposal')==False:              
        os.mkdir(dossier+'/graph_law/evol_proposal')
        
    ND_para = list(range(simuPara.number_of_parameters))
    for i in range(simuPara.number_of_parameters):
        if simuPara.parameters_ranges[i,0]==simuPara.parameters_ranges[i,1]:
            ND_para.remove(i)
            
    N_AMIS = number_of_amis_iterations(dossier)
    for k in np.arange(0,N_AMIS,item_evol_laws):
        mean_prop = get_amis_data(dossier, getmeanProp=True)[0][k]
        var_prop = get_amis_data(dossier, getvarProp=True)[0][k]
        for i in range(len(ND_para)):
            pl.clf()
            x = np.linspace(mean_prop[i]-3*np.sqrt(var_prop[i,i]),
                            mean_prop[i]+3*np.sqrt(var_prop[i,i]), 100)
            pl.plot(x, scipy.stats.multivariate_normal.pdf(x, mean_prop[i],
                                                            var_prop[i,i]))
            pl.xlabel(legen[ND_para[i]], fontsize=16)
            pl.savefig(dossier+'/graph_law/evol_proposal/law_'+
                           labels[ND_para[i]] + '_iter_'+str(k) + ext)
        
            for j in range(len(ND_para)):
                if j>i:
                    pl.clf()
                    y = np.linspace(mean_prop[j]-3*np.sqrt(var_prop[j,j]),
                            mean_prop[j]+3*np.sqrt(var_prop[j,j]), 100)
                    [range_y,range_x] = np.meshgrid(y,x)
                    p = np.zeros((100,100,2))
                    p[:, :, 0] = range_x
                    p[:, :, 1] = range_y
                    z = scipy.stats.multivariate_normal.pdf(p, mean_prop[[i,j]],
                                                var_prop[[i,j]][:,[i,j]])
                    pl.clf()
                    plot_graphs.pcolormesh_flat(range_x, range_y, z)
                    cbar = pl.colorbar()
                    cbar.set_label('density', fontsize = 25)
                    cbar.ax.tick_params(labelsize=15)
                    pl.xlabel(legen[ND_para[i]], fontsize=25)
                    pl.ylabel(legen[ND_para[j]], fontsize=25)
                    pl.savefig(dossier+'/graph_law/evol_proposal/law_'+
                                labels[ND_para[i]]+'_'+labels[ND_para[j]] +
                                   '_iter_'+str(k) + ext)  



    
def plot_evolution_chalarose(dossier, para, infection1, infection2, plot_2020):
    if os.path.exists(dossier)==False: os.mkdir(dossier)
    data = ['nb_frenes', 'para_bernoulli', 'new_infection',
                  'spores_quantity', 'saturated_spores_quantity', 'fct_temp',
                   'chi_a']
    plotting_functions.evolution_chalarose(dossier, para, infection1,
                    infection2, liste_data_to_plot=data, annee_sup=plot_2020)

    
def dynamics_best_parameter(dossier, algo, plot_2020=False):

    if algo == 'MH':
        [sampled_para,vraisemblance] =  get_para_and_likelihood(dossier, algo)
        i_max_vrai = np.argmax(vraisemblance[vraisemblance!=0])
        bestpara = sampled_para[vraisemblance!=0][i_max_vrai]
        print('Log-likelihood of the best parameter :', end=' ')
        print(vraisemblance[i_max_vrai])

    if algo == "AMIS":
        [sampled_para, vrais_priori, weight] = get_amis_data(dossier,
                                    getpara=True, getvrai=True, getweights=True)
        sampled_para = sampled_para[vrais_priori!=0]
        vrais_priori = vrais_priori[vrais_priori!=0]
        
        i_max_vrai = np.unravel_index(np.argmax(vrais_priori[vrais_priori!=0]),
                                          vrais_priori.shape)
        bestpara = sampled_para[i_max_vrai]
        print('log-likelihood of the best parameter :', end=' ')
        print(vrais_priori[i_max_vrai])
    infection_1 = np.load(dossier + '/infection_1.npy')
    infection_2 = np.load(dossier + '/infection_2.npy')

    print("The dynamics for the best parameter are plotted in the folder:")
    print(dossier + '/evolution_best_para')
    print("for the set of parameters:")
    for i in range(10): print(labels[i] + "=", bestpara[i])
        
    plot_evolution_chalarose(dossier + '/evolution_best_para',
                              bestpara[:10], infection_1,
                              infection_2, plot_2020)

def parameters_evolution(dossier, algo):
    if os.path.exists(dossier + '/algo_iterations')==False:                     
        os.mkdir(dossier + '/algo_iterations')
    print("The evolutions of the sampled parameters are plotted in the folder:")
    print(dossier + '/algo_iterations')
    
    [estimated_para,vraisemblance] =  get_para_and_likelihood(dossier, algo)
    for i in [0,1,2,3,4,5,6,7,8,9]:
        pl.clf()
        pl.plot(np.arange(np.size(estimated_para[:,i])),
                estimated_para[:,i], 'x')
        pl.xlabel('iteration')
        pl.ylabel(legen[i], fontsize=16)
        pl.savefig(dossier + '/algo_iterations/' + labels[i] + "_evolution.pdf")

        
def log_vraisemblance(dossier, algo):
    [estimated_para,vraisemblance] =  get_para_and_likelihood(dossier, algo)
    if os.path.exists(dossier + '/algo_iterations')==False:                     
        os.mkdir(dossier + '/algo_iterations')
    for i in [0,1,2,3,4,5,6,7,8,9]:
        pl.clf()
        pl.plot(estimated_para[:,i][vraisemblance!=0],
                    vraisemblance[vraisemblance!=0], 'x')
        pl.xlabel(legen[i], fontsize=16)
        pl.ylabel('log-likelihood', fontsize=14)
        pl.tick_params(axis='y', labelsize=8)
        pl.savefig(dossier + '/algo_iterations/' + labels[i] +
                       "_log_likelihood.pdf")

def weights(dossier, algo):
    if algo=="AMIS":
        [estimated_para, weights] =  get_amis_data(dossier, getpara=True,
                                                            getweights=True)
   
        for i in [0,1,2,3,4,5,6,7,8,9]:
            pl.clf()
            pl.plot(estimated_para[:,i], weights, 'x')
            pl.xlabel(legen[i], fontsize=16)
            pl.ylabel('weights')
            pl.savefig(dossier + "/" + labels[i] + "_weights.pdf")

    
def acceptation_rate(dossier):
    if os.path.exists(dossier + '/algo_iterations')==False:                     
        os.mkdir(dossier + '/algo_iterations')
    accept_rate = np.load(dossier + '/acceptation_rate.npy')
    pl.clf()
    pl.plot(range(np.size(accept_rate)), accept_rate)
    pl.xlabel('iteration')
    pl.ylabel('acceptation rate')
    pl.savefig(dossier + '/algo_iterations/acceptation_rate.pdf')
