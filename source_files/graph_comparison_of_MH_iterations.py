# -*- coding:Utf-8 -*-
"""This script plots the evolution of the parameters of different runs of the
Metropolis-Hastings algorithm. It can be used for example to observe the 
convergence of parameters for the algorithm started from different initial 
conditions.
To plot the evolution of the parameters, enter the names of the simulations to 
compare in the file 6_simulations_MH_to_compare.py, then run main_script.py 
and select action 6.
The graphs will be plotted in the folder 'Simulations/Compare_MH_1' (or
in the folder 'Simulations/Compare_MH_(i+1)' if the folder 
'Simulations/Compare_MH_i' already exists).
"""
import numpy as np
import matplotlib.pyplot as pl
import shutil
import os
import time
import sys
import plot_graphs as pg
import matplotlib.pyplot as plt
#plt.style.use('classic')

i = 1
while os.path.exists('Simulations/Compare_MH_' +str(i)):
    i += 1
dossier = 'Simulations/Compare_MH_' +str(i) +'/'
os.mkdir(dossier)
shutil.copy("6_simulations_MH_to_compare.py", dossier +
                '/simulations_MH_to_compare.py')
sys.path.append(os.getcwd()+'/'+dossier)
from simulations_MH_to_compare import *

print("Graphs are added in the folder ", dossier)

Hmax = 322.2 #maximal rainfall value

label_x = pg.legen_iter
titre_x = pg.labels

    
estimated_para = ['a']*len(simu)
log_vrai = ['a']*len(simu)

for i in range(len(simu)):
    folder = 'Simulations/' + simu[i]
    estimated_para[i] = np.load(folder +'/sampled_parameters.npy')
    log_vrai[i] = np.load(folder + '/log_likelihood.npy')

#graph of the log-likelihood evolution
pl.clf()
for j in range(len(simu)):
    pl.plot(np.arange(np.size(log_vrai[j]))[::1],
                log_vrai[j][::1], c[j])
#pl.xlim([0,55000])
pl.xlabel(r'iteration $k$', fontsize=fontsizelabelx)
pl.ylabel(r'log-likelihood $\mathcal{L}(\theta^k)$', fontsize=fontsizelabelx)
pl.tick_params(axis='x', labelsize=fontsizetickx)
pl.tick_params(axis='y', labelsize=fontsizeticky)
pl.savefig(dossier + "log_likelihood" + format_)

#graphs of the evolution of the parameters
for i in range(10):
    pl.clf()
    for j in range(len(simu)):
        pl.plot(np.arange(np.size(estimated_para[j][:,i]))[::k[i]],
                estimated_para[j][::k[i],i], 'x', color=c[j])
    pl.xlabel(r'iteration $k$', fontsize=fontsizelabelx)
    pl.ylabel(label_x[i], fontsize=fontsizelabely)
    pl.tick_params(axis='x', labelsize=fontsizetickx)
    pl.tick_params(axis='y', labelsize=fontsizeticky)
    #pl.xlim([0,55000])
    pl.savefig(dossier + titre_x[i] + "_evolution" + format_)

    if plot_log[i]==1:
        pl.clf()
        for j in range(len(simu)):
            pl.plot(np.arange(np.size(estimated_para[j][:,i]))[::k[i]],
                    estimated_para[j][::k[i],i], 'x', color=c[j])
        pl.xlabel(r'iteration $k$', fontsize=fontsizelabelx)
        pl.ylabel(label_x[i], fontsize=fontsizelabely)
        pl.tick_params(axis='x', labelsize=fontsizetickx)
        pl.tick_params(axis='y', labelsize=fontsizeticky)
        #pl.xlim([0,55000])
        pl.yscale('log')
        pl.savefig(dossier + titre_x[i] + "_evolution_log" + format_)

if evol2D :
    for i in [0,1,2,3,4,5,6,7,8,9]:
        for l in [1,2,3,4,5,6,7,8,9]:
            if l>i:
                pl.clf()
                for j in range(len(simu)):
                    pl.plot(estimated_para[j][:,i], estimated_para[j][:,l],
                            ':', color=c[j])
                    pl.plot(estimated_para[j][0,i], estimated_para[j][0,l],
                            'o', color=c[j])
                pl.xlabel(label_x[i], fontsize=fontsizetickx)
                pl.ylabel(label_x[l], fontsize=fontsizeticky)
                if plot_log[i]==1: pl.xscale('log')
                if plot_log[l]==1: pl.yscale('log')
                pl.savefig(dossier + "2D_" + titre_x[i] + '_' + titre_x[l] +
                            "_evolution" + format_)

pl.clf()
for j in range(len(simu)):
    pl.plot(np.arange(np.size(estimated_para[j][:,3]))[::k[3]],
                estimated_para[j][::k[3],3]*Hmax/estimated_para[j][::k[3],2],
                'x', color=c[j])
pl.xlabel(r'iteration $k$', fontsize=fontsizelabelx)
pl.ylabel(r'$\max_{a,i}\, h_a^i\times \beta_1^k/\beta_0^k$',
            fontsize=fontsizelabelx)
pl.tick_params(axis='x', labelsize=fontsizetickx)
pl.tick_params(axis='y', labelsize=fontsizeticky)
#pl.xlim([0,55000])
if plot_log[3]==1: pl.yscale('log')
pl.savefig(dossier + "beta1_Hmax_beta0_evolution" + format_)
