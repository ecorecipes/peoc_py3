# -*- coding:Utf-8 -*-
"""This script plots the posterior distributions of different AMIS runs. It can
be used for example to compare the posterior distributions for different climate
variables.
To plot the posterior distributions, enter the names of the simulations to 
compare in the file 7_simulations_AMIS_to_compare.py, then run main_script.py 
and select action 7.
The graphs will be plotted in the folder 'Simulations/Compare_posterior_1' (or
in the folder 'Simulations/Compare_posterior_(i+1)' if the folder 
'Simulations/Compare_posterior_i' already exists).
"""

import numpy as np
import matplotlib.pyplot as pl
import shutil
import run_config
import os
import time
import sys
import plot_graphs
import matplotlib.pyplot as plt
#plt.style.use('classic')

ext = plot_graphs.ext
legen = plot_graphs.legen
labels = plot_graphs.labels

i = 1
while os.path.exists('Simulations/Compare_posterior_' +str(i)): i += 1
dossier = 'Simulations/Compare_posterior_' +str(i) +'/'
os.mkdir(dossier)
shutil.copy(run_config.source("7_simulations_AMIS_to_compare.py"), dossier +
                '/simulations_AMIS_to_compare.py')
sys.path.append(os.getcwd()+'/'+dossier)
from simulations_AMIS_to_compare import *

print("Graphs are added in the folder ", dossier)

para = ['a']*len(simu)
weight = ['a']*len(simu)
ND_para = ['a']*len(simu)

for i in range(len(simu)):
    para[i]  = np.load('Simulations/' + simu[i] + '/sampled_parameters.npy')
    para[i] = para[i].reshape((para[i].shape[0]*para[i].shape[1],
                                para[i].shape[2]))
    weight[i]  = np.load('Simulations/' + simu[i] +'/weight.npy')
    weight[i] = weight[i].reshape(weight[i].shape[0]*weight[i].shape[1])
    
for j in range(10):
    pl.clf()
    for i in range(len(simu)):
        plot_graphs.draw_histo_1D(para[i][:,j], legen[j], col=c[i],
                                  weights_x=weight[i], alpha_=0.2, label_=l[i])
    pl.legend(fontsize=20)
    pl.savefig(dossier + '/law_' + labels[j] + ext)
