# -*- coding:Utf-8 -*-
import time
import sys
import numpy as np
import os
import shutil
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
import scipy.optimize
from matplotlib.colors import LogNorm
import modify_save_format_regular_grid as msfrg
import plot_graphs as pg
#plt.style.use('classic')

i = 1
while os.path.exists('Simulations/Likelihoods_on_regular_grids_' +str(i)):
    i += 1
dossier = 'Simulations/Likelihoods_on_regular_grids_' +str(i) +'/'
os.mkdir(dossier)
shutil.copy("9_graph_log_likelihood_on_regular_grid.py", dossier +
                '/graph_log_likelihood_on_regular_grid.py')
sys.path.append(os.getcwd()+'/'+dossier)
from graph_log_likelihood_on_regular_grid import *

for j in range(len(simus)):
    if os.path.exists('Simulations/'+ simus[j] + '/log_likeli_tab.npy')==False:
        msfrg.transform_data('Simulations/'+ simus[j])
    elif (os.path.getmtime('Simulations/'+ simus[j] + '/log_likelihood.npy')>
            os.path.getmtime('Simulations/'+ simus[j] + '/log_likeli_tab.npy')):
        msfrg.transform_data('Simulations/'+ simus[j])

print("Graphs are added in the folder ", dossier)

def para_meshgrid(v_p_all, p_simu):
    index = np.ones(np.size(v_p_all))
    index[[l for l in range(np.size(v_p_all))
                if v_p_all[l] not in p_simu]] = 0
    
    if np.size(p_simu)>1:
        p_add = 2*p_simu[-1]-p_simu[-2]
    else:
        ind = np.argwhere(v_p_all==p_simu[-1])
        if p_simu[0]<v_p_all[-1]:
            p_add = v_p_all[ind+1]
        else:
            p_add = 2*p_simu[-1]-v_p_all[ind-1]

    return [index, np.append(p_simu, p_add)]

def tick_string(l):
    if l.is_integer(): return str(int(l))
    else: return str(l)

    
#-------- parameters recovery
ext = pg.ext
label_p = [pg.labels[2], pg.labels[4], pg.labels[6], pg.labels[9], pg.labels[0],
           pg.labels[1], pg.labels[3], pg.labels[5], pg.labels[7], pg.labels[8]]
label_x = [pg.legen[2], pg.legen[4], pg.legen[6], pg.legen[9], pg.legen[0],
           pg.legen[1], pg.legen[3], pg.legen[5], pg.legen[7], pg.legen[8]]

parameters_number = len(label_p)

v_p = [ np.zeros(0)]*parameters_number
parameters = ['a']*parameters_number

for i in range(parameters_number):
    parameters[i] = ['a']*len(simus)
    for j in range(len(simus)):
        parameters[i][j] = np.load('Simulations/'+ simus[j] + '/' + label_p[i]
                                       + '_values.npy')
        v_p[i] = np.append(v_p[i], parameters[i][j])
    v_p[i] = np.unique(np.sort(v_p[i]))
    
err = 10**(-30)
for i in range(parameters_number):
    for j in range(np.size(v_p[i])-1):
        if np.abs(v_p[i][j]-v_p[i][j+1])<err:
            for k in range(len(simus)):
                parameters[i][k][parameters[i][k]==v_p[i][j+1]] = v_p[i][j]
            v_p[i][j+1] = v_p[i][j]
    v_p[i] = np.unique(np.sort(v_p[i]))

print("Values of the parameters:")
for i in range(parameters_number):
    print(label_p[i], "=", v_p[i])


#-------- likelihood recovery
size_p = [0]*parameters_number
for i in range(parameters_number):
    size_p[i] = np.size(v_p[i])
log_vraisemblance = np.ones(tuple(size_p))

index_p = ['a']*parameters_number
for j in range(len(simus)):
    change_index = np.ones(tuple(size_p))
    for i in range(parameters_number):
        axis1 = np.arange(parameters_number)
        axis1[axis1<=i] -= 1
        axis1[0] = i
        axis2 = np.arange(parameters_number)
        axis2[axis2<i] += 1
        axis2[i] = 0
        change_index = np.transpose(change_index, tuple(axis1))
        change_index[[l for l in np.arange(size_p[i])
                    if v_p[i][l] not in parameters[i][j]]] = 0
        change_index = np.transpose(change_index, tuple(axis2))
    log_vraisemblance[change_index==1] = np.load('Simulations/'+simus[j] +
                                            '/log_likeli_tab.npy').ravel()

#keep only parameters values to plot
para_to_plot = [plot_beta0, plot_D, plot_gamma, plot_kappa,
                plot_r_sym,  plot_r_allee, plot_beta1, plot_S_sat,
                plot_C_recover, plot_C_init]
def indice_in_plot_p(plot_p, vect_p):
    "return the indexes of para of vect_p in plot_p"
    ax = list(range(np.size(vect_p)))
    for j in range(np.size(vect_p)):
          if np.sum(np.abs(vect_p[j]-plot_p)<err)==0:
                ax.remove(j)
    return ax
for i in range(parameters_number):
    if para_to_plot[i][0]!="all":
        ax = indice_in_plot_p(para_to_plot[i], v_p[i])
        v_p[i] = v_p[i][ax]
        log_vraisemblance = log_vraisemblance.take(ax, axis=i)
        
        for j in range(len(simus)):
            ax = indice_in_plot_p(para_to_plot[i], parameters[i][j])
            parameters[i][j] = parameters[i][j][ax]

#-----
if np.size(log_vraisemblance[log_vraisemblance==1])!=0:
    print("Number of missing parameters on the total regular grid:", end=' ')
    print(np.size(log_vraisemblance[log_vraisemblance==1]), end=' ')    
    print("on ", np.size(log_vraisemblance))

n_para = np.sum(log_vraisemblance!=1, axis=(2,3,4,5,6,7,8,9))

maxi_vrai = np.max(log_vraisemblance[log_vraisemblance<0])
if maxi!="None":
    maxi_vrai = max(maxi_vrai, maxi)
mini_vrai = np.min(log_vraisemblance)
if mini!=None and mini<maxi_vrai:
    mini = min(max(mini,mini_vrai), maxi_vrai)
else:
    mini = mini_vrai
    log_vraisemblance[log_vraisemblance==0] = mini_vrai*2
log_vraisemblance[log_vraisemblance==1] = mini_vrai*3

print("Maximal likelihood = ", maxi_vrai, end=' ')
print("obtained with:")
ind = np.unravel_index(np.argmax(log_vraisemblance, axis=None),
       log_vraisemblance.shape)
for i in range(parameters_number):
    print("   ", label_p[i], "=", v_p[i][ind[i]])

#-------- tested parameters
log_vrai_max = np.max(log_vraisemblance, axis=(2,3,4,5,6,7,8,9))
log_vrai_max[(log_vrai_max<mini)*(log_vrai_max>mini_vrai*3)] = mini
log_vrai_max[log_vrai_max==mini_vrai*3] = non_test

nbmax_simu_by_para = np.max(n_para)
pl.clf()
for i in range(np.size(v_p[0])):
    D_temp = v_p[1][n_para[i]!=0]
    pl.plot(v_p[0][i]*np.ones(np.size(D_temp)), D_temp, 'x', color='r')

    D_temp = v_p[1][n_para[i]==nbmax_simu_by_para]
    pl.plot(v_p[0][i]*np.ones(np.size(D_temp)), D_temp, 'x', color='b')
pl.xlabel(label_x[0])
pl.ylabel(label_x[1])
pl.savefig(dossier+'tested_para' + ext)

def plot_best_para(loglikeli, para, folder, index_fig="", title_fig=""):
    for i in range(2,parameters_number):
        if np.size(para[i])!=1:
            ax = list(range(2,parameters_number))
            ax.remove(i)
            ax = tuple(ax)
            p_max = para[i][np.argmax(np.max(loglikeli, axis=ax), axis=2)]
            pl.clf()
            pl.contourf(para[0], para[1], p_max.T,20)
            pl.xlabel(label_x[0], fontsize=fontsizex)
            pl.ylabel(label_x[1], fontsize=fontsizey)
            pl.tick_params(axis='both', labelsize=fontsizeticks)
            cbar = pl.colorbar()
            cbar.set_label('best ' + label_x[i], fontsize=fontsizecbar)
            cbar.ax.tick_params(labelsize = fontsizeticks)

            if title_fig!="":
                pl.title(title_fig)
                
            pl.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)
            pl.savefig(folder + 'best_' + label_p[i] + index_fig + ext)

            if index_fig=="":
                p_max[log_vrai_max==non_test] = np.min(v_p[i])
                pl.clf()
                [index1, x] = para_meshgrid(para[0],para[0])
                [index1, y] = para_meshgrid(para[1],para[1])
                [mat_D,mat_beta_0] = np.meshgrid(y,x)
                pl.pcolormesh(mat_beta_0, mat_D, p_max,
                              vmin=np.min(p_max), vmax=np.max(p_max))
                pl.xlabel(label_x[0], fontsize=fontsizex)
                pl.ylabel(label_x[1], fontsize=fontsizey)
                pl.colorbar().set_label('best ' + label_x[i],
                                            fontsize=fontsizecbar)
                pl.tick_params(axis='both', labelsize=fontsizeticks)
                pl.xlim(np.min(x), np.max(x))
                pl.ylim(np.min(y), np.max(y))
                pl.savefig(dossier + label_p[i] + '_max2' + ext)
 

            
def plot_likelihood(loglikeli, para, folder, index_fig="", title_fig="",
                        nb_graphe=1):
    for i in range(parameters_number):
        for j in range(parameters_number):
            if j>i and np.size(para[i])!=1 and np.size(para[j])!=1:
                ax = list(range(parameters_number))
                ax.remove(i)
                ax.remove(j)
                log_vrai_max = np.max(loglikeli, axis=tuple(ax))
                log_vrai_max[(log_vrai_max<mini)*(log_vrai_max>mini_vrai*3)
                                 ] = mini
                log_vrai_max[log_vrai_max==mini_vrai*3] = non_test
                #-------
                for graphe in np.arange(nb_graphe)+1:
                    pl.clf()
                    if graphe==1:
                        pl.contourf(para[i], para[j], log_vrai_max.T,20)
                        cbar = pl.colorbar()
                        oldticks = cbar.ax.get_yticks()
                        newticks = [tick_string(l) for l in
                                    np.interp(oldticks,
                                    cbar.ax.get_ylim(),
                                    # Colorbar.get_clim() was removed in
                                    # matplotlib 3.x; the mappable shares
                                    # the colorbar's norm, so same values.
                                    cbar.mappable.get_clim())]
                        newticks[0] = '<' + newticks[0]
                        pl.clim(mini, maxi_vrai)
                        locator = cbar.ax.yaxis.get_major_locator()
                        if len(getattr(locator, 'locs', newticks)
                               ) != len(newticks):
                            # matplotlib >= 3 gives contourf colorbars a
                            # FixedLocator on all the levels (only some are
                            # visible); matplotlib 2.2 (the reference) holds
                            # exactly the visible ticks, so this branch is
                            # never taken there.
                            # Since 3.x the colorbar axis is in data units.
                            cbar.set_ticks(oldticks)
                        cbar.ax.set_yticklabels(newticks)
                        #cbar.set_ticks(cbar.ax.get_yticks(), labels = newticks)
                        #cbar.set_ticks(newticks)
                    if graphe==2:
                        [mat_y,mat_x] = np.meshgrid(para[j], para[i])
                        pg.pcolormesh_flat(mat_x,mat_y, log_vrai_max,
                                  vmin=non_test, vmax=maxi_vrai)
                        cbar = pl.colorbar()
                    #if title_fig<>"":
                    #    pl.title(title_fig)
                    pl.xlabel(label_x[i], fontsize=fontsizex)
                    pl.ylabel(label_x[j], fontsize=fontsizey)
                    pl.tick_params(axis='both', labelsize=fontsizeticks)
                    cbar.ax.tick_params(labelsize = fontsizeticks)
                    cbar.set_label('maximal log-vraisemblance',
                                       fontsize=fontsizecbar)
                    pl.subplots_adjust(left=0.1,right=0.9,top=0.95,bottom=0.15)
                    pl.savefig(folder+'vrais_' + label_p[i] + '_' + label_p[j] +
                                   str(graphe) + index_fig + ext)
              
plot_best_para(log_vraisemblance, v_p, dossier)
plot_likelihood(log_vraisemblance, v_p, dossier, nb_graphe = 2)

for i in [2,3,4,5,6,7,8,9]:
    if np.size(v_p[i])!=1:
        if os.path.exists(dossier + label_p[i] + 'fixe/')==False:     
            os.mkdir(dossier + label_p[i] + 'fixe/')
        para = v_p[:]
        for l in range(np.size(v_p[i])): 
            para[i] = [v_p[i][l]]
            plot_likelihood(log_vraisemblance.take([l], axis=i), para,
                            dossier + label_p[i] + 'fixe/', index_fig='_' +
                            str(l), title_fig=label_p[i] + '=' + str(v_p[i][l]))
            plot_best_para(log_vraisemblance.take([l], axis=i), para,
                            dossier + label_p[i] + 'fixe/', index_fig='_' +
                            str(l), title_fig=label_p[i] + '=' + str(v_p[i][l]))

if logscale_r_allee :
    v_p[5][v_p[5]==0] = value_0
    v_p[5][v_p[5]!=0] = np.log(v_p[5][v_p[5]!=0])/np.log(10)
    
    
    ax = list(range(2,parameters_number))
    ax.remove(5)
    ax = tuple(ax)
    p_max = v_p[5][np.argmax(np.max(log_vraisemblance, axis=ax), axis=2)]
    pl.clf()
    pl.contourf(v_p[0], v_p[1], p_max.T,20)
    pl.xlabel(label_x[0], fontsize=fontsizex)
    pl.ylabel(label_x[1], fontsize=fontsizey)
    pl.tick_params(axis='both', labelsize=fontsizeticks)

    cbar = pl.colorbar()
    oldticks = cbar.ax.get_yticks()
    newticks = [tick_string(10**l) for l in np.interp(oldticks,
                            cbar.ax.get_ylim(), cbar.mappable.get_clim())]
    newticks[0] = 0
    newticks[1] = ""
    # see plot_likelihood: only taken on matplotlib >= 3
    if len(getattr(cbar.ax.yaxis.get_major_locator(), 'locs', newticks)
           ) != len(newticks):
        cbar.set_ticks(oldticks)
    cbar.ax.set_yticklabels(newticks)
    cbar.set_label('best ' + label_x[5], fontsize=fontsizecbar)
    cbar.ax.tick_params(labelsize = fontsizeticks)
            
    pl.savefig(dossier + label_p[5] + '_max_log_scale' + ext)
