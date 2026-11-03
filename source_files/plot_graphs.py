# -*- coding:Utf-8 -*-
import numpy as np
import matplotlib.pyplot as pl
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.lines
from py2_compat import py2_str
            
#plt.style.use('classic')
ext = '.png'

#markers of the observation data
colormarkers = ['k', 'w', 'c', 'm', 'r']
labelmarkers = [r'$p_a(i)=0$', r'$p_a(i) \in ]0,25]$', r'$p_a(i) \in ]25,50]$',
                r'$p_a(i) \in ]50,75]$', r'$p_a(i) \in ]75,1]$']

#graph parameters for the 2D-distributions
echelle = 10
nb_classes = 120

#labels of each parameter
legen = [r'$r_S$', r'$r$', r'$\beta_0$', r'$\beta_1$',
         r'$D$', r'$S$', r'$\gamma$', r'$C_{pers}$', r'$C_{init}$', r'$\kappa$']
legen_iter = [r'$r_S^k$', r'$r^k$', r'$\beta_0^k$', r'$\beta_1^k$', r'$D^k$',
           r'$S^k$', r'$\gamma^k$', r'$C_{pers}^k$', r'$C_{init}^k$',
               r'$\kappa^k$']
labels = ['r_symp', 'r_allee', 'beta_0', 'beta_1', 'D', 'S', 'gamma',
           'C_pers', 'C_init', 'kappa']

#size of labels and ticks
font_size_x = 16
font_size_y = 16
font_size_z = 16
ticksize = 12

    
def pcolormesh_flat(x, y, z, **kwargs):
    """pcolormesh as in matplotlib < 3.3 (used by the original PEOC): with
    shading='flat' and C of the same shape as X and Y, the last row and column
    of C are silently dropped. Recent matplotlib raises an error (or uses
    'nearest' shading) instead, so drop them explicitly."""
    if np.shape(z) == np.shape(x):
        z = z[:-1, :-1]
    return pl.pcolormesh(x, y, z, shading='flat', **kwargs)


def label_type(type_of_data, year, type_label):
    if type_of_data == 'nb_frenes':
        label_bar = 'density of infected ashes in ' + str(year)
        path = '/density_infected_ashes_' + str(year) + ext
    elif type_of_data == 'para_bernoulli':
        label_bar = r'Bernoulli parameter $q_{a}$'
        path = '/bernoulli_parameter_' + str(year) + ext
    elif type_of_data == 'new_infection':
        label_bar = r'new infection probability $\tilde q_{0}$'.format('{'+
                                                                str(year+1)+'}')
        path = '/new_infection_probability_' + str(year) + ext
    elif type_of_data == 'fct_temp':
        label_bar = '1-f(T('+ str(year) +'))'
        path = '/fct_temperature_' + str(year) + ext
    elif type_of_data =='spores_quantity':
        label_bar = r'spores quantity  $w_{a-1}(\tau,x)$'
        path = '/spores_quantity_' + str(year) + ext
    elif type_of_data =='saturated_spores_quantity':
        label_bar = r'saturated spores quantity  $w_{a-1}(\tau,x)\wedge S$'
        path =  '/spores_quantity_sat' + str(year) + ext
    elif type_of_data =='chi_a':
        label_bar = r'$\chi_{a-1}$'
        path =  '/chi_a_' + str(year) + ext

    if type_label=='label_bar': return label_bar
    if type_label=='path': return path

def plot_data(annee):
    chal_0_25 = (annee[2,:]>0)*(annee[2,:]<=0.25)
    chal_25_50 = (annee[2,:]>0.25)*(annee[2,:]<=0.50)
    chal_50_75 = (annee[2,:]>0.50)*(annee[2,:]<=0.75)
    chal_75_100 = (annee[2,:]>0.75)
    pl.plot(annee[0,annee[2,:]==0],annee[1,annee[2,:]==0], 'x',
                color=colormarkers[0], label=labelmarkers[0])
    pl.plot(annee[0,chal_0_25>0], annee[1,chal_0_25>0], 'x',
                color=colormarkers[1], label=labelmarkers[1])
    pl.plot(annee[0,chal_25_50>0], annee[1,chal_25_50>0], 'x',
                color=colormarkers[2], label=labelmarkers[2])
    pl.plot(annee[0,chal_50_75>0], annee[1,chal_50_75>0], 'x',
                color=colormarkers[3], label=labelmarkers[3])
    pl.plot(annee[0,chal_75_100>0], annee[1,chal_75_100>0], 'x',
                color=colormarkers[4], label=labelmarkers[4])
    #pl.legend(loc=3, prop={'size': 11})

def plot_legend_separatly(path):
    pl.clf()
    handles_list = []
    for i in range(len(colormarkers)):
        handles_list += [matplotlib.lines.Line2D([],[], color = colormarkers[i],
                            marker = 'x', linestyle = 'None', markersize = 15,
                            label = labelmarkers[i])]
    pl.legend(handles = handles_list, facecolor = "gainsboro")
    pl.axis('off')
    pl.savefig(path)


def plot_heat_map_with_data(x, y, z, path, vmin=None, vmax=None, cbarlabel="",
                            title="", plotobservationdata = False,
                            observationdata = None, alpha = 0.7, savefig=True):
    pl.clf()
    pcolormesh_flat(x, y, z, alpha=alpha, vmin=vmin, vmax=vmax)
    pl.xlim([0,np.max(x)])
    pl.ylim([0,np.max(y)])
    cbar = pl.colorbar()
    cbar.ax.tick_params(labelsize=15)
    if cbarlabel==r'saturated spores quantity  $w_{a-1}(\tau,x)\wedge S$':
        cbar.set_label(cbarlabel, fontsize = 14)
    else:
        cbar.set_label(cbarlabel, fontsize = 18)
    pl.title(title, fontsize=18)
    if plotobservationdata:
        plot_data(observationdata)
    pl.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    pl.tick_params(axis='y', left=False, right=False, labelleft=False)
    pl.axis('off')
    if savefig:
        pl.savefig(path)


def draw_histo_1D(x, label_x, col, weights_x=None, alpha_=1, label_=None):
    h = (np.max(x)-np.min(x))/echelle + 1*(np.max(x)==np.min(x))
    classes_histo = np.linspace(np.min(x)-h, np.max(x)+h, nb_classes)
    pl.hist(x, classes_histo , density=True, weights=weights_x, stacked=True,
                color=col, histtype='stepfilled', alpha = alpha_, label=label_)
    pl.xlabel(label_x, fontsize=font_size_x)
    pl.tick_params(axis='x', labelsize=ticksize)
    pl.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)
    pl.tick_params(axis='y', left=False, right=False, labelleft=False)
    
def law_2D(path, x, y, index_x, index_y, weights_x=None):
    "2-dimensional distribution of the index_x-th and index-y-th  parameters."
    nb_range_x = 30+1
    nb_range_y = 30+1
    h_x = (np.max(x)-np.min(x))/echelle + 1*(np.max(x)==np.min(x))
    h_y = (np.max(y)-np.min(y))/echelle + 1*(np.max(y)==np.min(y))
    classes_histo_x = np.linspace(np.min(x)-h_x,np.max(x)+h_x,nb_range_x)
    classes_histo_y = np.linspace(np.min(y)-h_y,np.max(y)+h_y,nb_range_y)
    [range_y,range_x] = np.meshgrid(classes_histo_y,classes_histo_x)

    density = np.histogram2d(x, y, density=True,
                             bins=(classes_histo_x, classes_histo_y),
                             weights = weights_x)[0]
    h = ((np.max(x)-np.min(x)+2*h_x)/(nb_range_x-1)
         *(np.max(y)-np.min(y)+2*h_y)/(nb_range_y-1))
    
    v_x = (classes_histo_x[:-1] + classes_histo_x[1:])/2
    v_y = (classes_histo_y[:-1] + classes_histo_y[1:])/2
    [v_y,v_x] = np.meshgrid(v_y,v_x)
    cov_xy = (np.sum(density*h*v_x*v_y)
              -np.sum(density*h*v_x)*np.sum(density*h*v_y))
    var_x = (np.sum(density*h*v_x*v_x)-np.sum(density*h*v_x)**2)
    var_y = (np.sum(density*h*v_y*v_y)-np.sum(density*h*v_y)**2)

    if var_x<=0 or var_y<=0: r=0
    else: r = cov_xy/(np.sqrt(var_x*var_y))

    pl.clf()
    pl.pcolormesh(range_x, range_y, density)

    cbar = pl.colorbar()
    cbar.set_label('density', fontsize = font_size_z)
    cbar.ax.tick_params(labelsize=ticksize)
    pl.title('Correlation coefficient: ' + py2_str(round(float(r), 2)),
                 fontsize=font_size_x)
    pl.xlabel(legen[index_x], fontsize=font_size_x)
    pl.ylabel(legen[index_y], fontsize=font_size_y)
    pl.tick_params(axis='both', labelsize=ticksize)
    pl.xlim([np.min(range_x),np.max(range_x)])
    pl.ylim([np.min(range_y),np.max(range_y)])
    pl.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.13)
    pl.savefig(path)
