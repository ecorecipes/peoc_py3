# -*- coding:Utf-8 -*-
"""
This script contains the environmental data (rainfall and temperature index) of
the file 'climate_data' as well as the density of the file 'density.txt':
- H[a,x,y] is the rainfall of year a of the quadrat (x,y)
- T[a,x,y] is the temperature index of year a of the quadrat (x,y)
- ash_density[x,y] is the ash density of the quadrat (x,y)

The dimension of H and T is (2022-2007+1,simulationParameters.nb_quadrats_x,
simulationParameters.nb_quadrats_y).
The dimension of ash_density (simulationParameters.nb_quadrats_x,
simulationParameters.nb_quadrats_y).
The temperature index is associated to the label temp_index of the model 
parameters file.
"""

import numpy as np
import simulationParameters as para
import useful_functions

n_x = para.nb_quadrats_x
n_y = para.nb_quadrats_y

nb_years = para.last_year_of_data-para.first_year_of_data+1
file_name = 'climate_data'

def return_environmental_data(row):
    data = np.loadtxt(file_name, dtype=str)[1:,[0,1, 2, row]]
    data = data.astype(float)
    data[:,0] *= 1000
    data[:,1] *= 1000

    data = useful_functions.transform_data(data, 0, 1)
    
    data = data[data[:,2]>para.first_year_of_data-1]
    data[:,2] -= para.first_year_of_data

    X = np.zeros((nb_years,n_x,n_y))-1

    X[np.int_(data[:,2]),np.int_(data[:,0]),np.int_(data[:,1])] = data[:,3]
    X[X==-1] = 0
    return X

# Rainfall
H = return_environmental_data(5)

# Temperatures  
def return_temp(label):    
    if label == 'T24': return return_environmental_data(11)
    if label == 'T26': return return_environmental_data(12)
    if label == 'T28': return return_environmental_data(9)
    if label == 'T30': return return_environmental_data(13)
    if label == 'T35': return return_environmental_data(14)
def return_label_temp(label):
    temp = label[1:]
    return r'number of days over ' +temp + r'$^{\circ}$C'
        
T = return_temp(para.temp_index)
label_T = return_label_temp(para.temp_index)

# Ashes density
density = np.loadtxt('density.txt', dtype=str)
density = density[1:,:].astype(float)

density = useful_functions.transform_data(density, 0, 1)

ash_density = np.zeros((n_x, n_y))
ash_density[np.int_(density[:,0]),
np.int_(density[:,1])] = density[:,-1]
