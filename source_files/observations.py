# -*- coding:Utf-8 -*-
import numpy as np
import simulationParameters as para
"""The variable 'observation' contains the data observation such that
    observation = [data2008,...,data2023]
with the following structure for each year: for data of 2008
    data2008[0,i] : abscissa index of the i-th quadrat visited in 2008
    data2008[1,i] : ordinate index of the i-th quadrat visited in 2008
    data2008[2,i] : % of observed infected trees of the i-th quadrat in 2008
    data2008[3,i] : number of observations of the i-th quadrat in 2008
"""

dataDSF = np.loadtxt('data_chalara.txt', dtype=str)
dataDSF = dataDSF[1:]
y = dataDSF[:,[3,6,7,10]].astype(float)
'''y[i,0] = year of the i-th observation
   y[i,1] = quadrat abscissa index
   y[i,2] = quadrat ordinate index
   y[i,3] = proportion of infected trees'''
y[:,1] += para.nb_fic_x_left #add the fictive quadrats
y[:,2] += para.nb_fic_y_bottom #add the fictive quadrat

xl2 = np.loadtxt('data_chalara.txt', dtype=str)[:,6]
yl2 = np.loadtxt('data_chalara.txt', dtype=str)[:,7]

observation = []

for k in np.arange(16):
    annee = 2008+k
    a = np.zeros((np.size(y[y[:,0]==annee,0]),4))
    a[:,0] = y[y[:,0]==annee,1] #quadrat abscissa
    a[:,1] = y[y[:,0]==annee,2] #quadrat ordinate
    a[:,2] = y[y[:,0]==annee,3] #proportion of infected trees
    a[:,3] = 1 #number of observations of the quadrat during the year

    #agregation of the multiple observations of the quadrat during the year
    i = 0
    while i<np.size(a[:,0]):
        j = i+1
        while j<np.size(a[:,0]):
            if  a[j,0]==a[i,0] and a[j,1]==a[i,1]:
                a[i,2] = ((a[i,2]*a[i,3]+a[j,2]*a[j,3])/(a[i,3]+a[j,3]))
                a[i,3] += a[j,3]
                a = np.delete(a, j, axis=0)
            else:
                j+=1
        i+=1
    a[:,2] *= 0.01
    observation.append(a.T)
