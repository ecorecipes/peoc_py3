# -*- coding:Utf-8 -*-
"""
Return the initialisation of the infected rachis
- infection_1[j,k] = initialisation of the infected rachis of the quadrat (j,k) for the year -2
- infection_2[j,k] = initialisation of the infected rachis of the quadrat (j,k) for the year -1
"""

import numpy as np
import simulationParameters as para
import useful_functions

n_x = para.nb_quadrats_x
n_y = para.nb_quadrats_y

#memory allocation
infection_1 = np.zeros((n_x, n_y))   #infected rachis for the year -2
infection_2 = np.zeros((n_x, n_y))   #infected rachis for the year -1

init_chalara = np.loadtxt('rachis_initialization.txt',
                                dtype=str)[1:,[1,2,8,9]]
    
init_chalara = init_chalara[init_chalara[:,3]!='0'].astype(float)
init_chalara[:,0] /= 1000
init_chalara[:,1] *= 1000

init_chalara = useful_functions.transform_data(init_chalara, 0, 1)

infection_1[np.int_(init_chalara[:,0]),
                        np.int_(init_chalara[:,1])] = init_chalara[:,2]
infection_2[np.int_(init_chalara[:,0]),
                        np.int_(init_chalara[:,1])] = init_chalara[:,3]

