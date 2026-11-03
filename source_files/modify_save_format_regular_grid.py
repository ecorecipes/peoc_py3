# -*- coding:Utf-8 -*-
import sys
import numpy as np
import os
from py2_compat import py2_str
                        
def transform_data(dossier):
    if (os.path.exists(dossier + '/tested_parameters.npy')==False or
            os.path.exists(dossier + '/log_likelihood.npy')==False):
        print("The simulation ", dossier, end=' ')
        print(" is not a computation on a regular grid.")
        return False

    print("Modification of the save format of parameters of simulation", dossier)
    tested_para = np.load(dossier + '/tested_parameters.npy')
    vraisemblance = np.load(dossier + '/log_likelihood.npy')
    rmax = np.size(vraisemblance)

    name_file = ['r_symp', 'r_allee', 'beta_0', 'beta_1', 'D', 'S',
                     'gamma', 'C_pers', 'C_init', 'kappa']
    
    N_para = np.zeros(10)
    vect_all_para = []
    for i in range(10):
        vect_para = tested_para[:,i]
        vect_para = np.unique(vect_para)
        vect_para = np.sort(vect_para)
        N_para[i] = np.size(vect_para)
        vect_all_para += [vect_para]
        print('   ', name_file[i], "=",  vect_para)
        np.save(dossier + '/' + name_file[i] + '_values.npy', vect_para)
    N_para = np.int_(N_para)
    log_vraisemblance = np.zeros((N_para[2],N_para[4],N_para[6],N_para[9],
                                  N_para[0],N_para[1],N_para[3],N_para[5],
                                  N_para[7],N_para[8]))
  
    nb_para = len(tested_para)
    for i in range(nb_para):
        sys.stdout.write("\r{0}/{1}={2}%".format(i,nb_para,py2_str(i*100.0/nb_para)))
        sys.stdout.flush()
        p = tested_para[i]
        log_vraisemblance[vect_all_para[2]==p[2],
                              vect_all_para[4]==p[4],
                              vect_all_para[6]==p[6],
                              vect_all_para[9]==p[9],
                              vect_all_para[0]==p[0],
                              vect_all_para[1]==p[1],
                              vect_all_para[3]==p[3],
                              vect_all_para[5]==p[5],
                              vect_all_para[7]==p[7],
                              vect_all_para[8]==p[8]] = vraisemblance[i]
    print("")

    np.save(dossier + '/log_likeli_tab.npy', log_vraisemblance)

