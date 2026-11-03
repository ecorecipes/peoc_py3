# -*- coding:Utf-8 -*-
import numpy as np
import matplotlib.pyplot as pl
import simulationParameters as para
import observations
import environmental_data as env_data
import sys
import os
import initialisation_rachis as init_rachis
import plot_graphs as pg
import reaction_diffusion as RD
import matplotlib.pyplot as plt
import matplotlib.lines
            
#plt.style.use('classic')

ext = pg.ext
ash_density = env_data.ash_density

def initialisation(dossier):
    obs = observations.observation

    x = np.arange(np.size(ash_density[:,0]))
    y = np.arange(np.size(ash_density[0,:]))
    [Y,X] = np.meshgrid(y,x)

    x = X[ash_density!=0]
    y = Y[ash_density!=0]

    rS = 1000
    rachis = [init_rachis.rachis_a_moins_2*ash_density,
                  init_rachis.rachis_a_moins_1*ash_density]
    maxi = max(np.max(init_rachis.rachis_a_moins_2*ash_density)
               ,np.max(init_rachis.rachis_a_moins_1*ash_density))
    rachis[0][ash_density==0] = -0.1*maxi
    rachis[1][ash_density==0] = -0.1*maxi
    
    l = 0
    for k in [para.initialization_year-2008-2, para.initialization_year-2008-1]:
        if k>=-1:
            pl.clf()
            #pl.plot(x,y)
            #pl.plot(x2,y2)
            pl.plot(X[ash_density!=0],Y[ash_density!=0],'o',color='g',alpha=0.3)
            pl.plot(x[rachis[l][ash_density!=0]!=0],
                    y[rachis[l][ash_density!=0]!=0], 'o', color='g')
            annee = obs[k+1]
            chal_0_25 = (annee[2,:]>0)*(annee[2,:]<0.25)
            chal_25_50 = (annee[2,:]>=0.25)*(annee[2,:]<0.50)
            chal_50_75 = (annee[2,:]>=0.50)*(annee[2,:]<0.75)
            chal_75_100 = (annee[2,:]>=0.75)
            pl.plot(annee[0,annee[2,:]==0],annee[1,annee[2,:]==0],'x',color='b')
            pl.plot(annee[0,chal_0_25>0],annee[1,chal_0_25>0],'x',color='c')
            pl.plot(annee[0,chal_25_50>0],annee[1,chal_25_50>0],'x',color='m')
            pl.plot(annee[0,chal_50_75>0], annee[1,chal_50_75>0],'x',color='r')
            pl.plot(annee[0,chal_75_100>0],annee[1,chal_75_100>0],'x',color='k')
            pl.xlim([0,np.max(X)])
            pl.ylim([0,np.max(Y)])
            pl.savefig(dossier+'/initialization'+str(2008+k)+ ext)

        pl.clf()
        pg.pcolormesh_flat(X, Y, rachis[l], alpha=0.7, vmin=-0.1*maxi, vmax = maxi)
        cbar = pl.colorbar()#boundaries=np.linspace(0,1,101))
        cbar.set_label('infected rachis', fontsize = 20)
        cbar.ax.tick_params(labelsize=15)
        pl.xlim([0,np.max(X)])
        pl.ylim([0,np.max(Y)])
        if k>=-1:
            pl.plot(annee[0,annee[2,:]==0],annee[1,annee[2,:]==0],'x',color='b')
            pl.plot(annee[0,chal_0_25>0],annee[1,chal_0_25>0], 'x', color='c')
            pl.plot(annee[0,chal_25_50>0],annee[1,chal_25_50>0],'x',color='m')
            pl.plot(annee[0,chal_50_75>0],annee[1,chal_50_75>0],'x',color='r')
            pl.plot(annee[0,chal_75_100>0],annee[1,chal_75_100>0],'x',color='k')
        pl.savefig(dossier+'/initialization'+str(2008+k)+'bis' + ext)
        l+=1

def plot_data(dossier):
    if os.path.exists(dossier+'/environmental_data')==False:
        os.mkdir(dossier+'/environmental_data')
    x = np.arange(np.size(ash_density[:,0]))
    y = np.arange(np.size(ash_density[0,:]))
    [y,x] = np.meshgrid(y,x)

    data_to_plot = np.ma.masked_where(ash_density==0, ash_density)
    pg.plot_heat_map_with_data(x, y, data_to_plot,
                dossier+'/environmental_data/ash_density' + ext,
                vmin=0, vmax=np.max(data_to_plot),
                cbarlabel = r'basal area of ashes (m$^2$/ha)', alpha=1)

    temperature = env_data.T
    label = env_data.label_T
    for i in np.arange(16):
        data_to_plot = np.ma.masked_where(ash_density==0, temperature[i])
        pg.plot_heat_map_with_data(x, y, data_to_plot,
                dossier+'/environmental_data/temperature_'+ str(para.temp_index)
                + '_' +str(2007+i)+ext,
                vmin=0, vmax=np.max(temperature),
                cbarlabel = label, alpha=1, title = str(2007+i))
    

    rainfall = env_data.H[:12]
    for i in np.arange(12):
        data_to_plot = np.ma.masked_where(ash_density==0, rainfall[i])
        pg.plot_heat_map_with_data(x, y, data_to_plot,
                dossier+'/environmental_data/rainfall'+str(2007+i)+ext,
                vmin=0, vmax=np.max(rainfall),
                cbarlabel = r'rainfall (mm)', alpha=1, title = str(2007+i))


def evolution_chalarose(path,p, infection_1,infection_2,liste_data_to_plot,
                            annee_sup = False):
    (r_symp, r_allee, beta_0, beta_1, D, S_sat, gamma, C_recover, C_init,
         puiss) = p
    index_temp = para.initialization_year - para.first_year_of_data
    f_temp1 = para.f_temperature(env_data.T[index_temp],gamma,puiss)
    f_temp2 = para.f_temperature(env_data.T[1+index_temp],gamma,puiss)
    f_temp1[f_temp1==0] = 1.0
    f_temp2[f_temp2==0] = 1.0
    
    relative_fongi_1 = C_init*r_symp*infection_1 /f_temp1
    relative_fongi_2 = C_init*r_symp*((infection_2-C_recover*infection_1)/
                        ((1-C_recover*infection_1*C_init)*f_temp2))
    
    [fungi_per_ash, spore_non_sature] = RD.reaction_di(r_allee, beta_0, beta_1,
                                        D, S_sat,relative_fongi_1,
                                        relative_fongi_2, annee_sup, True)

    np.savetxt(path + '/parameters_of_the_model.txt',p)
    np.save(path + '/relative_fongi_1.npy',relative_fongi_1)
    np.save(path + '/relative_fongi_2.npy',relative_fongi_2)
    
    nx = np.size(ash_density[:,0])
    ny = np.size(ash_density[0,:])
    x = np.arange(nx)
    y = np.arange(ny)
    [y,x] = np.meshgrid(y,x)

    for type_of_data in liste_data_to_plot:
        l = 0
        if type_of_data in ['spores_quantity', 'saturated_spores_quantity',
                           'fct_temp', 'chi_a']:
            k_annee = 2
        else:
            k_annee = 0
        
        old_q = np.zeros((nx,ny))
        para_bernoulli = np.zeros((nx,ny))
        while ((l<para.nb_years - k_annee) or ((l==para.nb_years - k_annee)
                and annee_sup and (type_of_data in ['nb_frenes',
                    'para_bernoulli', 'new_infection']))):
        
            old_q = para_bernoulli*1.0
            f_temp = para.f_temperature(env_data.T[l+index_temp],gamma,puiss)
            q_temp = para.para_bernoulli_symptom(fungi_per_ash[l],
                                                 env_data.T[l+index_temp],
                                                 gamma,puiss,r_symp)
            para_bernoulli = q_temp*(1-C_recover*old_q)+C_recover*old_q
        
            if type_of_data == 'nb_frenes':
                (data_to_plot, maxi_v) = (para_bernoulli*ash_density,
                                            np.max(para_bernoulli*ash_density))
            elif type_of_data == 'para_bernoulli':
                (data_to_plot, maxi_v) = (para_bernoulli, 1)
            elif type_of_data == 'new_infection':
                (data_to_plot, maxi_v) = (q_temp, 1)
            elif type_of_data == 'fct_temp':
                data_to_plot = 1-para.f_temperature(env_data.T[k_annee+l],gamma,
                                                     puiss)
                maxi_v = 1
            elif type_of_data =='spores_quantity':
                data_to_plot = spore_non_sature[l]
                maxi_v = np.max(spore_non_sature)
            elif type_of_data =='saturated_spores_quantity':
                (data_to_plot, maxi_v) = (spore_non_sature[l], S_sat)
                data_to_plot[data_to_plot>S_sat] = S_sat
            elif type_of_data =='chi_a':
                (data_to_plot, maxi_v) = (spore_non_sature[l],
                                              S_sat*np.max(ash_density))
                data_to_plot[data_to_plot>S_sat] = S_sat
                data_to_plot*=ash_density
                
            data_to_plot = np.ma.masked_where(ash_density==0, data_to_plot)
            plotdata = False
            obserdata = None
            if l<para.nb_years - k_annee:
                plotdata = True
                obserdata = observations.observation[k_annee+l]
            pg.plot_heat_map_with_data(x, y, data_to_plot,
                path+ pg.label_type(type_of_data, k_annee+2008+l,'path'),
                vmin=0, vmax=maxi_v,
                cbarlabel = pg.label_type(type_of_data,k_annee+2008+l-1,
                                          'label_bar'),
                title='$a=$' + str(k_annee+2008+l),
                plotobservationdata = plotdata, observationdata = obserdata)
            pg.plot_legend_separatly(path+'/legend' + ext)
            l+=1
