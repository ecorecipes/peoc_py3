# -*- coding:Utf-8 -*-
"""
Enter in this file the parameters of the different actions you want to run by
the script main_script.py.

The parameters are organized by sections of parameters
- parameters of the model (actions 1, 2, ...)
- parameters for the discretization of the reaction-diffusion equation
- parameters of algorithms for parameters estimation (action 1 or 2)
"""

################################################################################
##    PARAMETERS OF THE MODEL      
################################################################################
lenght_of_a_quadrat = 16.      #lenght of each quadrat (in km)
number_of_observed_trees = 30          #number of observed trees by observation

def f_temperature(T, gamma, kappa):
    """return f(T), where T is the temperature index and gamma and kappa are
     parameters of the function f"""
    z = 1-T*1.0/gamma
    z[z<0] = 0
    return z**kappa #return ([1-T/gamma]_+)^kappa

def para_bernoulli_symptom(fongi_per_ash, T, gamma, kappa, r_symp):
    """the symptoms development by tree follows a Bernoulli distribution 
     depending on the pathogen quantity (fongi_per_ash) and the temperature"""
    q_ber = fongi_per_ash*f_temperature(T,gamma,kappa)/r_symp
    return q_ber

################################################################################
##    ENVIRONMENTAL DATA      
################################################################################
#Environmental data : temp_max in the temperature index type : it can be T24,
#T26, T28, T30 or T35 for the number of days of July and August for which the
#temperature is over 24, 26, 28, 30 or 35°C respectively
temp_index = 'T28'
first_year_of_data = 2007 #first year of temperature and rainfall data
last_year_of_data = 2022 #last year of temperature and rainfall data

plot_curves = True #if True, then plot the data on the folder of the simulation

################################################################################
##    PARAMETERS FOR THE DISCRETIZATION OF THE REACTION - DIFFUSION EQUATION  
################################################################################
nb_fic_x_left = 3 #number of fictitious quadrats in the west of France
nb_fic_x_right = 3 #number of fictitious quadrats in the east of France
nb_fic_y_bottom = 3 #number of fictitious quadrats in the south of France
nb_fic_y_top = 3 #number of fictitious quadrats in the nord of France

#discretisation parameters of the reaction-diffusion
discreti = 1 #number of discretisation points by qudrats in each direction
delta_t = 0.1  #time discretisation
diffusion_period = 60   #period of spore diffusion : 60 days (July and August)
diffusion_solver = 'exact' #'exact': dense Crank-Nicolson as in PEOC 1.0
                           #         (outputs identical to the original)
                           #'fast': the same scheme stepped with a sparse LU,
                           #         ~10x faster, within ~1e-12 relative

#estimation of the parameters on the period initialization_year to
#final_year_of_the_estimation_period
initialization_year = 2007  #first year of infected rachis initialization
final_year_of_the_estimation_period = 2019
nb_years = final_year_of_the_estimation_period-initialization_year 

#Some useful computations (not to modify)
nb_quadrats_x = 62 +nb_fic_x_left+nb_fic_x_right
nb_quadrats_y = 61 + nb_fic_y_bottom + nb_fic_y_top
n_x_RD = nb_quadrats_x * discreti
n_y_RD = nb_quadrats_y * discreti
h = lenght_of_a_quadrat*1.0 / discreti #discretisation step in space
N = int(diffusion_period/delta_t) #number of time steps
