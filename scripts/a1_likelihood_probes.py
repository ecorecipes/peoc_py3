"""Probe the likelihood around the best point of each group of re-run A.1 chains.
usage (from the code folder): .venv/bin/python scripts/a1_likelihood_probes.py
"""
import os, sys, shutil, tempfile
import numpy as np
sys.dont_write_bytecode = True
code = os.getcwd(); run = tempfile.mkdtemp()
shutil.copy('parameters_of_the_model.py', os.path.join(run, 'simulationParameters.py'))
open(os.path.join(run, 'simulationParameters.py'), 'a').write("\ndiffusion_solver = 'fast'\n")
sys.path += [os.path.join(code, 'source_files'), run]
import initialisation_rachis as ir, likelihood
def ll(p): return likelihood.log_likelihood(np.array(p, float), ir.infection_1, ir.infection_2)[0]
N = ['r_symp','r_allee','beta_0','beta_1','D','S_sat','gamma','C_pers','C_init','kappa']
TH = [1000,0,23.26,0,18.51,92.0,21.50001,1,0.00855,0.056]
def best(k):
    d = 'Simulations/v1_MH_initial_condition_%d/' % k
    P, L = np.load(d+'sampled_parameters.npy'), np.load(d+'log_likelihood.npy').ravel()
    return P[L.argmax()].copy(), L.max()
A, la = best(7); B, lb = best(8); C, lc = best(1)
print('theta_max', ll(TH)); print('A best stored', la, 'recomputed', ll(A)); print('B', lb, ll(B)); print('C', lc, ll(C))
print('\nA best, moving one parameter at a time to theta_max:')
p = A.copy(); prev = ll(p)
for i in (1, 3, 7, 6, 2, 4, 5, 8, 9):
    p[i] = TH[i]; v = ll(p); print('  %-8s -> %-9g ll %10.3f  change %+8.3f' % (N[i], TH[i], v, v-prev)); prev = v
print('\nstraight line from A best to B best:')
for t in np.linspace(0, 1, 11): print('  %.1f %10.1f' % (t, ll((1-t)*A+t*B)))
print('\nprofile in gamma, other parameters fixed:')
for g in (21.3, 21.45, 21.4999, 21.50001, 21.51, 21.6, 21.8, 22.0, 22.4, 22.7, 22.95, 22.999, 23.001, 23.2, 24):
    a = A.copy(); a[6] = g; b = B.copy(); b[6] = g
    print('  gamma %8.5f   A %10.1f   B %10.1f' % (g, ll(a), ll(b)))
