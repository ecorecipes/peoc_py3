# Runs under Python 2.7 and 3.x. Imports a PEOC code tree and dumps every
# intermediate model quantity plus log-likelihoods at fixed parameter sets.
# usage: python dump_model.py <code_dir> <out_dir>
from __future__ import print_function
import os, sys, shutil, time, tempfile
import numpy as np
# Never write .pyc into the code tree: the reference tree's shipped bytecode
# (e.g. Simulations/*/simulationParameters.pyc) is provenance evidence.
sys.dont_write_bytecode = True

code_dir = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

run_dir = tempfile.mkdtemp()
shutil.copy(os.path.join(code_dir, 'parameters_of_the_model.py'),
            os.path.join(run_dir, 'simulationParameters.py'))
shutil.copy(os.path.join(code_dir,
                         '1_or_2_parameters_of_the_AMIS_or_MH_algorithm.py'),
            os.path.join(run_dir,
                         'simulationParameters_of_the_AMIS_or_MH_algorithm.py'))
os.chdir(code_dir)
sys.path.append(os.path.join(code_dir, 'source_files'))
sys.path.append(run_dir)

import environmental_data as env
import observations
import initialisation_rachis as ir
import preliminary_computations_binomial_law as pcbl
import reaction_diffusion as RD
import likelihood

def save(name, a):
    np.save(os.path.join(out_dir, name + '.npy'), np.asarray(a))

save('H', env.H); save('T', env.T); save('ash_density', env.ash_density)
save('obs_y', observations.y)
for k, a in enumerate(observations.observation):
    save('observation_%02d' % k, a)
for k, a in enumerate(pcbl.clb_likeli):
    save('clb_likeli_%02d' % k, a)
save('infection_1', ir.infection_1); save('infection_2', ir.infection_2)
if RD.M_lap is not None:  # None with the port's diffusion_solver = 'fast'
    save('M_lap', RD.M_lap)

PARAS = [
    [1000.0, 15.0, 0.001, 0.0001, 30.0, 60.0, 40.0, 0.01, 0.002, 0.01],
    [1000.0, 0.0, 30.0, 0.0, 20.0, 90.0, 22.0, 1.0, 0.0085, 0.1],
    [1000.0, 50.0, 40.0, 2.0, 25.0, 80.0, 30.0, 0.5, 0.01, 0.5],
]
res = []
for i, p in enumerate(PARAS):
    t0 = time.time()
    ll, z = likelihood.log_likelihood(np.array(p), ir.infection_1,
                                      ir.infection_2)
    print('para %d: loglik=%r terme0=%r  (%.1fs)' % (i, ll, z,
                                                       time.time() - t0))
    res.append([ll, float(z)])
save('loglik', np.array(res))

# fungi fields for parameter set 2, including the spore output used in plots
p = PARAS[2]
(r_symp, r_allee, beta_0, beta_1, D, S_sat, gamma, C_nr, C_init, kappa) = p
para = __import__('simulationParameters')
it = para.initialization_year - para.first_year_of_data
f1 = para.f_temperature(env.T[it], gamma, kappa); f1[f1 == 0] = 1.0
f2 = para.f_temperature(env.T[it + 1], gamma, kappa); f2[f2 == 0] = 1.0
rf1 = C_init * r_symp * ir.infection_1 / f1
rf2 = C_init * r_symp * ((ir.infection_2 - C_nr * ir.infection_1) /
                         ((1 - C_nr * ir.infection_1 * C_init) * f2))
fungi, spores = RD.reaction_di(r_allee, beta_0, beta_1, D, S_sat, rf1, rf2,
                               False, True)
save('fungi', fungi); save('spores_unsat', spores)
shutil.rmtree(run_dir)
print('done')
