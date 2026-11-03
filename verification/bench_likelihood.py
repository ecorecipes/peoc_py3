# Times likelihood evaluations in a component-wise MH pattern: a start point,
# a new D, then three other single-parameter changes at that D.
# Runs under Python 2.7 (PEOC 1.0) and 3.x (port).
# usage: python bench_likelihood.py <code_dir> [exact|fast]
from __future__ import print_function
import os, sys, shutil, time, tempfile
import numpy as np
sys.dont_write_bytecode = True  # never write .pyc into the code tree

code_dir = os.path.abspath(sys.argv[1])
solver = sys.argv[2] if len(sys.argv) > 2 else None
run_dir = tempfile.mkdtemp()
shutil.copy(os.path.join(code_dir, 'parameters_of_the_model.py'),
            os.path.join(run_dir, 'simulationParameters.py'))
if solver:
    with open(os.path.join(run_dir, 'simulationParameters.py'), 'a') as f:
        f.write("\ndiffusion_solver = '%s'\n" % solver)
os.chdir(code_dir)
sys.path.append(os.path.join(code_dir, 'source_files'))
sys.path.append(run_dir)

t0 = time.time()
import initialisation_rachis as ir
import likelihood
t_import = time.time() - t0

p0 = np.array([1000.0, 0.0, 24.0, 0.0, 18.0, 90.0, 21.50001, 1.0, 0.0085, 0.05])
calls = [('start', p0)]
p = p0.copy(); p[4] = 19.0; calls.append(('new D', p))
for i, v in [(2, 25.0), (5, 95.0), (8, 0.009)]:
    p = p.copy(); p[i] = v; calls.append(('same D', p))
times = []
for label, p in calls:
    t = time.time()
    ll, z = likelihood.log_likelihood(p, ir.infection_1, ir.infection_2)
    times.append(time.time() - t)
    print('%-7s D=%-5g ll=%r  %.2fs' % (label, p[4], ll, times[-1]))
    sys.stdout.flush()
print('import %.1fs; mean per likelihood %.2fs (new D: %.2fs, same D: %.2fs)'
      % (t_import, np.mean(times), np.mean(times[:2]), np.mean(times[2:])))
