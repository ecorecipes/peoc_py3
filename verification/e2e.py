"""End-to-end driver: copies a PEOC code tree, applies identical small test
configurations, and drives main_script.py through actions 1-9 via stdin.
Run with any Python 3; the PEOC tree runs under the given interpreter.

usage: e2e.py <python_exe> <code_dir> <work_dir> [full|mh|amis|amis1d]
              [--continue] [--solver=exact|fast]
  full:   actions 1, 2, 8, 4, 5, 3 (incl. the shipped MH_initial_condition_1),
          6, 7, 9 (default)
  mh:     quick check of the likelihood path: action 1 with N_MH = 6 (one D
          proposal, then updates at a cached D) and action 8
  amis:   AMIS-only sequence with N_L=8 (> 5 free parameters, so the
          weighted proposal covariance can be full rank)
  amis1d: same sequence with only beta_0 free (N_L=4); weights collapse onto
          one sample at these sample sizes, which makes any multi-parameter
          proposal covariance singular
  --continue: reuse an existing work tree and skip steps already in its log
  --solver:   set diffusion_solver in the copied parameters_of_the_model.py
              (peoc_py3 only; PEOC 1.0 ignores it); applied on the first pass
"""
import os, re, shutil, subprocess, sys, time

py, code_dir, work = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
ARGS = [a for a in sys.argv[4:] if not a.startswith('--')]
MODE = ARGS[0] if ARGS else 'full'
CONT = '--continue' in sys.argv
DONE = set()
if CONT:
    DONE = set(re.findall(r'^===== (.*?) \| stdin', open(os.path.join(work, 'e2e_stdout.txt')).read(), re.M))
else:
    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(code_dir, work, ignore=shutil.ignore_patterns(
        '*.pyc', '__pycache__', '.git', 'verification'))
SIM = os.path.join(work, 'Simulations')
LOG = open(os.path.join(work, 'e2e_stdout.txt'), 'a' if CONT else 'w')
ENV = dict(os.environ, MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')

def append(path, text):
    if CONT and os.path.basename(path).startswith(('1_or_2_', '8_')):
        return  # configuration already applied in the first pass
    with open(path, 'a') as f:
        f.write('\n# ---- e2e test overrides ----\n' + text)

def replace_line(path, pattern, new):
    src = open(path).read()
    src2, n = re.subn(pattern, new, src, flags=re.M)
    assert n == 1, (path, pattern, n)
    open(path, 'w').write(src2)

# ---- identical test configuration (valid in Python 2 and 3) -----------------
ALGO = os.path.join(work, '1_or_2_parameters_of_the_AMIS_or_MH_algorithm.py')
append(ALGO, '''
N_MH = 4
N_AMIS = 2
N_L = %d
num_workers = 4
# reduced model of the paper (eq. 13) and AMIS initialisation of Table 2
def initial_parameters():
    return np.array([1000.0, 0.0, 24.0, 0.0, 18.0, 90.0, 21.50001, 1.0, 0.0085, 0.05])
parameters_ranges = np.array([[1000.0, 1000.0], [0.0, 0.0], [0.0, 800.0], [0.0, 0.0],
                              [0.0, 200.0], [0.0, 500.0], [21.50001, 21.50001],
                              [1.0, 1.0], [0.0, 200.0], [0.0, 10.0]])
scale_para_AMIS = np.array([0., 0., 3., 0., 0.4, 1., 0., 0., 1.0e-8, 1.5e-5])
''' % (8 if MODE == 'amis' else 4))
if MODE == 'amis1d':
    append(ALGO, '''
parameters_ranges = np.array([[1000.0, 1000.0], [0.0, 0.0], [0.0, 800.0], [0.0, 0.0],
                              [18.0, 18.0], [90.0, 90.0], [21.50001, 21.50001],
                              [1.0, 1.0], [0.0085, 0.0085], [0.05, 0.05]])
''')
SOLVER = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--solver=')]
if SOLVER and not CONT:  # peoc_py3 only; PEOC 1.0 ignores the variable
    with open(os.path.join(work, 'parameters_of_the_model.py'), 'a') as f:
        f.write("\ndiffusion_solver = '%s'  # e2e\n" % SOLVER[0])
GRID = os.path.join(work, '8_parameters_for_computation_on_regular_grid.py')
append(GRID, '''
v_r_symp = np.linspace(1000, 1000, 1)
v_beta_1 = np.linspace(0.0, 0.0, 1)
v_S_sat = np.linspace(90., 90., 1)
v_C_recover = np.linspace(1., 1., 1)
v_C_init = np.linspace(0.0085, 0.0085, 1)
v_gamma = np.linspace(21.50001, 21.50001, 1)
v_beta_0 = np.linspace(23., 25., 2)
v_D = np.linspace(17., 19., 2)
v_kappa = np.linspace(0.05, 0.05, 1)
v_r_allee = np.linspace(0, 0.0, 1)
num_workers = 4
''')

def sims():
    return sorted(os.listdir(SIM))

def run(label, stdin):
    if label in DONE:
        print('%-28s (already done)' % label, flush=True)
        return
    t0 = time.time()
    LOG.write('\n===== %s | stdin=%r\n' % (label, stdin))
    LOG.flush()
    p = subprocess.run([py, 'main_script.py'], cwd=work, input=stdin.encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=ENV)
    out = p.stdout.decode('utf-8', 'replace')
    LOG.write(out + '\n----- exit=%d\n' % p.returncode)
    LOG.flush()
    print('%-28s exit=%d  %6.0fs' % (label, p.returncode, time.time() - t0), flush=True)

def newest(suffix):
    c = [s for s in sims() if s.endswith(suffix) and s[0].isdigit()]
    assert len(c) == 1, (suffix, c)
    return c[0]

def idx(name):
    return sims().index(name) + 1

def set_iters(folder, var, value):
    # the run folder's copy already holds the original value and the test
    # override; a further appended assignment takes precedence
    with open(os.path.join(SIM, folder, 'simulationParameters_of_the_AMIS_or_MH_algorithm.py'), 'a') as f:
        f.write('\n%s = %d  # e2e: extend run\n' % (var, value))

if MODE == 'mh':
    # quick check of the likelihood path: one D proposal (update 2 of 5) then
    # four updates at a cached D, plus the regular grid over two D values
    append(ALGO, 'N_MH = 6\n')
    run('1 new MH', '1\n'); mh = newest('_MH'); time.sleep(1.1)
    run('8 regular grid', '8\n'); gr = newest('_regular_grid')
    open(os.path.join(work, 'e2e_roles.txt'), 'w').write('%s - %s\n' % (mh, gr))
    sys.exit(0)

if MODE in ('amis', 'amis1d'):
    run('2 new AMIS', '2\n'); am = newest('_AMIS')
    run('4 graphs AMIS', '4\n%d\n' % idx(am))
    run('5 RMSE AMIS', '5\n%d\n' % idx(am))
    set_iters(am, 'N_AMIS', 3); run('3 resume AMIS', '3\n%d\n' % idx(am))
    append(os.path.join(work, '7_simulations_AMIS_to_compare.py'),
           "simu = ['%s', '%s']\nl = ['run A', 'run B']\n" % (am, am))
    run('7 compare AMIS', '7\n')
    open(os.path.join(work, 'e2e_roles.txt'), 'w').write('- %s -\n' % am)
    sys.exit(0)

run('1 new MH', '1\n');                 mh = newest('_MH');    time.sleep(1.1)
run('2 new AMIS', '2\n');               am = newest('_AMIS');  time.sleep(1.1)
run('8 regular grid', '8\n');           gr = newest('_regular_grid')
run('4 graphs MH', '4\n%d\n' % idx(mh))
run('4 graphs AMIS', '4\n%d\n' % idx(am))
run('5 RMSE MH', '5\n%d\n' % idx(mh))
run('5 RMSE AMIS', '5\n%d\n' % idx(am))
set_iters(mh, 'N_MH', 6);   run('3 resume MH', '3\n%d\n' % idx(mh))
set_iters(am, 'N_AMIS', 3); run('3 resume AMIS', '3\n%d\n' % idx(am))
# The shipped chain was saved after 56063 iterations although its config says
# N_MH = 60000, so extend by exactly one iteration from the saved length
# (MH runs while k < N_MH, starting from k = length - 1).
def npy_len(path):
    with open(path, 'rb') as f:
        f.read(8)
        h = f.read(int.from_bytes(f.read(2), 'little')).decode('latin1')
    return int(re.search(r"'shape': \((\d+)", h).group(1))
n_ship = npy_len(os.path.join(SIM, 'MH_initial_condition_1', 'log_likelihood.npy'))
set_iters('MH_initial_condition_1', 'N_MH', n_ship)
run('3 resume shipped Py2 run', '3\n%d\n' % idx('MH_initial_condition_1'))
run('6 compare MH (default)', '6\n')
append(os.path.join(work, '7_simulations_AMIS_to_compare.py'),
       "simu = ['%s', '%s']\nl = ['run A', 'run B']\n" % (am, am))
run('7 compare AMIS', '7\n')
append(os.path.join(work, '9_graph_log_likelihood_on_regular_grid.py'), "simus = ['%s']\n" % gr)
run('9 plot regular grid', '9\n')
print('roles:', {'MH': mh, 'AMIS': am, 'GRID': gr})
open(os.path.join(work, 'e2e_roles.txt'), 'w').write('%s %s %s\n' % (mh, am, gr))
