import os, subprocess, sys, time
py, work = sys.argv[1], sys.argv[2]
stdin = (sys.argv[3] if len(sys.argv) > 3 else '9').replace(',', '\n') + '\n'
tag = stdin.strip().replace('\n', '_')
env = dict(os.environ, MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
t0 = time.time()
p = subprocess.run([py, 'main_script.py'], cwd=work, input=stdin.encode(), stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT, env=env, timeout=1800)
open(os.path.join(work, 'rerun_%s.txt' % tag), 'wb').write(p.stdout)
print(work, 'exit=%d %.0fs' % (p.returncode, time.time() - t0))
