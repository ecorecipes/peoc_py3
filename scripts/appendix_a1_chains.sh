#!/bin/bash
# Re-run the paper's ten Appendix A.1 Metropolis-Hastings chains with the
# released model (PEOC 1.0 model code, port's fast solver).
#
# The shipped Simulations/MH_initial_condition_{1..10} were made with an
# earlier, unreleased model and cannot be resumed with this code (see
# lab/wiki/port/reproducing-the-paper.md).  This script starts one new chain
# per shipped configuration: same starting point, prior, proposal scales and
# N_MH (60 000), released model, diffusion_solver = 'fast'.  Chain k uses
# numpy seed k so the chains are independent (main_script.py would give all
# of them seed 1).
#
#   scripts/appendix_a1_chains.sh start [k ...]    start chains (default 1-10)
#   scripts/appendix_a1_chains.sh resume [k ...]   continue after a stop/crash
#   scripts/appendix_a1_chains.sh status           iterations done per chain
#
# Runs go to Simulations/v1_MH_initial_condition_k, logs to logs/.  Each
# chain is one single-threaded process under `nice` and `caffeinate -i`
# (so the Mac does not sleep).  Checkpoints every 50 iterations (save_every);
# `resume` restarts from the last checkpoint.
set -e
cd "$(dirname "$0")/.."
PY=${PEOC_PYTHON:-.venv/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
cmd=${1:-status}; shift || true
chains=${*:-1 2 3 4 5 6 7 8 9 10}
mkdir -p logs

case $cmd in
start)
  for k in $chains; do
    run=Simulations/v1_MH_initial_condition_$k
    [ -e $run ] && { echo "chain $k: $run exists (use resume)"; continue; }
    nohup caffeinate -i nice -n 10 $PY run.py mh --run-dir $run --seed $k \
      --config Simulations/MH_initial_condition_$k/simulationParameters_of_the_AMIS_or_MH_algorithm.py \
      --solver fast --config-set save_every=50 \
      > logs/v1_chain_$k.log 2>&1 < /dev/null &
    echo "chain $k: started, pid $!, log logs/v1_chain_$k.log"
  done ;;
resume)
  for k in $chains; do
    run=Simulations/v1_MH_initial_condition_$k
    [ -f $run/log_likelihood.npy ] || { echo "chain $k: no checkpoint yet"; continue; }
    if pgrep -f "python run.py .*$run( |\$)" > /dev/null; then
      echo "chain $k: still running"; continue
    fi
    nohup caffeinate -i nice -n 10 $PY run.py resume $run \
      >> logs/v1_chain_$k.log 2>&1 < /dev/null &
    echo "chain $k: resumed, pid $!"
  done ;;
status)
  $PY - $chains <<'EOF'
import sys, numpy as np
for k in sys.argv[1:]:
    run = 'Simulations/v1_MH_initial_condition_%s' % k
    try:  # older runs wrote checkpoints in place: a read can race a write
        ll = np.load(run + '/log_likelihood.npy')
        a = np.load(run + '/acceptation_rate.npy')
    except (IOError, OSError, ValueError, EOFError) as e:
        print('chain %2s: no readable checkpoint (%s)' % (k, type(e).__name__))
        continue
    nz = ll[ll != 0]
    print('chain %2s: %6d iterations saved, acceptance %.3f, '
          'last log-lik %.1f, best %.1f' % (k, ll.size - 1, a[-1], ll[-1],
                                            nz.max() if nz.size else 0))
EOF
  echo "running chain processes: $(pgrep -f 'python run.py .*v1_MH_initial_condition' | wc -l | tr -d ' ')" ;;
*)
  echo "usage: $0 start|resume|status [chain numbers]"; exit 2 ;;
esac
