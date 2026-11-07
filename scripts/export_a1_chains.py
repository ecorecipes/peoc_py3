#!/usr/bin/env python
"""Thin the re-run Appendix A.1 chains for vignette 5 of AshDieback.jl.

usage (from the code folder): .venv/bin/python -I scripts/export_a1_chains.py ../AshDieback.jl/vignettes/05_reproducing_the_paper/results

a1_chains.csv keeps every 100th iteration; a1_best.csv has each chain's best log-likelihood over all iterations.
The chains were run with this port (scripts/appendix_a1_chains.sh), not with the Julia package.
"""
import sys
import numpy as np

NAMES = ['r_allee', 'beta_0', 'beta_1', 'D', 'S_sat', 'gamma', 'C_pers', 'C_init', 'kappa']
FIRST = 1  # column 0 is r_symp, fixed at 1000
STEP = 100


def main(out):
    with open(out + '/a1_chains.csv', 'w') as f, open(out + '/a1_best.csv', 'w') as b:
        f.write('chain,iteration,loglikelihood,' + ','.join(NAMES) + '\n')
        b.write('chain,iterations,best_loglikelihood\n')
        for k in range(1, 11):
            d = 'Simulations/v1_MH_initial_condition_%d/' % k
            s = np.load(d + 'sampled_parameters.npy')
            l = np.load(d + 'log_likelihood.npy').ravel()
            b.write('%d,%d,%.3f\n' % (k, len(l) - 1, l.max()))
            for i in range(0, len(l), STEP):
                f.write('%d,%d,%.3f,' % (k, i, l[i]) + ','.join('%.7g' % v for v in s[i, FIRST:]) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
