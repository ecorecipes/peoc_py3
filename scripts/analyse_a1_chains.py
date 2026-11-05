#!/usr/bin/env python
"""Summarise the re-run Appendix A.1 chains: levels, convergence, parameters, theta_max.
usage (from the code folder): .venv/bin/python scripts/analyse_a1_chains.py [burn_in]
"""
import sys
import numpy as np

NAMES = ['r_symp', 'r_allee', 'beta_0', 'beta_1', 'D', 'S_sat', 'gamma', 'C_pers', 'C_init', 'kappa']
THETA_MAX = [1000, 0, 23.26, 0, 18.51, 92.0, 21.50001, 1, 0.00855, 0.056]
LL_THETA_MAX = -24717.2
K = 10
burn = int(sys.argv[1]) if len(sys.argv) > 1 else 30000


def load(k):
    d = 'Simulations/v1_MH_initial_condition_%d/' % k
    return np.load(d + 'sampled_parameters.npy'), np.load(d + 'log_likelihood.npy').ravel()


def split_rhat(x):
    """Split R-hat over chains in the rows of x (chains x draws)."""
    n = x.shape[1] // 2
    s = np.vstack([x[:, :n], x[:, n:2 * n]])
    w = s.var(axis=1, ddof=1).mean()
    b = n * s.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * w + b / n) / w) if w > 0 else np.nan


def ess(x):
    """Effective sample size of one chain, Geyer's initial positive sequence."""
    x = x - x.mean()
    if x.var() == 0:
        return np.nan
    f = np.fft.rfft(x, 2 * len(x))
    r = np.fft.irfft(f * np.conj(f))[:len(x)]
    r = r / r[0]
    t = 1.0
    for i in range(1, len(r) - 1, 2):
        p = r[i] + r[i + 1]
        if p < 0:
            break
        t += 2 * p
    return len(x) / t


P, L = zip(*[load(k) for k in range(1, K + 1)])
P, L = np.array(P), np.array(L)
print('shapes', P.shape, L.shape, 'burn-in', burn)
print('finite ll: %d of %d' % (np.isfinite(L).sum(), L.size))

print('\nstart of each chain, and iteration of first reaching within 50 of its final level')
for k in range(K):
    final = np.median(L[k, burn:])
    first = int(np.argmax(L[k] > final - 50))
    print('chain %2d  ll0 %11.1f  median %10.1f  best %10.1f  reached at %6d  '
          % (k + 1, L[k, 0], final, L[k].max(), first)
          + ' '.join('%.3g' % v for v in P[k, 0]))

print('\npost-burn-in medians by chain')
print('chain    ll       ' + ' '.join('%10s' % n for n in NAMES))
order = np.argsort(-np.median(L[:, burn:], axis=1))
for k in order:
    print('%5d %9.1f ' % (k + 1, np.median(L[k, burn:]))
          + ' '.join('%10.4g' % v for v in np.median(P[k, burn:], axis=0)))
print('theta_max %9.1f ' % LL_THETA_MAX + ' '.join('%10.4g' % v for v in THETA_MAX))

levels = np.median(L[:, burn:], axis=1)
groups = [np.where(np.abs(levels - c) < 100)[0] for c in sorted({round(v, -2) for v in levels}, reverse=True)]
print('\ngroups', [[int(i) + 1 for i in g] for g in groups])
print('\nsplit R-hat and pooled summaries after burn-in')
print('%-8s %8s | ' % ('', 'all') + ' | '.join('group %d: rhat  ess       q2.5        q50      q97.5' % (i + 1) for i in range(len(groups))))
for j, n in enumerate(NAMES + ['ll']):
    get = (lambda idx: L[idx, burn:]) if n == 'll' else (lambda idx: P[idx, burn:, j])
    x = get(np.arange(K))
    row = '%-8s %8.3f | ' % (n, split_rhat(x)) if x.std() > 0 else '%-8s %8s | ' % (n, 'fixed')
    cells = []
    for g in groups:
        y = get(g)
        rh = split_rhat(y) if len(g) > 1 and y.std() > 0 else np.nan
        e = np.nansum([ess(c) for c in y])
        q = np.percentile(y, [2.5, 50, 97.5])
        cells.append('      %6.3f %5.0f %10.5g %10.5g %10.5g' % (rh, e, q[0], q[1], q[2]))
    print(row + ' | '.join(cells))

print('\nrange of each parameter in each chain after burn-in (min, max)')
for j, n in enumerate(NAMES):
    if P[:, burn:, j].std() == 0:
        continue
    print('%-8s ' % n + ' '.join('%d:[%.4g,%.4g]' % (k + 1, P[k, burn:, j].min(), P[k, burn:, j].max()) for k in range(K)))

print('\nbest point of each group')
for g in groups:
    k = g[np.argmax(L[g].max(axis=1))]
    i = L[k].argmax()
    print('chain %2d iter %6d ll %10.2f  ' % (k + 1, i, L[k, i]) + ' '.join('%s=%.5g' % (n, v) for n, v in zip(NAMES, P[k, i])))
