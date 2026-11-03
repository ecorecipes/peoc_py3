"""Compare the common prefix of the extended shipped chain MH_initial_condition_1.

Usage: compare_shipped.py <original_peoc_dir> <e2e_root>
<e2e_root> holds py27/, py37/, py3modern/ work trees produced by e2e.py.
"""
import sys
import numpy as np

orig, root = sys.argv[1], sys.argv[2]
base = orig + '/Simulations/MH_initial_condition_1/'
envs = ['py27', 'py37', 'py3modern']
d = {e: '%s/%s/Simulations/MH_initial_condition_1/' % (root, e) for e in envs}
L0 = np.load(base + 'log_likelihood.npy').shape[0]
print('shipped chain length', L0)
for f in ['sampled_parameters', 'log_likelihood', 'acceptation_rate']:
    a = {e: np.load(d[e] + f + '.npy') for e in envs}
    n = min(len(v) for v in a.values())
    print(f, {e: len(v) for e, v in a.items()}, 'common', n, 'new iterations', n - L0)
    o = np.load(base + f + '.npy')
    print('  shipped prefix unchanged in all:',
          all(np.array_equal(v[:L0], o) for v in a.values()))
    print('  py37 == py27 (bytes):', a['py37'][:n].tobytes() == a['py27'][:n].tobytes())
    m, r = a['py3modern'][:n], a['py27'][:n]
    rel = np.abs(m - r) / np.maximum(np.abs(r), 1e-300)
    print('  py3modern == py27 exactly:', np.array_equal(m, r), ' max rel diff %.3g' % rel.max())
p = np.load(d['py27'] + 'sampled_parameters.npy')[L0 - 1:]
print('distinct parameter sets in extension (incl. start):', len({tuple(x) for x in p}))
