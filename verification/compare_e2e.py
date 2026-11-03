"""Compare two e2e work trees produced by e2e.py.
usage: compare_e2e.py <ref_tree> <test_tree> [--pixels]
- .npy/.txt: byte identity, else numeric diff summary
- figures (.png): same set of files; with --pixels also decoded-pixel identity
- .pdf: same set of files (contents embed dates/versions)
- stdout: diff after replacing timestamped folder names
"""
import os, re, sys, difflib
import numpy as np

ref, tst = sys.argv[1], sys.argv[2]
PIX = '--pixels' in sys.argv

def roles(tree):
    names = open(os.path.join(tree, 'e2e_roles.txt')).read().split()
    return {n: r for n, r in zip(names, ['RUN_MH', 'RUN_AMIS', 'RUN_GRID']) if n != '-'}

def collect(tree):
    rm = roles(tree)
    out = {}
    sim = os.path.join(tree, 'Simulations')
    for root, dirs, files in os.walk(sim):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith(('.pyc',)):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, sim)
            top = rel.split(os.sep)[0]
            rel = os.sep.join([rm.get(top, top)] + rel.split(os.sep)[1:])
            out[rel] = full
    return out, rm

R, rmR = collect(ref)
T, rmT = collect(tst)
skip_ext = ('.py',)  # parameter-file copies: py2 vs py3 syntax by construction
keys_R = {k for k in R if not k.endswith(skip_ext)}
keys_T = {k for k in T if not k.endswith(skip_ext)}
only = sorted(keys_R ^ keys_T)
print('files compared: %d common; only-in-one: %d' % (len(keys_R & keys_T), len(only)))
for k in only[:30]:
    print('  ONLY IN %s: %s' % ('ref ' if k in keys_R else 'test', k))

stats = {'npy_same': 0, 'npy_diff': [], 'txt_same': 0, 'txt_diff': [], 'png_same': 0,
         'png_diff': [], 'png_unchecked': 0, 'pdf': 0}
for k in sorted(keys_R & keys_T):
    a, b = R[k], T[k]
    if k.endswith('.npy'):
        if open(a, 'rb').read() == open(b, 'rb').read():
            stats['npy_same'] += 1
        else:
            x, y = np.load(a), np.load(b)
            if x.dtype.kind in 'SU' or x.shape != y.shape:
                stats['npy_diff'].append('%s: %r vs %r' % (k, x.tolist() if x.size < 8 else x.shape,
                                                           y.tolist() if y.size < 8 else y.shape))
            else:
                d = np.abs(x.astype(float) - y.astype(float))
                rel = d / np.maximum(np.abs(x.astype(float)), 1e-300)
                stats['npy_diff'].append('%s: n_diff=%d/%d max_rel=%.2e' % (
                    k, int(np.sum(~((x == y) | (np.isnan(x) & np.isnan(y))))), x.size,
                    np.nanmax(rel)))
    elif k.endswith('.txt'):
        if open(a, 'rb').read() == open(b, 'rb').read():
            stats['txt_same'] += 1
        else:
            stats['txt_diff'].append(k)
    elif k.endswith('.png'):
        if not PIX:
            stats['png_unchecked'] += 1
            continue
        import matplotlib.image as mpimg
        ia, ib = mpimg.imread(a), mpimg.imread(b)
        if ia.shape == ib.shape and np.array_equal(ia, ib):
            stats['png_same'] += 1
        else:
            frac = (np.mean(np.any(ia != ib, axis=-1)) if ia.shape == ib.shape else float('nan'))
            stats['png_diff'].append('%s: shape %s vs %s, differing pixels %.4f' % (
                k, ia.shape, ib.shape, frac))
    elif k.endswith('.pdf'):
        stats['pdf'] += 1

print('npy: %d byte-identical, %d differ' % (stats['npy_same'], len(stats['npy_diff'])))
for s in stats['npy_diff']:
    print('  ' + s)
print('txt: %d byte-identical, %d differ %s' % (stats['txt_same'], len(stats['txt_diff']), stats['txt_diff'][:5]))
if PIX:
    print('png: %d pixel-identical, %d differ' % (stats['png_same'], len(stats['png_diff'])))
    for s in stats['png_diff'][:15]:
        print('  ' + s)
else:
    print('png: %d present in both (pixels not compared)' % stats['png_unchecked'])
print('pdf: %d present in both' % stats['pdf'])

def norm_stdout(tree, rm):
    s = open(os.path.join(tree, 'e2e_stdout.txt')).read()
    for n, r in rm.items():
        s = s.replace(n, r)
    s = re.sub(r'(Compare_MH_|Compare_posterior_|Likelihoods_on_regular_grids_)\d+', r'\1N', s)
    s = s.replace(os.path.abspath(tree), '<TREE>')
    return s.splitlines()

a, b = norm_stdout(ref, rmR), norm_stdout(tst, rmT)
diff = [l for l in difflib.unified_diff(a, b, 'ref', 'test', n=0, lineterm='')
        if not l.startswith(('---', '+++', '@@'))]
print('stdout: %d lines ref, %d lines test, %d differing lines' % (len(a), len(b), len(diff)))
for l in diff[:60]:
    print('  ' + l[:220])
