"""Compare two directory trees of .npy files: bitwise identity, else max abs/rel diff.
usage: compare_npy.py ref_dir test_dir"""
import sys, os, numpy as np
ref, tst = sys.argv[1], sys.argv[2]
def files(d):
    out = set()
    for root, _, fs in os.walk(d):
        for f in fs:
            if f.endswith('.npy'):
                out.add(os.path.relpath(os.path.join(root, f), d))
    return out
R, T = files(ref), files(tst)
for f in sorted(R ^ T):
    print('ONLY IN', 'ref ' if f in R else 'test', f)
nid = 0
for f in sorted(R & T):
    a = np.load(os.path.join(ref, f), allow_pickle=False)
    b = np.load(os.path.join(tst, f), allow_pickle=False)
    same_bytes = open(os.path.join(ref, f), 'rb').read() == open(os.path.join(tst, f), 'rb').read()
    if same_bytes:
        nid += 1; continue
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        print('DIFF %-40s shape/dtype %s %s vs %s %s' % (f, a.shape, a.dtype, b.shape, b.dtype)); continue
    if a.dtype.kind in 'SU':
        print('DIFF %-40s %r vs %r' % (f, a.tolist(), b.tolist())); continue
    eq = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b)
    d = np.abs(a.astype(float) - b.astype(float))
    rel = d / np.maximum(np.abs(a.astype(float)), 1e-300)
    print('%s %-40s n_diff=%d/%d max_abs=%.3g max_rel=%.3g%s' % (
        'VALEQ' if eq else 'DIFF', f, int(np.sum(a != b)), a.size, np.nanmax(d) if d.size else 0,
        np.nanmax(rel) if rel.size else 0, '' if a.dtype == b.dtype else ' dtype %s/%s' % (a.dtype, b.dtype)))
print('byte-identical files: %d / %d common' % (nid, len(R & T)))
