# -*- coding:Utf-8 -*-
"""Helpers reproducing Python 2 behaviours that leak into PEOC's outputs.

The original PEOC ran under Python 2.7. A few Python 2 semantics are visible in
the files and text it produces; this module reproduces them so that the
Python 3 port gives exactly the same outputs (and can resume simulations that
were started with the Python 2 version, and vice versa).

- py2_str: str() of a Python float in Python 2 (12 significant digits).
- py2_round: round() of Python 2 (round half away from zero, returns a float).
- save_random_state / load_random_state: the format of seed.npy/seed2.npy.
  Python 2 + numpy 1.16 saved the list [name, pos, has_gauss, cached_gaussian]
  as a byte-string array ('|S...') built with Python 2 str(), so the cached
  Gaussian is truncated to 12 significant digits. Resuming a run therefore
  restarts from this truncated state; we reproduce it exactly.
"""
import math
import numpy as np


def py2_str(x):
    """Return str(x) as Python 2.7 would print a Python float or int."""
    if isinstance(x, float) and type(x) is float:
        s = '%.12g' % x
        if not any(c in s for c in '.eEn'):  # 'n' covers 'nan' and 'inf'
            s += '.0'
        return s
    return str(x)


def py2_round(x):
    """round(x) of Python 2: half away from zero, returned as a float."""
    x = float(x)
    if x >= 0.0:
        return float(math.floor(x + 0.5))
    return float(math.ceil(x - 0.5))


def save_random_state(folder):
    """Save the state of numpy's global RandomState as PEOC (Python 2) did:
    seed.npy  = byte-string array [name, pos, has_gauss, cached_gaussian]
    seed2.npy = the uint32 key array of the Mersenne Twister."""
    randomstate = np.random.get_state()
    seed = [randomstate[0], randomstate[2], randomstate[3], randomstate[4]]
    np.save(folder + '/seed.npy',
            np.array([py2_str(v).encode('ascii') for v in seed]))
    np.save(folder + '/seed2.npy', randomstate[1])


def _as_text(v):
    if isinstance(v, bytes):
        return v.decode('ascii')
    return str(v)


def load_random_state(folder, key_dtype=None):
    """Restore numpy's global RandomState from seed.npy/seed2.npy (files
    written either by the Python 2 version or by this port)."""
    seed = np.load(folder + '/seed.npy')
    seed2 = np.load(folder + '/seed2.npy')
    if key_dtype is not None:
        seed2 = np.array(seed2, dtype=key_dtype)
    np.random.set_state((_as_text(seed[0]), seed2, int(_as_text(seed[1])),
                         int(_as_text(seed[2])), float(_as_text(seed[3]))))
