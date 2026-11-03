# Verification scripts for the Python 3 port of PEOC

These scripts compare `../../peoc` (the original, run with Python 2.7) with `..` (the port). The environments and results are in `../../../wiki/port/environments.md` and `../../../wiki/port/verification.md`.

| script | runs under | purpose |
|---|---|---|
| `dump_model.py <code_dir> <out_dir>` | Python 2.7 **and** 3.x | Imports a PEOC tree and saves every intermediate array: climate, density, observations, binomial constants, initial conditions, Laplacian, 3 log-likelihoods, and the fungus and spore fields. |
| `compare_npy.py <ref_dir> <test_dir>` | Python 3 + numpy | Byte identity of each `.npy`. Otherwise prints the max absolute and relative differences. |
| `e2e.py <python> <code_dir> <work_dir> [full\|mh\|amis\|amis1d] [--continue] [--solver=exact\|fast]` | Python 3 (it drives `<python>`) | Copies a tree, appends identical small test configs, and drives `main_script.py` by stdin. **`full`** runs 1, 2, 8, 4, 5, 3 (resume MH, AMIS, and the shipped Py2 chain `MH_initial_condition_1` by one iteration), 6, 7 and 9. **`mh`** is a ~10 min check of the likelihood path: action 1 with N_MH = 6 (one D proposal, then updates at a cached D) and action 8. **`amis`** runs AMIS with N_L = 8 through 2, 4, 5, 3 and 7. **`amis1d`** is the same with only β₀ free. **`--continue`** reuses a work tree and skips the steps already in its log. **`--solver`** sets `diffusion_solver` in the copied `parameters_of_the_model.py`; PEOC 1.0 ignores it. Stdout goes to `<work_dir>/e2e_stdout.txt`. |
| `compare_solvers.py <peoc_py3_dir>` | Python 3 + scipy | On 7 parameter sets, compares the dense scheme with a DCT closed form and with sparse-LU stepping, for the fungus field (elementwise relative error, sign changes) and the log-likelihood. This is the evidence for the `fast` solver's design and for rejecting the DCT form. |
| `bench_likelihood.py <code_dir> [exact\|fast]` | Python 2.7 **and** 3.x | Times 5 likelihoods in a component-wise MH pattern: a start point, a new D, then three changes at that D. |
| `compare_e2e.py <ref_tree> <test_tree> [--pixels]` | Python 3 + numpy (+ matplotlib for `--pixels`) | Maps timestamped run folders to their roles (`RUN_MH`, `RUN_AMIS`, `RUN_GRID`). Compares every `.npy` and `.txt` byte-for-byte and the set of figures (PNG pixels with `--pixels`), and diffs stdout. |
| `compare_shipped.py <peoc_dir> <e2e_root>` | Python 3 + numpy | For `<e2e_root>/{py27,py37,py3modern}`: checks the extension of the shipped chain over the common prefix. |
| `rerun_action.py <python> <tree> [stdin]` | Python 3 | Re-runs one menu action in a work tree (stdin as comma-separated answers, default `9`). Output goes to `<tree>/rerun_<answers>.txt`. |

Example:

```
python3 e2e.py ENV/py27/bin/python ../../peoc /tmp/ref
python3 e2e.py ENV/py37/bin/python .. /tmp/port
python3 compare_e2e.py /tmp/ref /tmp/port --pixels
```

`e2e.py` sets `MPLBACKEND=Agg`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `PYTHONHASHSEED=0`. A `full` run takes 1–2 h under Rosetta, because each likelihood takes about 1 minute.

Notes:
- Run folders are named by timestamp, so `e2e.py` sleeps 1.1 s between the runs it creates.
- Feed stdin from a file, a pipe opened by `subprocess`, or `< /dev/null`. In some sandboxed shells an interpreter that inherits the shell's stdin can block.
- Actions 6, 7 and 9 create a new numbered output folder each time (`Compare_MH_N`, `Likelihoods_on_regular_grids_N`, …). Delete extra ones before comparing trees.
