# -*- coding:Utf-8 -*-
"""Non-interactive runner for PEOC (port only).

main_script.py asks for the action and the simulation on stdin.  This script
does the same work from command-line arguments, so runs can be scripted,
launched in parallel or submitted to a cluster.  It follows main_script.py
step by step (folder creation, copies of the input files, random seed, the
module calls), so the outputs of an action are the same as with the menu.

Examples
--------
  python3 run.py mh                                   # action 1, as the menu
  python3 run.py mh --solver fast --run-dir Simulations/chain_A \\
      --config Simulations/MH_initial_condition_2/simulationParameters_of_the_AMIS_or_MH_algorithm.py \\
      --config-set N_MH=60000 --config-set save_every=50
  python3 run.py resume chain_A --config-set N_MH=80000   # action 3
  python3 run.py plots chain_A                            # action 4
  python3 run.py compare-mh --config-set "simu=['chain_A','chain_B']"

Overrides
---------
--set NAME=EXPR and --config-set NAME=EXPR replace the top-level statement that
defines NAME (an assignment, or a function such as initial_parameters; the
last one if NAME is defined several times) in the
copy of the model file or of the action's input file.  EXPR is Python source,
evaluated when the copy is imported.  Because the statement is replaced where
it stands, values derived from it later in the file follow the new value.
A name the file does not define (e.g. save_every) is appended at the end.
For actions 3, 4 and 5 the copies are the ones inside the existing run folder
and they are edited in place, as the README asks for action 3.
"""

import argparse
import ast
import os
import shutil
import sys
import tempfile
import time

ACTIONS = {
    '1': '1', 'mh': '1',
    '2': '2', 'amis': '2',
    '3': '3', 'resume': '3',
    '4': '4', 'plots': '4',
    '5': '5', 'rmse': '5',
    '6': '6', 'compare-mh': '6',
    '7': '7', 'compare-amis': '7',
    '8': '8', 'grid': '8',
    '9': '9', 'plot-grid': '9',
}

MODEL_FILE = 'parameters_of_the_model.py'
ALGO_FILE = '1_or_2_parameters_of_the_AMIS_or_MH_algorithm.py'
INPUT_FILES = {
    '6': '6_simulations_MH_to_compare.py',
    '7': '7_simulations_AMIS_to_compare.py',
    '8': '8_parameters_for_computation_on_regular_grid.py',
    '9': '9_graph_log_likelihood_on_regular_grid.py',
}
RUN_MODEL_COPY = 'simulationParameters.py'
RUN_ALGO_COPY = 'simulationParameters_of_the_AMIS_or_MH_algorithm.py'
RUN_GRID_COPY = 'parameters_for_computation_on_regular_grid.py'
CODE_DIR = os.path.dirname(os.path.abspath(__file__))


class UsageError(Exception):
    pass


# ---------------------------------------------------------------- overrides

def _statement_ranges(tree, n_lines):
    """(node, first_line, last_line) of each top-level statement, 1-based.

    Python < 3.8 has no end_lineno; a statement then runs up to the next one,
    so trailing comments and blank lines are replaced with it (harmless)."""
    body = tree.body
    ranges = []
    for i, node in enumerate(body):
        start = node.lineno
        if getattr(node, 'decorator_list', None):
            start = min(d.lineno for d in node.decorator_list)
        end = getattr(node, 'end_lineno', None)
        if end is None:
            end = body[i + 1].lineno - 1 if i + 1 < len(body) else n_lines
        ranges.append((node, start, end))
    return ranges


def _defines(node, name):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name
                   for t in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    return False


def apply_overrides(path, overrides):
    """Replace the top-level definitions named in `overrides` in file `path`.

    `overrides` is a list of (name, expression).  If the name is defined
    more than once, the last definition (the one in effect) is replaced; a
    name not defined at all is appended at the end.  The result must
    compile."""
    if not overrides:
        return
    with open(path) as f:
        text = f.read()
    for name, expr in overrides:
        if name == 'initial_parameters':
            new = ('def initial_parameters():  # run.py override\n'
                   '    return np.array(%s, dtype=float)\n' % expr)
        else:
            new = '%s = %s  # run.py override\n' % (name, expr)
        lines = text.splitlines(True)
        tree = ast.parse(text, path)
        hits = [(s, e) for node, s, e in
                _statement_ranges(tree, len(lines)) if _defines(node, name)]
        if hits:
            # the last top-level definition is the one in effect
            start, end = hits[-1]
            lines[start - 1:end] = [new]
        else:
            if lines and not lines[-1].endswith('\n'):
                lines[-1] += '\n'
            lines.append(new)
        text = ''.join(lines)
        print('run.py: %s: %s' % (path, new.rstrip()))
    try:
        compile(text, path, 'exec', dont_inherit=True)
    except SyntaxError as e:
        raise UsageError('%s: override gives invalid Python: %s' % (path, e))
    with open(path, 'w') as f:
        f.write(text)


def parse_override(text):
    name, sep, expr = text.partition('=')
    name = name.strip()
    if not sep or not name.isidentifier() or not expr.strip():
        raise UsageError('override %r is not NAME=EXPR' % text)
    return name, expr.strip()


# ---------------------------------------------------------------- arguments

def build_parser():
    p = argparse.ArgumentParser(
        prog='run.py',
        description='Run a PEOC action without the interactive menu.',
        epilog='Examples' + __doc__.split('Examples', 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', help='1-9 or mh, amis, resume, plots, rmse, '
                   'compare-mh, compare-amis, grid, plot-grid')
    p.add_argument('simulation', nargs='?',
                   help='actions 3, 4, 5: the run folder (a path, or a name '
                   'under Simulations/)')
    p.add_argument('--run-dir',
                   help='actions 1, 2, 8: folder to create (default '
                   'Simulations/<yymmddHHMMSS>_<MH|AMIS|regular_grid>); '
                   'it must not exist')
    p.add_argument('--model', help='actions 1, 2, 8: model parameter file '
                   '(default %s)' % MODEL_FILE)
    p.add_argument('--config', help="the action's input file (actions 1, 2: "
                   "default %s; 6-9: the matching top-level file)" % ALGO_FILE)
    p.add_argument('--set', dest='model_set', action='append', default=[],
                   metavar='NAME=EXPR', help='override a model parameter')
    p.add_argument('--config-set', action='append', default=[],
                   metavar='NAME=EXPR', help="override a setting of the "
                   "action's input file (for 3, 4, 5: the run's algorithm "
                   "file); initial_parameters=[...] sets the starting point")
    p.add_argument('--solver', choices=['exact', 'fast'],
                   help='shorthand for --set diffusion_solver=...')
    p.add_argument('--seed', type=int, default=1,
                   help='numpy random seed for new runs (default 1, as '
                   'main_script.py); resumed runs restore their saved state')
    p.add_argument('--plots', action='store_true',
                   help='actions 1, 2, 3: also make the action 4 graphs at '
                   'the end')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except UsageError as e:
        sys.stderr.write('run.py: error: %s\n' % e)
        return 2


def run(args):
    action = ACTIONS.get(args.action.lower())
    if action is None:
        raise UsageError('unknown action %r' % args.action)

    model_set = [parse_override(s) for s in args.model_set]
    if args.solver:
        model_set.append(('diffusion_solver', repr(args.solver)))
    config_set = [parse_override(s) for s in args.config_set]

    # Paths on the command line are relative to the caller's directory, but
    # PEOC loads its data by relative path and must run from the code folder.
    model_src = args.model and os.path.abspath(args.model)
    config_src = args.config and os.path.abspath(args.config)
    run_dir = args.run_dir and os.path.relpath(os.path.abspath(args.run_dir),
                                               CODE_DIR)
    simulation = None
    if args.simulation is not None:
        cand = os.path.abspath(args.simulation)
        if not os.path.isdir(cand):
            cand = os.path.join(CODE_DIR, 'Simulations', args.simulation)
        simulation = os.path.relpath(cand, CODE_DIR)
    os.chdir(CODE_DIR)

    # Check the arguments against the action before doing anything.
    if action in ('3', '4', '5'):
        if simulation is None:
            raise UsageError('action %s needs a simulation folder' % action)
        if not os.path.isdir(simulation):
            raise UsageError('no simulation folder %s' % args.simulation)
        if args.model or args.config:
            raise UsageError('--model/--config are for new runs; use '
                             '--set/--config-set to edit the run copies')
    elif simulation is not None:
        raise UsageError('action %s takes no simulation argument' % action)
    if run_dir is not None and action not in ('1', '2', '8'):
        raise UsageError('--run-dir is for actions 1, 2 and 8')
    if action in ('6', '7', '9') and (model_set or args.model):
        raise UsageError('action %s has no model parameters' % action)
    if args.plots and action not in ('1', '2', '3'):
        raise UsageError('--plots is for actions 1, 2 and 3')
    for src in (model_src, config_src):
        if src and not os.path.isfile(src):
            raise UsageError('no such file %s' % src)

    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    np.random.seed(args.seed)
    sys.path.append(os.path.join(os.getcwd(), 'source_files'))

    # ---- the run folder and its copies of the input files
    dossier = None
    if action in ('1', '2', '8'):
        if run_dir is None:
            if not os.path.exists('Simulations'):
                os.mkdir('Simulations')
            run_dir = time.strftime('Simulations/%y%m%d%H%M%S',
                                    time.localtime())
            run_dir += {'1': '_MH', '2': '_AMIS', '8': '_regular_grid'}[action]
        if os.path.exists(run_dir):
            raise UsageError('%s already exists' % run_dir)
        if os.path.dirname(run_dir):
            os.makedirs(os.path.dirname(run_dir), exist_ok=True)
        dossier = run_dir
        os.mkdir(dossier)
        try:
            shutil.copy(model_src or MODEL_FILE, dossier + '/' + RUN_MODEL_COPY)
            apply_overrides(dossier + '/' + RUN_MODEL_COPY, model_set)
            if action in ('1', '2'):
                shutil.copy(config_src or ALGO_FILE,
                            dossier + '/' + RUN_ALGO_COPY)
                apply_overrides(dossier + '/' + RUN_ALGO_COPY, config_set)
            else:
                shutil.copy(config_src or INPUT_FILES['8'],
                            dossier + '/' + RUN_GRID_COPY)
                apply_overrides(dossier + '/' + RUN_GRID_COPY, config_set)
        except BaseException:
            shutil.rmtree(dossier)
            raise
        sys.path.append(os.path.join(os.getcwd(), dossier))
    elif action in ('3', '4', '5'):
        dossier = simulation
        apply_overrides(dossier + '/' + RUN_MODEL_COPY, model_set)
        apply_overrides(dossier + '/' + RUN_ALGO_COPY, config_set)
        sys.path.append(os.path.join(os.getcwd(), dossier))
    elif config_src or config_set:  # actions 6, 7, 9
        default = INPUT_FILES[action]
        path = os.path.join(tempfile.mkdtemp(prefix='peoc_run_'), default)
        shutil.copy(config_src or default, path)
        apply_overrides(path, config_set)
        import run_config
        run_config.overrides[default] = path

    algo = ''
    if action in ('3', '4', '5'):
        if os.path.exists(dossier + '/acceptation_rate.npy'):
            algo = 'MH'
        if os.path.exists(dossier + '/log_likelihood_times_prior.npy'):
            algo = 'AMIS'
        if not algo:
            raise UsageError('%s holds no MH or AMIS results' % dossier)

    # ---- the action itself, in main_script.py's order
    if action in ('1', '2', '3', '8'):
        print('Path of the simulation: ', dossier)
        import simulationParameters as para
        if para.plot_curves:
            import plotting_functions
            plotting_functions.plot_data(dossier)

    graphs = action == '4'
    if action == '1' or (action == '3' and algo == 'MH'):
        import metropolis_hastings as MH
        MH.metropolis_hastings(dossier, action == '3')
        algo, graphs = 'MH', args.plots
    elif action == '2' or (action == '3' and algo == 'AMIS'):
        import amis
        amis.amis(dossier, action == '3')
        algo, graphs = 'AMIS', args.plots
    if action in ('1', '2', '3') and not graphs:
        print("To plot graphs, run 'run.py plots %s'." % dossier)

    if graphs:
        import graphs_sampling_algorithms as g
        if algo == 'MH':
            g.acceptation_rate(dossier)
            g.parameters_evolution(dossier, algo)
        g.log_vraisemblance(dossier, algo)
        if algo == 'AMIS':
            g.plot_para_proposal_evolution(dossier)
            g.plot_proposal_evolution(dossier)
            g.algo_convergence(dossier, algo)
            g.weights(dossier, algo)
        g.dynamics_best_parameter(dossier, algo, plot_2020=False)
        g.posterior_distributions(dossier, algo,
                                  convergence_to_posterior=False)

    if action == '5':
        import RMSE
        RMSE.RMSE(dossier, algo)
    elif action == '6':
        import graph_comparison_of_MH_iterations  # noqa: F401
    elif action == '7':
        import graph_comparison_of_AMIS_posterior_distributions  # noqa: F401
    elif action == '8':
        import log_likelihood_on_regular_grid
        log_likelihood_on_regular_grid.compute_log_likelihoods(dossier)
    elif action == '9':
        import plot_log_likelihood_on_regular_grid  # noqa: F401
    return 0


if __name__ == '__main__':
    sys.exit(main())
