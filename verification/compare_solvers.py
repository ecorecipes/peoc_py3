"""Compare the exact dense Crank-Nicolson solver with (a) a DCT closed form
and (b) the port's 'fast' sparse-LU solver (reaction_diffusion.fast_operators,
the shipped code), on the real PEOC model, for fields and likelihoods.
usage: python compare_solvers.py <peoc_py3_dir>"""
import os, sys, shutil, time, tempfile
import numpy as np

code_dir = os.path.abspath(sys.argv[1])
run_dir = tempfile.mkdtemp()
shutil.copy(os.path.join(code_dir, 'parameters_of_the_model.py'),
            os.path.join(run_dir, 'simulationParameters.py'))
os.chdir(code_dir)
sys.path.append(os.path.join(code_dir, 'source_files'))
sys.path.append(run_dir)
import simulationParameters as para
import environmental_data as env
import initialisation_rachis as ir
import reaction_diffusion as RD
import likelihood

nx, ny, N, dt, tau = para.n_x_RD, para.n_y_RD, para.N, para.delta_t, para.diffusion_period

def dct_matrix(m):
    k = np.arange(m)[:, None]; j = np.arange(m)[None, :]
    C = np.sqrt(2.0 / m) * np.cos(np.pi * k * (2 * j + 1) / (2.0 * m))
    C[0] /= np.sqrt(2.0)
    return C
Cx, Cy = dct_matrix(nx), dct_matrix(ny)
lam = (4 * np.sin(np.pi * np.arange(nx) / (2.0 * nx)) ** 2)[:, None] + \
      (4 * np.sin(np.pi * np.arange(ny) / (2.0 * ny)) ** 2)[None, :]

def gain(D):
    x = D * dt / (2 * para.h ** 2) * lam
    g = np.full(x.shape, float(N))
    pos = x > 0; xs = x[pos]; om = np.empty_like(xs)
    s = xs < 0.25
    om[s] = -np.expm1(N * (np.log1p(-xs[s]) - np.log1p(xs[s])))
    om[~s] = 1 - ((1 - xs[~s]) / (1 + xs[~s])) ** N
    g[pos] = om / (2 * xs)
    return g * dt / tau

def run(p, method):
    (r_symp, r_allee, beta_0, beta_1, D, S_sat, gamma, C_nr, C_init, kappa) = p
    orig = {}
    if method != 'exact':
        if method == 'dct':
            g = gain(D)
            diffuse = lambda s: Cx.T.dot(g * Cx.dot(s).dot(Cy.T)).dot(Cy)
        else:  # 'fast': the shipped operators and the shipped step
            lu, B = RD.fast_operators(D)
            def diffuse(s):
                src = dt * s.ravel() / tau
                w = np.zeros(nx * ny)
                for j in range(N):
                    w = lu.solve(B.dot(w) + src)
                return w.reshape(nx, ny)
        def fake_reaction_di(r_allee, beta_0, beta_1, D, S_sat, f1, f2, annee_supp=False,
                             return_spore=False, nb_years=para.nb_years):
            fpa = np.zeros((nb_years, nx, ny)); fpa[0] = f1; fpa[1] = f2
            ih = para.initialization_year + 2 - para.first_year_of_data
            for i in range(nb_years - 2):
                spores = (fpa[i] + fpa[i + 1]) * RD.d_frene
                if r_allee != 0:
                    spores[spores < r_allee] *= (spores[spores < r_allee] / r_allee)
                spores *= (beta_0 + beta_1 * RD.H[ih + i]) if beta_1 != 0. else beta_0
                sa = diffuse(spores)
                sa[sa > S_sat] = S_sat
                fpa[i + 2] = sa
            return fpa
        orig['rd'] = RD.reaction_di
        RD.reaction_di = fake_reaction_di
    t0 = time.time()
    fungi_holder = {}
    real_cll = likelihood.compute_log_likelihood
    def spy(*a, **k):
        fungi_holder['f'] = a[4]
        return real_cll(*a, **k)
    likelihood.compute_log_likelihood = spy
    ll, z = likelihood.log_likelihood(np.array(p), ir.infection_1, ir.infection_2)
    likelihood.compute_log_likelihood = real_cll
    if 'rd' in orig:
        RD.reaction_di = orig['rd']
    return ll, float(z), fungi_holder['f'], time.time() - t0

PARAS = [
    [1000.0, 15.0, 0.001, 0.0001, 30.0, 60.0, 40.0, 0.01, 0.002, 0.01],
    [1000.0, 0.0, 30.0, 0.0, 20.0, 90.0, 22.0, 1.0, 0.0085, 0.1],
    [1000.0, 50.0, 40.0, 2.0, 25.0, 80.0, 30.0, 0.5, 0.01, 0.5],
    [1000.0, 0.0, 24.0, 0.0, 18.0, 90.0, 21.50001, 1.0, 0.0085, 0.05],
    [1000.0, 0.0, 24.0, 0.0, 2.0, 90.0, 21.50001, 1.0, 0.0085, 0.05],
    [1000.0, 0.0, 24.0, 0.0, 150.0, 90.0, 21.50001, 1.0, 0.0085, 0.05],
]
ship = np.load(os.path.join(code_dir, 'Simulations', 'MH_initial_condition_1', 'sampled_parameters.npy'))
PARAS.append(list(ship[-1]))
for i, p in enumerate(PARAS):
    ll0, z0, f0, t0 = run(p, 'exact')
    print('para %d D=%g: exact ll=%r terme0=%g (%.1fs); zeros=%d min>0=%.3g'
          % (i, p[4], ll0, z0, t0, (f0 == 0).sum(), f0[f0 > 0].min()))
    for m in ['dct', 'fast']:
        ll, z, f, t = run(p, m)
        nz = f0 > 1e-300
        rel = np.abs(f[nz] - f0[nz]) / f0[nz]
        print('   %-5s ll=%r terme0=%g (%.2fs) |dll|/|ll|=%.2g  field: max relerr(f0>0)=%.2g '
              'median=%.2g  neg=%d  max abs/max=%.2g'
              % (m, ll, z, t, abs(ll - ll0) / max(abs(ll0), 1e-300), rel.max(), np.median(rel),
                 (f < 0).sum(), np.abs(f - f0).max() / np.abs(f0).max()))
    sys.stdout.flush()
