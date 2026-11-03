# -*- coding:Utf-8 -*-
#Solve the reaction-diffusion model with an Euler scheme
import warnings
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import simulationParameters as para
import environmental_data as env_data

if para.discreti==1 or para.discreti==1.0:
    d_frene = env_data.ash_density
    H = env_data.H
else:
    #finer discretisation than at the scale of the quadrat
    d_frene = np.repeat(env_data.ash_density,para.discreti, axis=0)
    d_frene = np.repeat(d_frene,para.discreti, axis=1)
    H = np.repeat(env_data.H,para.discreti, axis=1) 
    H = np.repeat(H,para.discreti, axis=2)


######################## Preliminary computations ##############################
def mat_laplacien_1D(n):
    """retourne la matrice du Laplacien 1D (de dimension n*n, où n est le
    nombre de points de discrétisation) avec les conditions de Neumann"""
    A = 2*np.eye(n) - np.diag(np.ones(n-1),1)- np.diag(np.ones(n-1),-1)
    A[0,0] = 1   #Neumann condition
    A[-1,-1] = 1 #Neumann condition
    return A

def mat_laplacien_2D(nx,ny):
    A1 = mat_laplacien_1D(nx)
    A2 = mat_laplacien_1D(ny)
    return np.kron(np.eye(nx),A2) +np.kron(A1,np.eye(ny))

n = para.n_x_RD*para.n_y_RD

# Diffusion solver (not in PEOC 1.0), set in parameters_of_the_model.py.
# 'exact': dense Crank-Nicolson, as in PEOC 1.0, bitwise reproducible.
# 'fast':  the same Crank-Nicolson scheme and steps, applied with a sparse LU
#          factorisation; ~10x faster, results within ~1e-12 relative.
# Run folders created before the option existed have no such variable and
# run 'exact'.
solver = getattr(para, 'diffusion_solver', 'exact')
if solver not in ('exact', 'fast'):
    raise ValueError("diffusion_solver must be 'exact' or 'fast', not %r"
                     % (solver,))

if solver == 'exact':
    M_lap = mat_laplacien_2D(para.n_x_RD, para.n_y_RD)
else:
    M_lap = None  # the dense matrix (166 MB) is not needed

def sparse_laplacien_2D(nx, ny):
    """mat_laplacien_2D(nx, ny) in sparse (CSC) form."""
    def lap_1D(m):
        d = 2.0*np.ones(m)
        d[0] = 1.0   #Neumann condition
        d[-1] = 1.0  #Neumann condition
        return sparse.diags([d, -np.ones(m-1), -np.ones(m-1)], [0, 1, -1])
    return (sparse.kron(sparse.identity(nx), lap_1D(ny)) +
            sparse.kron(lap_1D(nx), sparse.identity(ny))).tocsc()

###################### End of preliminary computations ########################

class _RecentCache(object):
    """The `size` most recently used values of an expensive function of one
    key, most recent last; size 0 disables caching.  Not in PEOC 1.0.
    When full, the oldest entry is dropped *before* the new value is built,
    so at most `size` values (not size + 1) are alive while building."""
    def __init__(self, size):
        self.size = size
        self._entries = []

    def get(self, key, build):
        for entry in self._entries:
            if entry[0] == key:
                self._entries.remove(entry)
                self._entries.append(entry)
                return entry[1]
        if self.size == 0:
            return build()
        del self._entries[:max(0, len(self._entries) - self.size + 1)]
        value = build()
        self._entries.append((key, value))
        return value

    def resize(self, size):
        self.size = size
        if len(self._entries) > size:
            del self._entries[:len(self._entries) - size]

# Dense inverses (166 MB each on the 68 x 67 grid).  Two suit component-wise
# MH (current D, and the previous one after a rejected D proposal).
_dense_cache = _RecentCache(2)

def set_dense_cache_size(size):
    """Number of dense inverses diffusion_operators keeps.  AMIS draws a new
    D for every likelihood, so its cache would never hit: amis.py sets 0,
    which keeps memory use as in PEOC 1.0."""
    _dense_cache.resize(size)

def diffusion_operators(D):
    """Crank-Nicolson operators M_inv = (I + c L)^-1 and A = 2 M_inv - I,
    with c = D dt / (2 h^2).

    The inverse dominates the cost of a likelihood (10-25 s), and depends
    only on D.  Component-wise Metropolis-Hastings changes D in one update out
    of every nb_varied_para, and after a rejected D proposal it returns to the
    previous D, so the last two inverses are kept.  A cached inverse is the
    result of the identical computation on the identical matrix, so the
    outputs are bitwise the same as recomputing it.  A is cheap and is
    rebuilt each time (keeping it too would double the cache's memory).

    With the 'fast' solver the dense operators are not used: (None, None)."""
    if solver == 'fast':
        return None, None
    cste = D*para.delta_t/(2*(para.h**2))
    M_inv = _dense_cache.get(
        cste, lambda: np.linalg.inv(np.eye(n)+cste*M_lap))
    A = 2*M_inv-np.eye(n)
    return M_inv, A

_L_sparse = []            # sparse Laplacian, built on first use
_lu_cache = _RecentCache(2)
_warned_large_c = []

def fast_operators(D):
    """'fast' solver: a sparse LU factorisation of (I + c L) and the sparse
    matrix B = I - c L, so that one Crank-Nicolson step
        w <- A w + dt M_inv nu / tau = (I + c L)^-1 (B w + dt nu / tau)
    costs a sparse product and two triangular solves instead of a dense
    product.  (I + c L) is a diagonally dominant M-matrix and B has no
    negative entry while c < 1/4 (D < 1280 for 16 km quadrats and dt = 0.1),
    so no step involves cancellation: tiny values far from the epidemic
    front, which enter the likelihood through log(q), keep their relative
    accuracy.  (A spectral/DCT solution does not, and gives NaN.)
    Above c = 1/4, i.e. D > h^2/(2 dt) (1280 at h = 16 km, dt = 0.1; less
    with discreti > 1), B has negative diagonal entries and that guarantee
    is lost, so a warning is issued once."""
    cste = D*para.delta_t/(2*(para.h**2))
    if 4*cste > 1 and not _warned_large_c:
        _warned_large_c.append(True)
        warnings.warn("diffusion_solver='fast' with D = %r: c = D*dt/(2h^2) = "
                      "%.3g > 1/4, so small values far from the front may "
                      "lose relative accuracy; use 'exact' for such D"
                      % (D, cste))

    def build():
        if not _L_sparse:
            _L_sparse.append(sparse_laplacien_2D(para.n_x_RD, para.n_y_RD))
        L_sparse = _L_sparse[0]
        identity = sparse.identity(n, format='csc')
        # symmetric fill-reducing ordering and diagonal pivots: the factors
        # of a symmetrically permuted M-matrix are M-matrices themselves
        lu = sparse_linalg.splu((identity + cste*L_sparse).tocsc(),
                                permc_spec='MMD_AT_PLUS_A',
                                diag_pivot_thresh=0.0,
                                options=dict(SymmetricMode=True))
        B = (identity - cste*L_sparse).tocsr()
        return lu, B
    return _lu_cache.get(cste, build)

def solve_reaction_di(r_allee, beta_0, beta_1, D, S_sat,
                fongi_1,fongi_2, M_inv, A, annee_supp = False,
                return_spore=False,
                nb_years = para.nb_years):
        
    if para.discreti==1:
        fongi_1_RD = fongi_1
        fongi_2_RD = fongi_2
    else:
        #finer discretisation than at the scale of the quadrat
        fongi_1_RD = np.repeat(fongi_1,para.discreti, axis=0)
        fongi_1_RD = np.repeat(fongi_1_RD,para.discreti, axis=1)
        fongi_2_RD = np.repeat(fongi_2,para.discreti, axis=0)
        fongi_2_RD = np.repeat(fongi_2_RD,para.discreti, axis=1)
        

    #memeroy allocation and initialization:
    fungi_per_ash = np.zeros((nb_years, para.n_x_RD, para.n_y_RD))
    fungi_per_ash[0] = fongi_1_RD
    fungi_per_ash[1] = fongi_2_RD
    
    if return_spore:
        spore_non_sature =  np.zeros((nb_years-2, para.n_x_RD, para.n_y_RD))
        
    #first index for the environemental data
    index_hum = para.initialization_year + 2 - para.first_year_of_data
    if solver == 'fast':
        lu, B = fast_operators(D)
    for i in np.arange(nb_years-2):
        #spores of the year i+2
        spores = (fungi_per_ash[i]+fungi_per_ash[i+1])*d_frene
        if r_allee!=0:
            spores[spores<r_allee] *= (spores[spores<r_allee]/r_allee)
        if beta_1!=0.:
            spores *= (beta_0+beta_1*H[index_hum+i])
        else:
            spores *= beta_0
            
        nu_a = spores.ravel()

        #spores diffusion
        w = nu_a*0.0
        if solver == 'fast':
            src = para.delta_t*nu_a/para.diffusion_period
            for j in np.arange(para.N):
                w = lu.solve(B.dot(w)+src)
        else:
            b = para.delta_t*np.dot(M_inv,nu_a)/para.diffusion_period
            for j in np.arange(para.N):
                w = np.dot(A,w)+b
        
        #spores after diffusion
        spores_automne = w.reshape(np.size(spores.T[0]), np.size(spores[0]))

        #rachis production
        if return_spore:
            spore_non_sature[i] = spores_automne*(d_frene>0)
        spores_automne[spores_automne>S_sat] = S_sat #spores saturation
        fungi_per_ash[i+2] = spores_automne*1.
        
        if np.min(fungi_per_ash[i+2])<0:
            print("WARNING : fungi_per_ash<0", i, end=' ')
            print(r_allee, beta_0, beta_1, D, S_sat)
    
    #if finer discretisation than at the scale of the quadrat
    if para.discreti!=1:
        R_temp = np.zeros((nb_years, para.nb_quadrats_x, para.n_y_RD)) 
        for a in np.arange(nb_years):
            for i in np.arange(para.discreti):
                R_temp += fungi_per_ash[a][i::para.discreti]
        fungi_per_ash = np.zeros((nb_years, para.nb_quadrats_x,
                               para.nb_quadrats_y))
        for a in np.arange(nb_years):
            for i in np.arange(para.discreti):
                fungi_per_ash += R_temp[a][:,i::para.discreti]
        fungi_per_ash /= discreti**2

    
    if return_spore:
        return [fungi_per_ash, spore_non_sature]
    else:
        return fungi_per_ash


def reaction_di(r_allee, beta_0, beta_1, D, S_sat,
                fongi_1,fongi_2, annee_supp = False, return_spore=False,
                nb_years = para.nb_years):

    
    """solves the raction-diffusion equation
    inputs:
      r_allee : Allee effect parameter
      beta_0  : spores production parameter
      beta_1  : rainfall impact parameter
      D       : diffusion parameter
      S_sat   : spores saturation
    outputs:
      3-dimensional matrix describing the fungi quantity by tree for the 
      differents years
    """

    M_inv, A = diffusion_operators(D)

    return solve_reaction_di(r_allee, beta_0, beta_1, D, S_sat,
                fongi_1,fongi_2, M_inv, A, annee_supp, return_spore,nb_years)
