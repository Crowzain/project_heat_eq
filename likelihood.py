# based on previous code by Rodolphe Le Riche

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize, Bounds
import sklearn.gaussian_process as gp


SEED = 42

def mlogLikelihood(
        thetas:np.ndarray,
        Xd:np.ndarray, 
        F:np.ndarray,
        sigma:float=0.0,
        )->float:
    """
    ```mlogLikelihood``` computes the opposite of the log likelihood for a squarred exponential
    kernel where:
        thetas :length scale anistropic parameter,
        Xd     :inputs,
        F      :DoE,
        sigma  :noise parameter
    """

    n = len(Xd)
    kernel = gp.kernels.RBF(length_scale=thetas)
    kXX = kernel(Xd, Xd)+ sigma*np.eye(n)

    c, low = linalg.cho_factor(kXX)
    
    LL = n/2*np.log(2*np.pi) - linalg.det(kXX)/2 - F.T@linalg.cho_solve((c, low), F)
    return -LL

def maxlogLikelihood(
        Xd:np.ndarray, 
        F:np.ndarray,
        sigma:float=0.0,
        tmin:np.ndarray|int|None=None,
        tmax:np.ndarray|int|None=None,
        nbtry:int=20,
        maxit:int=500,
        rng:np.random.Generator|int|None=None,
        verbose:bool=True,
    )->tuple[np.ndarray, float]:
    """
    
    .```maxlogLikelihood``` determines thetas which maximizes the log likelihood
    with L-BFGS-B algorithm with ```nbtry``` random tries of at least ```maxit```
    iterations. Here we use the function from scipy library where:
    *    Xd     :inputs,
    *    F      :DoE,
    *    sigma  :noise parameter
    *    tmin   :lower bound for thetas
    *    tmax   :upper bound for thetas
    """

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng(SEED)

    npar = Xd.shape[1]
    if tmin is None: tmin = np.repeat(0.1, npar)
    elif isinstance(tmin, int): tmin =np.repeat(tmin, npar)
    
    if tmax is None: tmax = np.repeat(5, npar)
    elif isinstance(tmax, int): tmax = np.repeat(tmax, npar)
    
    bestLL = -np.inf
    if verbose: print(f"\n Max Likelihood in {nbtry} restarts\n\n")
    for i in range(nbtry):
        tinit = rng.uniform(tmin, tmax)
        LL = mlogLikelihood(tinit, Xd, F, sigma)

        if verbose: print(f"i:{i}\ttheta_init = {tinit}\tLL={LL}")

        opt_out = minimize(
            mlogLikelihood,
            tinit,
            args=(Xd, F, sigma),
            method="L-BFGS-B",
            bounds=Bounds(tmin, tmax),
            options=({
                "maxiter":maxit,
                "disp":verbose
                })
            )
        if opt_out.fun>bestLL:
            best_thetas = opt_out.x
            bestLL = opt_out.fun
        
            if verbose: print(f"i:{i}\tthetas:{opt_out.x}\tLL={opt_out.fun}")


    return best_thetas, bestLL