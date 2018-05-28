# Reproduced from Matlab code
# Palarea-Albaladejo (2007)
# A Parametric Approach for Dealing with Compositional Rounded Zeros

"""
Consider just using np/scipy inverse for regression calculation - or instead PETSc4py.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import comb
from compositions import nancov


def finitemax(arr, *args):
    return np.nanmaxs(arr[np.isfinite(arr)], *args)

def matlab_length(arr):
    return max(arr.shape)


def alr(X: np.ndarray, d:int = -1):
    if d < 0: d += X.shape[1]
    alr = np.log(np.divide(X, X[:, d][:, np.newaxis]))
    alr = alr[:, [i for i in range(X.shape[1]) if not i==d]]
    return alr


def alr_inv(X: np.ndarray, d:int = -1):
    m, n = X.shape
    X = np.c_[X, np.zeros(m)]
    X = np.exp(X)
    X = np.divide(X,  np.sum(X, axis=1)[:, np.newaxis]) # Closure

    if d < 0: d += X.shape[1]
    indexes = np.array([*range(0, X.shape[1])])
    indexes = [i for i in indexes[:-1] if i < d] + [-1] + [i for i in indexes[:-1] if i >= d]
    X = X[:, indexes]
    return X


def sweep(g, ind: range(1)):
    g = np.asarray(g)
    n = g.shape[0]
    if g.shape != (n, n):
        raise ValueError('Not a square array')
    if not np.allclose(g - g.T, 0):
        raise ValueError('Not a symmetrical array')

    for k in ind:
        if k >= n:
            raise ValueError('Not a valid row number')
        #  Fill with the general formula
        assert g[k, k] != 0.
        h = g - np.outer(g[:, k], g[k, :]) / g[k, k]
        # h = g - g[:, k:k+1] * g[k, :] / g[k, k]
        # Modify the k-th row and column
        h[:, k] = g[:, k] / g[k, k]
        h[k, :] = h[:, k]
        # Modify the pivot
        h[k, k] = -1 / g[k, k]
    return h


def old_sweep(A: np.ndarray, ind: range(1)):
    """This subroutine executes the sweep operator.

    "As input, SWEEP requires a symmetric matrix A where mean vector m
    and covariance matrix S are arranged in a special manner that simplifies the
    calculations."

    See Dempster (1969), Goodnight (1979).
    The SWEEP operator allows a statistician to quickly regress all variables against
    one specified variable, obtaining OLS estimates for regression coefficients and
    variances in a single application. Subsequent applications of the SWP operator
    allows for regressing against more variables.
    """
    S = A.copy()
    p = A.shape[1]

    for j in ind:
        S[j, j] = -1 / A[j, j]
        for i in range(0, p):
            if i != j:
                S[i, j] = -A[i, j] * S[j, j]
                S[j, i] = S[i, j]

        for i in range(0, p):
            if i != j:
                for k in range(0, p):
                    if k != j:
                        S[i, k] = A[i ,k] - S[i, j] * A[j, k]
                        S[k, i] = S[i, k]

    return S


def reg_sweep(M: np.ndarray,
              C: np.ndarray,
              varobs: np.ndarray,
              error_threshold=100):
    """
    From Palarea-Albaladejo, J., Martin-Fernandez, J. A.
    "A modified EM alr-algorithm for replacing rounded zeros in compositional data sets".
    This subroutine calculates the estimated coefficients of the regression equations
    of the missing variables on the observed variables. It also provides the residual
    covariance matrix.

    "REG_SWEEP uses p-j to indicate to SWEEP the submatrix of A on which to apply the
    sweep, and also to select the elements of interest in A after sweeping.
    That is, the involved regression coefficients and the residual covariance matrix.
    Note that previously imputed data are not used to obtain these estimates.
    """
    assert np.isfinite(M).all()
    assert np.isfinite(C).all()
    #assert (np.abs(np.log(M)) < error_threshold).all()
    p = matlab_length(M)
    q = matlab_length(varobs)
    i = np.ones(p)
    i[varobs] -= 1
    dep = np.array(np.nonzero(i))[0] # indicies where i is nonzero
    # Shift the non-zero element to the end for pivoting
    reor = np.concatenate(([0], varobs+1, dep+1), axis=0)
    A = np.zeros((p+1, p+1))  # Square array
    A[0, 0] = -1
    A[0, 1:p+1] = M
    A[1:p+1, 0] = M.T
    A[1:p+1, 1:p+1] = C
    A = A[reor, :][:, reor]
    Astart = A.copy()
    assert (np.diag(A)!=0).all()  # Not introducing extra zeroes
    A = sweep(A, [i for i in range(0, q)])
    if not np.isfinite(A).all():  # Typically caused by infs
        print(A)
        A[~np.isfinite(A)] = 0
    β = A[0:q+1, q+1:p+1]
    σ2_res = A[q+1:p+1, q+1:p+1]
    return β, σ2_res


def EMCOMP(X, threshold_vector, tol = 0.0001):
    """
    EMCOMP replaces rounded zeros in a compositional data set by the algorithm
    described in:  Palarea-Albaladejo, J., Martin-Fernandez, J. A.
     "A modified EM alr-algorithm for replacing rounded zeros in compositional data sets".

    Input arguments:
       data: the compositional data set with rounded zeros.
       threshold_vector: row vector of threshold values for each component (in proportions).

    Output:
       X_ast: replaced compositional data set.
       prop_zeros: proportion of zeros in the original data set.
       ni_ters: number of iterations needed for convergence.

    At least one component without missing values is needed for the divisor. Rounded zeros/
    missing values are replaced by values below their respective detection limits.
    """
    X = X.copy()
    n_obs, p = X.shape
    D = p

    """Close the X rows to 1"""
    X = np.divide(X, np.nansum(X, axis=1)[:, np.newaxis])
    """Convert zeros into missing values"""
    X[X == 0] = np.nan
    """Use a divisor free of missing values"""
    assert np.all(np.isfinite(X), axis=0).any()
    pos = np.argmax(np.all(np.isfinite(X), axis=0))
    Yden = X[:, pos]
    """Compute the matrix of censure points Ψ"""
    cpoints = np.ones((n_obs, 1)) @ np.log(threshold_vector[np.newaxis, :]) -\
              np.log(Yden[:, np.newaxis]) @ np.ones((1,p)) - \
              np.spacing(1.)  # Machine epsilon
    assert np.isfinite(cpoints).all()
    cpoints[:, pos] = np.nan

    prop_zeroes = np.count_nonzero(~np.isfinite(X)) / (n_obs * p)
    Y = alr(X, pos)
    #Y = np.vstack((alr(X, pos).T, np.zeros(n_obs))).T
    #---------------Log Space--------------------------------
    n_obs, p = Y.shape
    M = np.nanmean(Y, axis=0)  # μ0
    C = nancov(Y)  # Σ0
    assert np.isfinite(M).all() and np.isfinite(C).all()
    """
    ------------------------------------------------
    Stage 2: Find and enumerate missing data patterns.
    ------------------------------------------------
    """
    misspattern = np.zeros(n_obs)
    miss = ~np.isfinite(Y)
    observationmiss = np.array(np.nonzero(~np.isfinite(np.sum(Y, axis=1)))[0])
    maximum_patterns = comb((D-1) * np.ones(D-2), np.arange(D-2)+1).sum().astype(int)
    misspattern[observationmiss] = maximum_patterns * 10


    """Missing data patterns"""
    pattern_no = 0
    for obs_no in range(0, n_obs):
        if misspattern[obs_no] > pattern_no:
            pattern_no += 1
            misspattern[obs_no] = pattern_no

            if obs_no < n_obs:
                for j in range(obs_no+1, n_obs):
                    if (misspattern[j] > pattern_no) & \
                       (miss[obs_no, :] == miss[j, :]).all():
                        misspattern[j] = pattern_no
    max_pattern = pattern_no
    misspattern = misspattern.astype(int)
    """
    ---------------------------------------------------------------------------
    """
    another_iter = True
    niters = 0
    while another_iter:
        niters += 1
        Mnew = M
        Cnew = C
        Ystar = Y.copy()
        V = np.zeros((p, p))
        for pattern_no in np.unique(misspattern):
            patternrows = np.array(np.nonzero(misspattern == pattern_no))[0]
            ni = matlab_length(patternrows)
            observed = np.isfinite(Y[patternrows[0], :])
            varobs = np.array(np.nonzero(observed)).flatten()
            varmiss = np.array(np.nonzero(~observed)).flatten()
            nvarobs = matlab_length(varobs)
            nvarmiss = matlab_length(varmiss)
            sigmas = np.zeros(p)
            """
            Regress yj on y-j by [Β(j, μ), σ2] = REG_SWEEP(μ(t-1), Σ(t-1), p-j).
            Regression against other variables.
            """
            if nvarobs: # Only where there observations
                B, σ2_res = reg_sweep(Mnew, Cnew, varobs)
                if B.size: # If there are missing elements in the pattern
                    B = B.flatten()
                    Ystar[np.ix_(patternrows, varmiss)] = (np.ones(ni) * B[0] + Y[np.ix_(patternrows, varobs)] @ B[1:(nvarobs+1)])[:, np.newaxis]
                    assert np.isfinite(Ystar[np.ix_(patternrows, varmiss)]).all()
                    sigmas[varmiss] = np.sqrt(np.diag(σ2_res))
                    for missvar_idx in varmiss:
                        sigma = sigmas[missvar_idx]
                        std_devs = (cpoints[patternrows, missvar_idx] - Ystar[patternrows, missvar_idx]) / sigma
                        fdN01 = stats.norm.pdf(std_devs, loc=0, scale=1)  #ϕ
                        fdistN01 = stats.norm.cdf(std_devs, loc=0, scale=1)  #Φ
                        sigdevs = sigma * fdN01 / fdistN01
                        sigdevs = np.where(np.isfinite(sigdevs), sigdevs, 0.) #np.zeros(ni)
                        Ystar[patternrows, missvar_idx] -= sigdevs  #Impute each yj
                    V[np.ix_(varmiss, varmiss)] += σ2_res * ni

        """Update and store parameter vector (μ(t), Σ(t))."""
        M = np.nanmean(Ystar, axis=0)
        dif = Ystar - np.ones(n_obs)[:, np.newaxis] @ M[np.newaxis, :]
        dif[np.isnan(dif)] = 0.
        try:
            PearsonCorr = np.dot(dif.T, dif)
            assert n_obs > 1
            C = (PearsonCorr + V)/(n_obs-1)
            assert np.isfinite(C).all()
        except AssertionError:
            print(PearsonCorr)
            C = Cnew
        Mdif = np.nanmax(np.abs(M - Mnew))
        Cdif = np.nanmax(np.abs(C - Cnew))
        if np.nanmax(np.vstack([Mdif, Cdif]).flatten()) < tol:  # Convergence checking
            another_iter = False

    Xstar = alr_inv(Ystar, pos)
    return Xstar, prop_zeroes, niters
