import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
import scipy


def close(X: np.ndarray):
    """Closure operator for numpy arrays."""
    return np.divide(X, np.sum(X, axis=1)[:, np.newaxis])


def nancov(X, method='replace'):
    """
    Generates a covariance matrix excluding nan-components.
    Done on a column-column/pairwise basis.
    The result Y may not be a positive definite matrix.
    """
    if method=='rowexclude':
        Xnanfree = X[np.all(np.isfinite(X), axis=1), :].T
        #assert Xnanfree.shape[1] > Xnanfree.shape[0]
        #(1/m)X^T*X
        return np.cov(Xnanfree)
    else:
        X = np.array(X, ndmin=2, dtype=float)
        X -= np.nanmean(X, axis=0)#[:, np.newaxis]
        cov = np.empty((X.shape[1], X.shape[1]))
        cols = range(X.shape[1])
        for n in cols:
            for m in [i for i in cols if i>=n] :
                fn = np.isfinite(X[:, n])
                fm = np.isfinite(X[:, m])
                if method=='replace':
                    X[~fn, n] = 0
                    X[~fm, m] = 0
                    fact = fn.shape[0] - 1
                    c= np.dot(X[:, n], X[:, m])/fact
                else:
                    f = fn & fm
                    fact = f.shape[0] - 1
                    c = np.dot(X[f, n], X[f, m])/fact
                cov[n, m] = c
                cov[m, n] = c
        return cov
    
    
def renormalise(df: pd.DataFrame, components:list=[]):
    """
    Renormalises compositional data to ensure closure.
    A subset of components can be used for flexibility.
    For data which sums to 0, 100 is returned - e.g. for TE-only datasets
    """
    dfc = df.copy()
    if components:
        cmpnts = [c for c in components if c in dfc.columns]
        dfc.loc[:, cmpnts] =  100. * dfc.loc[:, cmpnts].divide(
                              dfc.loc[:, cmpnts].sum(axis=1).replace(0, np.nan),
                                                               axis=0)
        return dfc
    else:
        dfc = dfc.divide(dfc.sum(axis=1).replace(0, 100), axis=0) * 100.
        return dfc
    
    
def add_ratio(df: pd.DataFrame, ratio, alias=''):
    """
    Add a ratio of components A and B, given in the form of string 'A/B'.
    Can be assigned an alias name
    """
    num, den = ratio.split('/')
    name = [ratio if not alias else alias][0]
    df.loc[:, name] = df.loc[:, num] / df.loc[:, den]
    


class LinearTransform(TransformerMixin):
    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = 'Crude'

    def transform(self, X, *args):
        X = np.array(X)
        return X

    def inverse_transform(self, Y, *args):
        Y = np.array(Y)
        return Y

    def fit(self, X, *args):
        return self


class ALRTransform(TransformerMixin):
    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = 'ALR'

    def transform(self, X, *args):
        X = np.array(X)
        return alr(X)

    def inverse_transform(self, Y, *args):
        Y = np.array(Y)
        return inv_alr(Y)

    def fit(self, X, *args):
        return self


class CLRTransform(TransformerMixin):
    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = 'CLR'

    def transform(self, X, *args):
        X = np.array(X)
        return clr(X)

    def inverse_transform(self, Y, *args):
        Y = np.array(Y)
        return inv_clr(Y)

    def fit(self, X, *args):
        return self


class ILRTransform(TransformerMixin):
    def __init__(self, **kwargs):
        self.kpairs = kwargs
        self.label = 'ILR'

    def transform(self, X, *args):
        X = np.array(X)
        self.X = X
        return ilr(X)

    def inverse_transform(self, Y, *args):
        Y = np.array(Y)
        return inv_ilr(Y, X=self.X)

    def fit(self, X, *args):
        return self


def alr(X: np.ndarray, ind: int=-1, **kwargs):
    """
    Additive log ratio transform.
    """
    if ind < 0: ind += X.shape[1]
    X = X / X[:, ind][:, np.newaxis]
    Y = np.log(X[:, [i for i in range(X.shape[1]) if not i==ind]])  # Log operation from D to D-1
    return Y


def inv_alr(Y: np.ndarray, **kwargs):
    """
    Inverse additive log ratio transform.
    """
    Y =  np.concatenate((Y, np.zeros((Y.shape[0], 1))), axis=1)
    X = np.exp(Y)  # Inverse log operation, from D-1 to D
    X = np.divide(X, np.nansum(X, axis=1)[:, np.newaxis])  # Closure operation
    return X


def ALR_mean(X, index=-1):
    alr_transform = ALRTransform()
    M_ALR = alr_transform.fit_transform(X)
    M_ALR_mean = np.nanmean(M_ALR, axis=0)
    M_mean = alr_transform.inverse_transform(M_ALR_mean)
    return M_mean


def clr(X: np.ndarray):
    X = np.divide(X, np.sum(X, axis=1)[:, np.newaxis])  # Closure operation
    Y = np.log(X)  # Log operation
    Y -= 1/X.shape[1] * np.nansum(Y, axis=1)[:, np.newaxis]
    return Y


def inv_clr(Y: np.ndarray):
    X = np.exp(Y)  # Inverse of log operation
    X = np.divide(X, np.nansum(X, axis=1)[:, np.newaxis])  #Closure operation
    return X


def orthagonal_basis(X: np.ndarray):
    D = X.shape[1]
    H = scipy.linalg.helmert(D, full=False)  # D-1, D Helmert matrix, exact representation of ψ as in Egozogue's book
    return H[::-1]


def ilr(X: np.ndarray):
    d = X.shape[1]
    Y = clr(X)
    psi = orthagonal_basis(X)  # Get a basis
    psi = orthagonal_basis(clr(X)) # trying to get right algorithm
    assert np.allclose(psi @ psi.T, np.eye(d-1))
    return Y @ psi.T


def inv_ilr(Y: np.ndarray, X: np.ndarray=None):
    psi = orthagonal_basis(X)
    C = Y @ psi
    X = inv_clr(C)  # Inverse log operation
    return X