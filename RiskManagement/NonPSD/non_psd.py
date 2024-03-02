import pandas as pd
import numpy as np
import scipy.linalg as la
import sys

sys.path.append( '../Cov' )
import CovEstimation as ce


def nearPSDCor(cor, epsilon = 0.0):
    vals, vecs = la.eig(cor)
    vals = np.real_if_close(np.maximum(vals, epsilon))
    T = 1 / np.dot(vecs * vecs, vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = np.dot(np.dot(T, vecs), l)
    near_psd_cor = pd.DataFrame(np.dot(B, B.T), columns = cor.columns, index = cor.columns)
    return near_psd_cor
    
def nearPSDCov(cov, epsilon = 0.0):
    sd, cor = ce.CovtoCor(cov)
    near_psd_cor = nearPSDCor(cor, epsilon)
    near_psd_cov = ce.CortoCov(sd, near_psd_cor)
    return near_psd_cov

def higham_nearestPSDCor(A, tol=[], weights=None, epsilon = 1e-9, max_iterations=100):
    S = np.zeros(np.shape(A))
    eps = np.spacing(1)
    if not tol:
        tol = eps * np.shape(A)[0] * np.array([1, 1])
    if weights is None:
        weights = np.ones(np.shape(A)[0])
    X = np.copy(A)
    Y = np.copy(A)
    gamma = np.inf
    Whalf = np.sqrt(np.outer(weights, weights))
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        R = X - S
        R_wtd = Whalf*R
        X = proj_spd(R_wtd)
        X = X / Whalf
        S = X - R
        Y = np.copy(X)
        np.fill_diagonal(Y, 1)
        gamma_old = gamma
        gamma = np.linalg.norm(Y, 'fro')
        vals, vecs = la.eig(X)
        if abs(gamma - gamma_old) < tol[0] and min(vals) > -epsilon:
            break
        X = np.copy(Y)
    return pd.DataFrame(X, columns = A.columns, index = A.columns)

def higham_nearestPSDCov(cov):
    sd, cor = ce.CovtoCor(cov)
    near_psd_cor = higham_nearestPSDCor(cor)
    near_psd_cov = ce.CortoCov(sd, near_psd_cor)
    return near_psd_cov

def proj_spd(A):
    d, v = np.linalg.eigh(A)
    A = (v * np.maximum(d, 0)).dot(v.T)
    A = (A + A.T) / 2
    return(A)

def chol_psd(A):
    L = la.cholesky(A).T
    return pd.DataFrame(L, columns = A.columns, index = A.columns)
