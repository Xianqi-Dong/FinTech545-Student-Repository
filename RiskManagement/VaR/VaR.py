import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
import statsmodels.api as sm
from fitter import Fitter
import sys

sys.path.append( '../RiskManagement/Cov' )
import CovEstimation as ce


def fit_normal(data):
    f = Fitter(data, distributions = ['norm'])
    f.fit()
    params = f.fitted_param['norm']
    return pd.DataFrame({"mu": [params[0]], "sigma": [params[1]]})

def fit_general_t(data):
    f = Fitter(data, distributions = ['t'])
    f.fit()
    params = f.fitted_param['t']
    return pd.DataFrame({"mu": [params[1]], 
                         "sigma": [params[2]], 
                         "nu": [params[0]]})

def MLE_t(X, Y):
    X = sm.add_constant(X)
    def ll_t(params):
        df = params[0]
        s = params[1]
        b = params[2:]
        e = Y - np.dot(X, b)
        ll = np.sum(stats.t.logpdf(e, df=df, loc=0, scale=s))
        return -ll
    beta = np.zeros(X.shape[1])
    s = np.std(Y - np.dot(X, beta))
    df = 1
    params = [df, s]
    for i in beta:
        params.append(i)
    bnds = ((1e-9, None), (1e-9, None), (None, None), (None, None), (None, None), (None, None))
    res = optimize.minimize(ll_t, params, bounds=bnds, options={"disp": True})
    beta_mle = res.x[2:]
    return beta_mle

def fit_regression_t(data):
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    betas = MLE_t(X, Y)
    X = sm.add_constant(X)
    e = Y - np.dot(X, betas)
    f = Fitter(e, distributions = ['t'])
    f.fit()
    params = f.fitted_param['t']
    out = {"mu": [params[1]], 
           "sigma": [params[2]], 
           "nu": [params[0]]}
    out["Alpha"] = betas[0]
    for i in range(1, len(betas)):
        out["B" + str(i)] = betas[i]
    return pd.DataFrame(out)

def VaR_normal_distribution(ror, alpha):
    params = fit_normal(ror)
    mu = params.loc[0, "mu"]
    sigma = params.loc[0, "sigma"]
    VaR = -stats.norm.ppf(alpha, loc = mu, scale = sigma)
    diff = -stats.norm.ppf(alpha, loc = 0, scale = sigma)
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [diff]})

def VaR_normal_distribution_ewvar(ror, alpha, lamb):
    mu = np.mean(ror)
    # ror = ror - mu
    ew_sigma = ce.ewCov(ror, lamb).iloc[0, 0] ** 0.5
    VaR = -stats.norm.ppf(alpha, loc = mu, scale = ew_sigma)
    diff = -stats.norm.ppf(alpha, loc = 0, scale = ew_sigma)
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [diff]})
    # return mu, ew_sigma, VaR

def VaR_t_distribution(ror, alpha):
    params = fit_general_t(ror)
    mu = params.loc[0, "mu"]
    sigma = params.loc[0, "sigma"]
    nu = params.loc[0, "nu"]
    VaR = -stats.t.ppf(alpha, df = nu, loc = mu, scale = sigma)
    diff = -stats.t.ppf(alpha, df = nu, loc = 0, scale = sigma)
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [diff]})
    # return mu, sigma, nu, VaR

def VaR_simulation(ror, alpha, n = 10000):
    params = fit_general_t(ror)
    mu = params.loc[0, "mu"]
    sigma = params.loc[0, "sigma"]
    nu = params.loc[0, "nu"]
    sim = np.random.standard_t(nu, n)
    sim = sim * sigma + mu
    return VaR_t_distribution(sim, alpha)

def VaR_historic(ror, alpha, seed = 2, n = 10000):
    np.random.seed(seed)
    ror = np.random.choice(ror.iloc[:, 0], size=n)    
    VaR = -np.quantile(ror, alpha)
    diff = VaR + np.mean(ror)
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [diff]})