import numpy as np
import pandas as pd
from scipy import stats
import sys

sys.path.append( '../VaR' )
sys.path.append( '../Cov' )
import VaR as v
import CovEstimation as ce


def ES_normal_distribution(ror, alpha):
    params = v.fit_normal(ror)
    mu = params.loc[0, "mu"]
    sigma = params.loc[0, "sigma"]
    VaR = v.VaR_normal_distribution(ror, alpha)
    x_a = -VaR.loc[0, "VaR Absolute"]
    x_d = -VaR.loc[0, "VaR Diff from Mean"]
    ES = -stats.norm.expect(ub = x_a, loc = mu, scale = sigma, conditional = True)
    diff = -stats.norm.expect(ub = x_d, loc = 0, scale = sigma, conditional = True)
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [diff]})

def ES_normal_distribution_ewvar(ror, alpha, lamb):
    mu = np.mean(ror)
    # ror = ror - mu
    ew_sigma = ce.ewCov(ror, lamb).iloc[0, 0] ** 0.5
    VaR = v.VaR_normal_distribution_ewvar(ror, alpha, lamb)
    x_a = -VaR.loc[0, "VaR Absolute"]
    x_d = -VaR.loc[0, "VaR Diff from Mean"]
    ES = -stats.norm.expect(ub = x_a, loc = mu, scale = ew_sigma, conditional = True)
    diff = -stats.norm.expect(ub = x_d, loc = 0, scale = ew_sigma, conditional = True)
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [diff]})

def ES_t_distribution(ror, alpha):
    params = v.fit_general_t(ror)
    mu = params.loc[0, "mu"]
    sigma = params.loc[0, "sigma"]
    nu = params.loc[0, "nu"]
    VaR = v.VaR_t_distribution(ror, alpha)
    x_a = -VaR.loc[0, "VaR Absolute"]
    x_d = -VaR.loc[0, "VaR Diff from Mean"]
    ES = -stats.t.expect(ub = x_a, args=(nu,), loc = mu, scale = sigma, conditional = True)
    diff = -stats.t.expect(ub = x_d, args=(nu,), loc = 0, scale = sigma, conditional = True)
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [diff]})

def ES_simulation(ror, alpha, n = 10000):
    params = v.fit_general_t(ror)
    mu = params.loc[0, "mu"]
    sigma = params.loc[0, "sigma"]
    nu = params.loc[0, "nu"]
    VaR = v.VaR_simulation(ror, alpha, n)
    x_a = -VaR.loc[0, "VaR Absolute"]
    x_d = -VaR.loc[0, "VaR Diff from Mean"]
    ES = -stats.t.expect(ub = x_a, args=(nu,), loc = mu, scale = sigma, conditional = True)
    diff = -stats.t.expect(ub = x_d, args=(nu,), loc = 0, scale = sigma, conditional = True)
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [diff]})

def ES_historic(ror, alpha, seed = 2, n = 10000):
    VaR = v.VaR_historic(ror, alpha, seed, n)
    np.random.seed(seed)
    ror = np.random.choice(ror.iloc[:, 0], size=n)
    VaR = -VaR.loc[0, "VaR Absolute"]
    ES = -np.mean(ror[ror <= VaR])
    diff = ES + np.mean(ror)
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [diff]})
