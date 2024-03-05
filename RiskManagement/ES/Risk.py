import pandas as pd
import numpy as np
from scipy import stats
import sys

# sys.path.append( '../VaR' )
# import VaR as v
# sys.path.append( '../Sim' )
# import Sim as s
sys.path.append( '../Cov' )
import CovEstimation as ce


def risk(portfolio, ror, lamb = 0.94):
    portfolio["currentValue"] = portfolio["Holding"]  * portfolio["Starting Price"]
    portfolio.index = list(portfolio["Stock"])
    portfolio = portfolio.drop(columns = ['Stock'])
    mu = {}
    ew_sigma = {}
    for stock in ror.columns:
        if stock not in portfolio.index:
            ror = ror.drop(columns=[stock])
    ror_ewCov = ce.ewCov(ror, lamb)
    risk = pd.DataFrame(columns = ["Stock", "VaR95", "VaR95_Pct"])
    for stock in portfolio.index:
        i = risk.shape[0]
        mu[stock] = 0
        ew_sigma[stock] = ror_ewCov.loc[stock, stock] ** 0.5
        risk.loc[i, "Stock"] = stock
        risk.loc[i, "VaR95_Pct"] = -stats.norm.ppf(0.05, loc = mu[stock], scale = ew_sigma[stock])
        risk.loc[i, "VaR95"] = risk.loc[i, "VaR95_Pct"] * portfolio.loc[stock, "currentValue"]
    total_value = sum(portfolio["currentValue"])
    W = portfolio["currentValue"] / total_value
    total_mu = W.T @ list(mu.values())
    total_sigma = np.sqrt(W.T @ ror_ewCov @ W)
    total_VaR = -stats.norm.ppf(0.05, loc = total_mu, scale = total_sigma)
    row_total = risk.shape[0]
    risk.loc[row_total, "Stock"] = "Total"
    risk.loc[row_total, "VaR95_Pct"] = total_VaR 
    risk.loc[row_total, "VaR95"] = total_VaR * total_value
    return risk
