import pandas as pd
import numpy as np
from scipy import stats
import sys

sys.path.append( '../VaR' )
import VaR as v
sys.path.append( '../Sim' )
import Sim as s


def copula_risk(portfolio, ror, nSim = 100000):
    portfolio["currentValue"] = portfolio["Holding"]  * portfolio["Starting Price"]
    portfolio.index = list(portfolio["Stock"])
    portfolio = portfolio.drop(columns = ['Stock'])
    models = {}
    mu = {}
    sigma = {}
    nu = {}
    U = pd.DataFrame()
    for stock in portfolio.index:
        if portfolio.loc[stock, "Distribution"] == "Normal":
            models[stock] = v.fit_normal(ror[stock])
            mu[stock] = models[stock].loc[0, "mu"]
            sigma[stock] = models[stock].loc[0, "sigma"]
        elif portfolio.loc[stock, "Distribution"] == "T":
            models[stock] = v.fit_general_t(ror[stock])
            mu[stock] = models[stock].loc[0, "mu"]
            sigma[stock] = models[stock].loc[0, "sigma"]
            nu[stock] = models[stock].loc[0, "nu"]
        U[stock] = (ror[stock] - mu[stock]) / sigma[stock]
    spcor = U.corr(method='spearman')
    uSim = s.simulate_pca(spcor, nSim)
    uSim = stats.norm.cdf(uSim)
    uSim = pd.DataFrame(uSim, columns = portfolio.index)
    simRet = pd.DataFrame()
    for stock in uSim.columns:
        if portfolio.loc[stock, "Distribution"] == "Normal":
            simRet[stock] = stats.norm.ppf(uSim[stock], loc = mu[stock], scale = sigma[stock])
        elif portfolio.loc[stock, "Distribution"] == "T":
            simRet[stock] = stats.t.ppf(uSim[stock], df = nu[stock], loc = mu[stock], scale = sigma[stock])
    # simulatedValue = portfolio["currentValue"] * (1 + simRet)
    pnl = portfolio["currentValue"] * simRet
    pnl["Total"] = 0
    risk = pd.DataFrame(columns = ["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])
    for stock in pnl.columns[:-1]:
        pnl["Total"] += pnl[stock]
        i = risk.shape[0]
        risk.loc[i, "Stock"] = stock
        risk.loc[i, "VaR95"] = -np.percentile(pnl[stock], 5)
        risk.loc[i, "VaR95_Pct"] = risk.loc[i, "VaR95"] / portfolio.loc[stock, "currentValue"]
        ub = -risk.loc[i, "VaR95"]
        # if portfolio.loc[stock, "Distribution"] == "Normal":
        #     risk.loc[i, "ES95_Pct"] = -stats.norm.expect(ub = ub, loc = mu[stock], scale = sigma[stock], conditional = True)
        # elif portfolio.loc[stock, "Distribution"] == "T":
        #     risk.loc[i, "ES95_Pct"] = -stats.t.expect(ub = ub, args=(nu[stock],), loc = mu[stock], scale = sigma[stock], conditional = True)
        risk.loc[i, "ES95"] = -np.mean(pnl[pnl[stock] <= ub][stock])
        risk.loc[i, "ES95_Pct"] = risk.loc[i, "ES95"] / portfolio.loc[stock, "currentValue"]
    total_value = sum(portfolio["currentValue"])
    # W = portfolio["currentValue"] / total_value
    # total_mu = np.mean(np.sum(W * simRet, axis = 1))
    # total_sigma = np.sqrt(W.T @ simRet.cov() @ W)
    total_VaR = -np.percentile(pnl["Total"], 5)
    row_total = risk.shape[0]
    risk.loc[row_total, "Stock"] = "Total"
    risk.loc[row_total, "VaR95"] = total_VaR
    risk.loc[row_total, "VaR95_Pct"] = total_VaR / total_value
    ub = -risk.loc[row_total, "VaR95"]
    risk.loc[row_total, "ES95"] = -np.mean(pnl[pnl["Total"] <= ub]["Total"])
    risk.loc[row_total, "ES95_Pct"] = risk.loc[row_total, "ES95"] / total_value
    return risk
