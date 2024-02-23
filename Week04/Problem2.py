import numpy as np
import pandas as pd
import math
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg

def return_calculate(method = "DISCRETE"):
    prices = pd.read_csv("./Project/DailyPrices.csv")
    dateColumn = prices["Date"]
    t = len(dateColumn)
    vars = prices.columns[1:]
    ror = pd.DataFrame(columns = prices.columns)
    ror["Date"] = dateColumn[1:]

    if method == "DISCRETE":
        for i in range(1, t):
            for symbol in vars:
                ror.loc[i, symbol] = prices.loc[i, symbol] / prices.loc[i - 1, symbol] - 1
    elif method == "LOG":
        for i in range(1, t):
            for symbol in vars:
                ror.loc[i, symbol] = math.log(prices.loc[i, symbol] / prices.loc[i - 1, symbol])
    print(ror)
    return ror

def VaR_normal_distribution(r, mean = 0):
    std = np.std(r)
    VaR_05 = stats.norm.ppf(0.05, loc = mean, scale = std)
    VaR_01 = stats.norm.ppf(0.01, loc = mean, scale = std)
    print("Normal Distribution: ")
    print("5% VaR: ", round(VaR_05 * 100, 2), "%", sep = "")
    print("1% VaR: ", round(VaR_01 * 100, 2), "%", sep = "")
    print()

def VaR_normal_distribution_EWMA(r, mean = 0):
    r = r.ewm(com=0.94).mean()
    std = np.std(r)
    VaR_05 = stats.norm.ppf(0.05, loc = mean, scale = std)
    VaR_01 = stats.norm.ppf(0.01, loc = mean, scale = std)
    print("Normal Distribution with an Exponentially Weighted Variance: ")
    print("5% VaR: ", round(VaR_05 * 100, 2), "%", sep = "")
    print("1% VaR: ", round(VaR_01 * 100, 2), "%", sep = "")
    print()

def VaR_MLE_t_distribution(r, mean = 0):
    t_params = stats.t.fit(list(r), method='mle')
    dof = len(r) - 1
    std = t_params[2]
    VaR_05 = stats.t.ppf(0.05, df = dof, loc = mean, scale = std)
    VaR_01 = stats.t.ppf(0.01, df = dof, loc = mean, scale = std)
    print("MLE fitted T Distribution: ")
    print("5% VaR: ", round(VaR_05 * 100, 2), "%", sep = "")
    print("1% VaR: ", round(VaR_01 * 100, 2), "%", sep = "")
    print()

def VaR_fitted_a1_model(r):
    ar = AutoReg(list(r), lags = 1).fit()
    r = r * ar.params[1] + ar.params[0]
    VaR_05 = np.percentile(r, 0.05)
    VaR_01 = np.percentile(r, 0.01)
    print("Fitte AR(1) Model: ")
    print("5% VaR: ", round(VaR_05 * 100, 2), "%", sep = "")
    print("1% VaR: ", round(VaR_01 * 100, 2), "%", sep = "")
    print()

def Var_historic_simulation(r):
    VaR_05 = np.percentile(r, 0.05)
    VaR_01 = np.percentile(r, 0.01)
    print("Historic Simulation: ")
    print("5% VaR: ", round(VaR_05 * 100, 2), "%", sep = "")
    print("1% VaR: ", round(VaR_01 * 100, 2), "%", sep = "")
    print()

def process():
    meta = return_calculate()["META"]
    VaR_normal_distribution(meta)
    VaR_normal_distribution_EWMA(meta)
    VaR_MLE_t_distribution(meta)
    VaR_fitted_a1_model(meta)
    Var_historic_simulation(meta)
    
if __name__ == "__main__":
    process()