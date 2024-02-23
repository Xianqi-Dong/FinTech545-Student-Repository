import numpy as np
import pandas as pd
from scipy import stats
import math

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
    return ror

def VaR_normal_distribution_EWMA(ror, mean = 0):
    print("{:-^60s}".format("EWMA"))
    prices = pd.read_csv("./Project/DailyPrices.csv")
    VaR = pd.DataFrame(columns = ["Symbol", "5% VaR", "1% VaR", "Arithmetic 5% VaR", "Geometric 5% VaR"])
    for symbol in ror.columns[1:]:
        r = ror[symbol].ewm(com=0.94).mean()
        std = np.std(r)
        p0 = prices.loc[1, symbol]
        VaR_05 = stats.norm.ppf(0.05, loc = mean, scale = std)
        VaR_01 = stats.norm.ppf(0.01, loc = mean, scale = std)
        VaR.loc[len(VaR)] = {"Symbol": symbol, 
                             "5% VaR": str(round(VaR_05 * 100, 2)) + "%", 
                             "1% VaR": str(round(VaR_01 * 100, 2)) + "%", 
                             "Arithmetic 5% VaR": p0 * VaR_05 - p0 * mean, 
                             "Geometric 5% VaR": -(p0 - p0 * math.e ** (mean + VaR_05))}
    print(VaR)
    print("Arithmetic Total VaR:", sum(VaR["Arithmetic 5% VaR"]))
    print("Geometric Total VaR:", sum(VaR["Geometric 5% VaR"]))
    print()

def VaR_MC(ror, mean = 0):
    print("{:-^60s}".format("Monte Carlo"))
    prices = pd.read_csv("./Project/DailyPrices.csv")
    t = len(prices)
    VaR = pd.DataFrame(columns = ["Symbol", "5% VaR", "1% VaR", "Arithmetic 5% VaR", "Geometric 5% VaR"])
    for symbol in ror.columns[1:]:
        r = ror[symbol].ewm(com=0.94).mean()
        std = np.std(r)
        r = np.random.normal(mean, std, t - 1)
        std = np.std(r)
        p0 = prices.loc[1, symbol]
        VaR_05 = stats.norm.ppf(0.05, loc = mean, scale = std)
        VaR_01 = stats.norm.ppf(0.01, loc = mean, scale = std)
        VaR.loc[len(VaR)] = {"Symbol": symbol, 
                             "5% VaR": str(round(VaR_05 * 100, 2)) + "%", 
                             "1% VaR": str(round(VaR_01 * 100, 2)) + "%", 
                             "Arithmetic 5% VaR": p0 * VaR_05 - p0 * mean, 
                             "Geometric 5% VaR": -(p0 - p0 * math.e ** (mean + VaR_05))}
    print(VaR)
    print("Arithmetic Total VaR:", sum(VaR["Arithmetic 5% VaR"]))
    print("Geometric Total VaR:", sum(VaR["Geometric 5% VaR"]))
    print()

def VaR_Historical(ror, mean = 0):
    print("{:-^60s}".format("Historical"))
    prices = pd.read_csv("./Project/DailyPrices.csv")
    t = len(prices)
    VaR = pd.DataFrame(columns = ["Symbol", "5% VaR", "1% VaR", "Arithmetic 5% VaR", "Geometric 5% VaR"])
    for symbol in ror.columns[1:]:
        r = ror[symbol].ewm(com=0.94).mean()
        index = np.random.randint(1, t - 1, size = t - 1)
        new_r = np.empty(t - 1)
        j = 0
        for i in index:
            new_r[j] = r[i]
            j += 1
        std = np.std(new_r)
        p0 = prices.loc[1, symbol]
        VaR_05 = stats.norm.ppf(0.05, loc = mean, scale = std)
        VaR_01 = stats.norm.ppf(0.01, loc = mean, scale = std)
        VaR.loc[len(VaR)] = {"Symbol": symbol, 
                             "5% VaR": str(round(VaR_05 * 100, 2)) + "%", 
                             "1% VaR": str(round(VaR_01 * 100, 2)) + "%", 
                             "Arithmetic 5% VaR": p0 * VaR_05 - p0 * mean, 
                             "Geometric 5% VaR": -(p0 - p0 * math.e ** (mean + VaR_05))}
    print(VaR)
    print("Arithmetic Total VaR:", sum(VaR["Arithmetic 5% VaR"]))
    print("Geometric Total VaR:", sum(VaR["Geometric 5% VaR"]))
    print()

def process():
    ror = return_calculate()
    VaR_normal_distribution_EWMA(ror)
    VaR_MC(ror)
    VaR_Historical(ror)
    
if __name__ == "__main__":
    process()
    