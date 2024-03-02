import pandas as pd
import sys

sys.path.append( '../RiskManagement/VaR' )
sys.path.append( '../RiskManagement/ES' )
sys.path.append( '../RiskManagement/Sim' )
sys.path.append( '../RiskManagement/Cov' )
sys.path.append( '../RiskManagement/NonPSD' )
import Return as rt
import Copula as c


dp = pd.read_csv("./Project/DailyPrices.csv")
portfolio = pd.read_csv("./Project/portfolio.csv")
row_dp = dp.shape[0] - 1
for i in range(portfolio.shape[0]):
    stock = portfolio.loc[i, "Stock"]
    portfolio.loc[i, "Starting Price"] = dp.loc[row_dp, stock]
    if portfolio.loc[i, "Portfolio"] == "A" or portfolio.loc[i, "Portfolio"] == "B":
        portfolio.loc[i, "Distribution"] = "T"
    elif portfolio.loc[i, "Portfolio"] == "C":
        portfolio.loc[i, "Distribution"] = "Normal"
ror = rt.return_calculate(dp, "Date")
ror = ror.drop(columns=['Date'])
for stock in ror.columns:
    if stock not in list(portfolio["Stock"]):
        ror = ror.drop(columns=[stock])
risk = c.copula_risk(portfolio, ror)
print(risk)