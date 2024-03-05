import pandas as pd
import sys

sys.path.append( '../RiskManagement/VaR' )
sys.path.append( '../RiskManagement/ES' )
sys.path.append( '../RiskManagement/Sim' )
sys.path.append( '../RiskManagement/Cov' )
sys.path.append( '../RiskManagement/NonPSD' )
import Return as rt
import Copula as c
import Risk as r


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

portfolio_a = portfolio.loc[portfolio["Portfolio"] == "A"]
risk_a = c.copula_risk(portfolio_a, ror)
print(risk_a.to_latex(index=False))
portfolio_b = portfolio.loc[portfolio["Portfolio"] == "B"]
risk_b = c.copula_risk(portfolio_b, ror)
print(risk_b.to_latex(index=False))
portfolio_c = portfolio.loc[portfolio["Portfolio"] == "C"]
risk_c = c.copula_risk(portfolio_c, ror)
print(risk_c.to_latex(index=False))
risk = c.copula_risk(portfolio, ror)
print(risk.to_latex(index=False))

# portfolio_a = portfolio.loc[portfolio["Portfolio"] == "A"]
# risk_a = r.risk(portfolio_a, ror)
# print(risk_a.to_latex(index=False))
# portfolio_b = portfolio.loc[portfolio["Portfolio"] == "B"]
# risk_b = r.risk(portfolio_b, ror)
# print(risk_b.to_latex(index=False))
# portfolio_c = portfolio.loc[portfolio["Portfolio"] == "C"]
# risk_c = r.risk(portfolio_c, ror)
# print(risk_c.to_latex(index=False))
# risk = r.risk(portfolio, ror)
# print(risk.to_latex(index=False))
