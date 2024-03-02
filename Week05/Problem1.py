import pandas as pd
import numpy as np
from scipy import stats
import sys

sys.path.append( '../RiskManagement/VaR' )
sys.path.append( '../RiskManagement/ES' )
sys.path.append( '../RiskManagement/Sim' )
sys.path.append( '../RiskManagement/Cov' )
sys.path.append( '../RiskManagement/NonPSD' )
import CovEstimation as ce
import non_psd as npsd
import Sim as s
import Return as rt
import VaR as v
import ES as es
import Copula as c


testfiles_folder = "../FinTech545_Spring2024/testfiles/"
## Question 1

# test function
def test(cout, filename, precision):
    filepath = testfiles_folder + 'data/' + filename + '.csv'
    df = pd.read_csv(filepath)
    diff = cout.reset_index(drop=True) - df.reset_index(drop=True)
    tol = 0.1 ** precision
    exceed_tol = diff >= tol
    print(filename, " " , exceed_tol.sum().sum() == 0)
    
def test1(cout, filename, precision):
    filepath = testfiles_folder + 'data/' + filename + '.csv'
    df = pd.read_csv(filepath)
    cout = cout.reset_index(drop=True).round(precision)
    # print(cout)
    df = df.reset_index(drop=True).round(precision)
    # print(df)
    print(filename, " ", cout.equals(df))

def test2(cout, filename, precision):
    filepath = testfiles_folder + 'data/' + filename + '.csv'
    df = pd.read_csv(filepath)
    cout = cout.reset_index(drop=True).round(precision)
    # print(cout)
    df = df.reset_index(drop=True).round(precision)
    # print(df)
    print(filename, " ", cout.equals(df))

# Test 1 - missing covariance calculations
# Generate some random numbers with missing values.

x = pd.read_csv(testfiles_folder + 'data/test1.csv')
# 1.1 Skip Missing rows - Covariance
cout = ce.Cov(x)
test(cout, 'testout_1.1', 9)
# 1.2 Skip Missing rows - Correlation
cout = ce.Cor(x)
test(cout, 'testout_1.2', 9)
# 1.3 Pairwise - Covariance
cout = ce.Cov(x, False)
test(cout, 'testout_1.3', 9)
# 1.4 Pairwise - Correlation
cout = ce.Cor(x, False)
test(cout, 'testout_1.4', 9)

# Test 2 - EW Covariance
x = pd.read_csv(testfiles_folder + "data/test2.csv")
# 2.1 EW Covariance λ=0.97
lam = 0.97
cout = ce.ewCov(x, lam)
test(cout, 'testout_2.1', 9)
# 2.2 EW Correlation λ=0.94
lam = 0.94
cout = ce.ewCor(x, lam)
test(cout, 'testout_2.2', 9)
# 2.3 EW Cov w/ EW Var(λ=0.94) EW Correlation(λ=0.97)
cout = ce.ewCovCor(x, 0.97, 0.94)
test(cout, 'testout_2.3', 9)

# Test 3 - non-psd matrices
x = pd.read_csv(testfiles_folder + "data/testout_1.3.csv")
# 3.1 near_psd covariance
cout = npsd.nearPSDCov(x)
test(cout, 'testout_3.1', 9)
# 3.2 near_psd Correlation
x = pd.read_csv(testfiles_folder + "data/testout_1.4.csv")
cout = npsd.nearPSDCor(x)
test(cout, 'testout_3.2', 9)
# 3.3 Higham covariance
x = pd.read_csv(testfiles_folder + "data/testout_1.3.csv")
cout = npsd.higham_nearestPSDCov(x)
test(cout, 'testout_3.3', 9)
# 3.4 Higham Correlation
x = pd.read_csv(testfiles_folder + "data/testout_1.4.csv")
cout = npsd.higham_nearestPSDCor(x)
test(cout, 'testout_3.4', 9)

# Test 4 - cholesky factorization
x = pd.read_csv(testfiles_folder + 'data/testout_3.1.csv')
cout = npsd.chol_psd(x)
test(cout, 'testout_4.1', 6)

# Test 5 - Normal Simulation

# 5.1 PD Input
x = pd.read_csv(testfiles_folder + 'data/test5_1.csv')
cout = ce.Cov(s.simulateNormal(x, 100000))
test(cout, 'testout_5.1', 3)
# txt = " Cannot compare simulation results"
# print('testout_5.1', txt)
# print('Simulated Cov Mat:')
# print(cout)
# 5.2 PSD Input
x = pd.read_csv(testfiles_folder + 'data/test5_2.csv')
cout = ce.Cov(s.simulateNormal(x, 100000))
test(cout, 'testout_5.2', 3)
# 5.3 nonPSD Input, near_psd fix
x = pd.read_csv(testfiles_folder + 'data/test5_3.csv')
cout = ce.Cov(s.simulateNormal(x, 100000, 'near_psd'))
test(cout, 'testout_5.3', 3)
# 5.4 nonPSD Input Higham Fix
x = pd.read_csv(testfiles_folder + 'data/test5_3.csv')
cout = ce.Cov(s.simulateNormal(x, 100000, 'higham_nearestPSD'))
test(cout, 'testout_5.4', 3)
# 5.5 PSD Input - PCA Simulation
x = pd.read_csv(testfiles_folder + 'data/test5_2.csv')
cout = ce.Cov(s.simulate_pca(x, 100000, 0.99))
test(cout, 'testout_5.5', 3)

# Test 6 - Returns
price = pd.read_csv(testfiles_folder + "data/test6.csv")
# 6.1 Arithmetic returns
rout = rt.return_calculate(price, "Date", "DISCRETE")
test1(rout, 'test6_1', 9)
# 6.2 Log returns
rout = rt.return_calculate(price, "Date", "LOG")
test1(rout, 'test6_2', 9)

# Test 7 - Fit Distribution
# Data simulation

# 7.1 Fit Normal Distribution
x = pd.read_csv(testfiles_folder + "data/test7_1.csv")
cout = v.fit_normal(x)
test1(cout, 'testout7_1', 3)
# 7.2 Fit TDist
x = pd.read_csv(testfiles_folder + "data/test7_2.csv")
cout = v.fit_general_t(x)
test1(cout, 'testout7_2', 4)
# 7.3 Fit T Regression
x = pd.read_csv(testfiles_folder + "data/test7_3.csv")
cout = v.fit_regression_t(x)
test1(cout, 'testout7_3', 3)

# Test 8 - VaR

# 8.1 VaR Normal
x = pd.read_csv(testfiles_folder + "data/test7_1.csv")
cout = v.VaR_normal_distribution(x, 0.05)
test1(cout, 'testout8_1', 3)

# 8.2 VaR TDist
x = pd.read_csv(testfiles_folder + "data/test7_2.csv")
cout = v.VaR_t_distribution(x, 0.05)
test1(cout, 'testout8_2', 3)

# 8.3 VaR Simulation
x = pd.read_csv(testfiles_folder + "data/test7_2.csv")
cout = v.VaR_simulation(x, 0.05)
test1(cout, 'testout8_3', 2)

# 8.4 ES Normal
x = pd.read_csv(testfiles_folder + "data/test7_1.csv")
cout = es.ES_normal_distribution(x, 0.05)
test1(cout, 'testout8_4', 3)

# 8.5 ES TDist
x = pd.read_csv(testfiles_folder + "data/test7_2.csv")
cout = es.ES_t_distribution(x, 0.05)
test1(cout, 'testout8_5', 3)

# 8.6 VaR Simulation
x = pd.read_csv(testfiles_folder + "data/test7_2.csv")
cout = es.ES_simulation(x, 0.05)
test1(cout, 'testout8_6', 2)

# Test 9

# 9.1
portfolio = pd.read_csv(testfiles_folder + "data/test9_1_portfolio.csv")
ror = pd.read_csv(testfiles_folder + "data/test9_1_returns.csv")
cout = c.copula_risk(portfolio, ror)
print(cout)
