import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import optimize
from scipy import stats
import statsmodels.api as sm
from matplotlib import pyplot as plt

def read_csv():
    X = pd.read_csv("./Project/problem2.csv")["x"]
    Y = pd.read_csv("./Project/problem2.csv")["y"]
    return X, Y

def OLS(X, Y):
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    sigma_ols = np.std(results.resid)
    a = results.params.const + results.params.x * np.mean(X) -np.mean(Y)
    print(a)
    print("OLS Beta:  ", [results.params.const, results.params.x])
    print("OLS std: ", sigma_ols)

def R2(X, Y, beta):
    rr = Y - np.dot(X, beta)
    SSR = np.dot(rr, rr)
    tt = Y - np.mean(Y)
    SST = np.dot(tt, tt)
    r2 = 1 - SSR / SST
    return r2

def MLE(X, Y):
    X = sm.add_constant(X)
    def ll_n(params):
        s = params[0]
        b = params[1:]
        n = len(Y)
        e = Y - np.dot(X, b)
        s2 = s * s
        # log likelihood of n
        ll = -n / 2 * np.log(s2 * 2 * np.pi) - np.dot(e, e) / (2 * s2)
        return -ll  # return negative ll because we minimize

    res = optimize.minimize(ll_n, [1,0,0], options={"disp": True})
    sigma_mle, *beta_mle = res.x
    a = beta_mle[0] + beta_mle[1] * np.mean(X) -np.mean(Y)
    print(a)
    print("MLE Beta: ", beta_mle)
    print("MLE std: ", sigma_mle)
    print("R^2: ", R2(X, Y, beta_mle))
    return beta_mle
   
def MLE_t(X, Y):
    X = sm.add_constant(X)
    
    def ll_t(params):
        df = params[0]
        s = params[1]
        b = params[2:]
        e = Y - np.dot(X, b)
        ll = np.sum(stats.t.logpdf(e, df=df, loc=0, scale=s))
        return -ll

    res = optimize.minimize(ll_t, [1,1,0,0], options={"disp": True})
    sigma_mle, *beta_mle = res.x
    print("MLE_t Beta: ", beta_mle[1:])
    print("MLE_t std: ", sigma_mle)
    print("R^2: ", R2(X, Y, beta_mle[1:]))

def expected_value():
    X = pd.read_csv("./Project/problem2_x.csv")["x1"].to_numpy()
    Y = pd.read_csv("./Project/problem2_x.csv")["x2"].to_numpy()
    X1 = pd.read_csv("./Project/problem2_x1.csv")["x1"].to_numpy()
    beta = MLE(X, Y)
    X2 = beta[0] + beta[1] * X1 
    X_bar = np.mean(X)
    cm = beta[0] + beta[1] * (X1 - X_bar)
    cv = np.var(Y) - beta[1] * np.cov(X, Y)[0, 1]    
    interval_95 = 1.96 * np.sqrt(cv)

    plt.figure(figsize=(10, 6))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Expected Value')
    plt.scatter(X1.flatten(), cm.flatten(), color='blue', label='Expected X2')
    plt.fill_between(X1.flatten(), cm.flatten() - interval_95, cm.flatten() + interval_95, color='gray', alpha=0.5)
    plt.show()
    pdf(X1)
    pdf(X2)

def pdf(X):
    plt.plot(np.sort(X), norm.pdf(np.sort(X)))
    plt.show()
    
def process():
    X, Y = read_csv()
    print("a.")
    OLS(X, Y)
    MLE(X, Y)
    print("--------------------------------------")
    print("b.")
    MLE_t(X, Y)
    print("--------------------------------------")
    print("c.")
    expected_value()
    
if __name__ == "__main__":
    process()