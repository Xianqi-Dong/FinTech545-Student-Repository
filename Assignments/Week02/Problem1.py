import pandas as pd
import numpy as np
from scipy import stats

def read_csv():
    return pd.read_csv("./Project/problem1.csv")["x"]

def first_4_moments(df):
    """a"""
    
    n = df.shape[0]
    
    #mean
    mu_hat = sum(df) / n

    #remove the mean from the sample
    sim_corrected = df - mu_hat
    sim_corrected2 = sim_corrected * sim_corrected
    cm2 = sum(sim_corrected2) / n

    #variance
    sigma2_hat = sum(sim_corrected2) / (n - 1)

    #skew
    sim_corrected3 = sim_corrected2 * sim_corrected
    skew_hat = sum(sim_corrected3) / n / pow(cm2*cm2*cm2, 0.5)

    #kurtosis
    sim_corrected4 = sim_corrected3 * sim_corrected
    kurt_hat = sum(sim_corrected4) / n / pow(cm2, 2)

    excessKurt_hat = kurt_hat - 3
    
    return mu_hat, sigma2_hat, skew_hat, excessKurt_hat


def pd_stat(df):
    """b"""
    
    mean = df.mean()
    var = df.var()
    skew = df.skew()
    kurt = df.kurt()
    
    return mean, var, skew, kurt
    
def bias_test():
    """c"""
    
    sample_size = 30
    samples = 1000
    mu_hat, sigma2_hat, skew_hat, excessKurt_hat = [0] * samples, [0] * samples, [0] * samples, [0] * samples
    mean, var, skew, kurt = [0] * samples, [0] * samples, [0] * samples, [0] * samples

    np.random.seed(1122)
    for i in range(samples):
        df = pd.Series(np.random.random(size = sample_size))
        mu_hat[i], sigma2_hat[i], skew_hat[i], excessKurt_hat[i] = first_4_moments(df)
        mean[i], var[i], skew[i], kurt[i] = pd_stat(df)

    print("------------------------------------")
    print("1st Moment:{:25f}".format(sum(mu_hat) / samples))
    print("2nd Moment:{:25f}".format(sum(sigma2_hat) / samples))
    print("3th Moment:{:25f}".format(sum(skew_hat) / samples))
    print("4th Moment:{:25f}".format(sum(excessKurt_hat) / samples))
    print("------------------------------------")

    print("------------------------------------")
    print("Mean:{:31f}".format(sum(mean) / samples))
    print("Var:{:32f}".format(sum(var) / samples))
    print("Skew:{:31f}".format(sum(skew) / samples))
    print("Kurt:{:31f}".format(sum(kurt) / samples))
    print("------------------------------------") 
    
    skew_pd = pd.Series(skew)
    kurt_pd = pd.Series(kurt)
    t_skew = (skew_pd.mean() - sum(skew_hat) / samples) / pow(skew_pd.var() / samples, 0.5)
    p_skew = (1 - stats.t.cdf(x = t_skew, df = samples)) * 2
    t_kurt = (kurt_pd.mean() - sum(excessKurt_hat) / samples) / pow(kurt_pd.var() / samples, 0.5)
    p_kurt = (1 - stats.t.cdf(x = t_kurt, df = samples)) * 2
    
    print("------------------------------------") 
    print("t_skew:{:29f}".format(t_skew))
    print("p_skew:{:29f}".format(p_skew))
    print("t_kurt:{:29f}".format(t_kurt))
    print("p_kurt:{:29f}".format(p_kurt))
    print("------------------------------------") 
    
    # r = stats.ttest_1samp(kurt_pd, sum(excessKurt_hat) / samples)
    # print("statistic:", r.__getattribute__("statistic"))
    # print("pvalue:", r.__getattribute__("pvalue"))

def process():
    print("a.")
    df = read_csv()
    mu_hat, sigma2_hat, skew_hat, excessKurt_hat = first_4_moments(df)
    print("------------------------------------")
    print("1st Moment:{:25f}".format(mu_hat))
    print("2nd Moment:{:25f}".format(sigma2_hat))
    print("3th Moment:{:25f}".format(skew_hat))
    print("4th Moment:{:25f}".format(excessKurt_hat))
    print("------------------------------------")
    
    print("b.")
    mean, var, skew, kurt = pd_stat(df)
    print("------------------------------------")
    print("Mean:{:31f}".format(mean))
    print("Var:{:32f}".format(var))
    print("Skew:{:31f}".format(skew))
    print("Kurt:{:31f}".format(kurt))
    print("------------------------------------")
    
    print("c.")
    bias_test()

if __name__ == "__main__":
    process()
    