import pandas as pd
import numpy as np


def Cov(df, skip_miss = True):
    if skip_miss == True:
        df = df.dropna()
    cov = df.cov()
    return cov

def Cor(df, skip_miss = True):
    if skip_miss == True:
        df = df.dropna()
    cor = df.corr()
    return cor

def CovtoCor(cov):
    var = np.diag(cov)
    var = var.astype('float64') 
    sd = np.sqrt(var)
    cor =  pd.DataFrame(np.dot(np.dot(np.diag(1 / sd), cov), np.diag(1 / sd)), 
                        columns = cov.columns, index = cov.columns)
    return sd, cor

def CortoCov(sd, cor):
    cov =  pd.DataFrame(np.dot(np.dot(np.diag(sd), cor), np.diag(sd)), 
                        columns = cor.columns, index = cor.columns)
    return cov

def ewCov(df, lamb):
    ew_cov = pd.DataFrame(columns = df.columns, index = df.columns)
    w = []
    for row in range(df.shape[0]):
        w.append((1 - lamb) * lamb ** row)
    sum_w = sum(w)
    w = [x / sum_w for x in w]
    w.reverse()
    for col1 in range(df.shape[1]):
        mu1 = np.mean(df.iloc[:, col1])
        for col2 in range(col1, df.shape[1]):
            mu2 = np.mean(df.iloc[:, col2])
            sigma2 = 0
            for row in range(df.shape[0]):
                sigma2 += w[row] * (df.iloc[row, col1] - mu1) *(df.iloc[row, col2] - mu2)
            ew_cov.iloc[col1, col2] = sigma2
            ew_cov.iloc[col2, col1] = sigma2
    return ew_cov

def ewCor(df, lamb):
    ew_cov = ewCov(df, lamb)
    sd, ew_cor = CovtoCor(ew_cov)
    return ew_cor

def ewCovCor(df, cov_lamb, cor_lamb):
    ew_cov = ewCov(df, cov_lamb)
    ew_var = np.diag(ew_cov)
    ew_var = ew_var.astype('float64') 
    ew_sd = np.sqrt(ew_var)
    ew_cor = ewCor(df, cor_lamb)
    ew_cov_cor = pd.DataFrame(np.dot(np.dot(np.diag(ew_sd), ew_cor), np.diag(ew_sd)), 
                              columns = df.columns, index = df.columns)
    return ew_cov_cor
