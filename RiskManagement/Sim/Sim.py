import pandas as pd
import numpy as np
import scipy.linalg as la
import sys

sys.path.append( '../Cov' )
import CovEstimation as ce
sys.path.append( '../NonPSD' )
import non_psd as npsd


def simulateNormal(cov, times = 100000, fixMethod = None):
    if fixMethod == "near_psd":
        cov = npsd.nearPSDCov(cov)
    elif fixMethod == "higham_nearestPSD":
        cov = npsd.higham_nearestPSDCov(cov)
    sim_df = pd.DataFrame(np.random.multivariate_normal(np.zeros(cov.shape[1]), cov, times), 
                          columns = cov.columns)
    # sim_cov = ce.Cov(sim_df)
    return sim_df

def simulate_pca(cov, times = 100000, pctExp = 0.99):
    vals, vecs = la.eig(cov)
    flip = np.argsort(vals)[::-1]
    vals = vals[flip]
    vecs = vecs[:, flip] 
    tv = np.sum(vals)
    posv = np.where(vals >= 1e-8)[0]
    if pctExp < 1:
        nval = 0
        pct = 0
        for i in posv:
            pct += vals[i] / tv
            nval += 1
            if pct >= pctExp:
                break
        posv = posv[:nval]
    vals = vals[posv]
    vecs = vecs[:, posv]
    B = vecs @ np.diag(np.sqrt(vals))
    r = np.random.randn(len(vals), times)
    out = np.real_if_close((B @ r).T)
    # out_cov = ce.Cov(pd.DataFrame(out, columns = cov.columns))
    return pd.DataFrame(out, columns = cov.columns)

# np.random.seed(4)
# cin = np.full((5,5), 0.75) + np.diag(np.full(5, 0.25))
# sd = 0.1 * np.random.rand(5) ** 2
# cin = np.dot(np.dot(np.diag(sd), cin), np.diag(sd))
# print(cin)

# np.random.seed(4)
# cin = np.full((5,5), 0.75) + np.diag(np.full(5, 0.25))
# cin[1,2] = 1
# cin[2,1] = 1
# cin = np.dot(np.dot(np.diag(sd), cin), np.diag(sd))
# print(cin)

# np.random.seed(4)
# cin = np.full((5,5), 0.75) + np.diag(np.full(5, 0.25))
# cin[1,2] = 0
# cin[2,1] = 0
# cin = np.dot(np.dot(np.diag(sd), cin), np.diag(sd))