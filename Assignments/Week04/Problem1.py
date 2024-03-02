import numpy as np
import math

def classical_brownian_motion(p0, mu, sigma, t):
    """P_t = P_{t-1} + r_t
    """
    r = np.random.normal(mu, sigma, t)
    pt = p0
    for i in range(t):
        pt = pt + r[i]
    return pt

def arithmetic_return_system(p0, mu, sigma, t):
    """P_t = P_{t-1}(1 + r_t)
    """
    r = np.random.normal(mu, sigma, t)
    pt = p0
    for i in range(t):
        pt = pt * (1 + r[i])
    return pt

def geometric_brownian_motion(p0, mu, sigma, t):
    """P_t = P_{t-1}e^{r_t}
    """
    r = np.random.normal(mu, sigma, t)
    pt = p0
    for i in range(t):
        pt = pt * (math.e ** r[i])
    return pt

def print_mean_and_std(pt):
    print("mean: ", np.mean(pt))
    print("std: ", np.std(pt))
  
def simulation(p0, mu, sigma, t, n):
    pt_cbm = np.empty(n)
    pt_ars = np.empty(n)
    pt_gbm = np.empty(n)
    for i in range(n):
        pt_cbm[i] = classical_brownian_motion(p0, mu, sigma, t)
        pt_ars[i] = arithmetic_return_system(p0, mu, sigma, t)
        pt_gbm[i] = geometric_brownian_motion(p0, mu, sigma, t)
    print("Classical Brownian Motion: ")
    print_mean_and_std(pt_cbm)
    print("Expectation: ")
    mean = p0
    std = sigma
    print("mean: ", mean, "\tdifference: ", np.mean(pt_cbm) - mean)
    print("std: ", std, "\tdifference: ", np.std(pt_cbm) - std)
    print()
    
    
    print("Arithmetic Return System: ")
    print_mean_and_std(pt_ars)
    print("Expectation: ")
    mean = p0
    std = p0 * sigma
    print("mean: ", mean, "\tdifference: ", np.mean(pt_ars) - mean)
    print("std: ", std, "\tdifference: ", np.std(pt_ars) - std)
    print()
    
    print("Geometric Brownian Motion: ")
    print_mean_and_std(pt_gbm)
    print("Expectation: ")
    mean = p0 * math.e ** (mu + 0.5 * sigma ** 2)
    std = math.sqrt(p0 ** 2 * (math.e ** (sigma ** 2) - 1) * math.e ** (2 * mu + sigma ** 2))
    print("mean: ", mean, "\tdifference: ", np.mean(pt_gbm) - mean)
    print("std: ", std, "\tdifference: ", np.std(pt_gbm) - std)
    print()

def process():
    p0 = 10
    mu = 0
    sigma = 0.01
    t = 1
    n = 1000
    simulation(p0, mu, sigma, t, n)

if __name__ == "__main__":
    process()
    