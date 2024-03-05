import pandas as pd
import sys

sys.path.append( '../RiskManagement/VaR' )
sys.path.append( '../RiskManagement/ES' )
sys.path.append( '../RiskManagement/Sim' )
sys.path.append( '../RiskManagement/Cov' )
sys.path.append( '../RiskManagement/NonPSD' )
import VaR as v
import ES as es


df = pd.read_csv("./Project/problem1.csv")
VaR_a = v.VaR_normal_distribution_ewvar(df, 0.05, 0.97)
ES_a = es.ES_normal_distribution_ewvar(df, 0.05, 0.97)
print(VaR_a)
print(ES_a)
VaR_b = v.VaR_t_distribution(df, 0.05)
ES_b = es.ES_t_distribution(df, 0.05)
print(VaR_b)
print(ES_b)
VaR_c = v.VaR_historic(df, 0.05, 50, 100000)
ES_c = es.ES_historic(df, 0.05, 50, 100000)
print(VaR_c)
print(ES_c)

# # draw the plot
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm, t
# u, sig, var = v.VaR_normal_distribution_ewvar(df, 0.05, 0.97)
# x = np.linspace(u - 3*sig, u + 3*sig, 50)
# y = norm.pdf(x, loc = u, scale = sig)
# plt.plot(x, y, "g", linewidth=2, color='green', label="Noraml")
# plt.axvline(-var, color='green')

# u, sig, nu, var = v.VaR_t_distribution(df, 0.05)
# x = np.linspace(u - 3*sig, u + 3*sig, 50)
# y = t.pdf(x, df = nu, loc = u, scale = sig)
# plt.plot(x, y, "g", linewidth=2, color='red', label="T")
# plt.axvline(-var, color='red')

# plt.show()