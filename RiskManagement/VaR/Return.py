import pandas as pd
import numpy as np


def return_calculate(prices, str_column, method = "DISCRETE"):
    date = prices.loc[1:, str_column].reset_index()[str_column]
    prices = prices.drop(columns = [str_column])
    if method == "DISCRETE":
        ror = np.matrix(prices.iloc[1:]) / np.matrix(prices.iloc[:-1]) - 1
    elif method == "LOG":
        ror = np.log(np.matrix(prices.iloc[1:]) / np.matrix(prices.iloc[:-1]))
    ror = pd.DataFrame(ror, columns = prices.columns)
    ror = pd.concat([date, ror], axis=1)
    return ror
