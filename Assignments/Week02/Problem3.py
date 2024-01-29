import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

def read_csv():
    return pd.read_csv("./Project/problem3.csv")["x"]
    
def arima():
    series = read_csv()
    sm.graphics.tsa.plot_acf(series, lags=40) 
    sm.graphics.tsa.plot_pacf(series, lags=40, method="ywm")
    pyplot.show()
    
    ar1 = ARIMA(series, order=(1,0,0)).fit()
    ar3 = ARIMA(series, order=(3,0,0)).fit()
    ma1 = ARIMA(series, order=(0,1,0)).fit()
    ma3 = ARIMA(series, order=(0,3,0)).fit()
    pyplot.subplot(2, 2, 1)
    pyplot.title("AR1")
    pyplot.plot(pd.DataFrame(ar1.resid))
    
    pyplot.subplot(2, 2, 2)
    pyplot.title("AR3")
    pyplot.plot(pd.DataFrame(ar3.resid))
    
    pyplot.subplot(2, 2, 3)
    pyplot.title("MA1")
    pyplot.plot(pd.DataFrame(ma1.resid))
    
    pyplot.subplot(2, 2, 4)
    pyplot.title("MA3")
    pyplot.plot(pd.DataFrame(ma3.resid))
    pyplot.show()
    
    sst = series.var() * len(series)
    print("         AIC            R^2")
    print("AR1    {:f}     {:f}".format(ar1.aic, 1 - ar1.sse / sst))
    print("AR3    {:f}     {:f}".format(ar3.aic, 1 - ar3.sse / sst))
    print("MA1    {:f}    {:f}".format(ma1.aic, 1 - ma1.sse / sst))
    print("MA3    {:f}    {:f}".format(ma3.aic, 1 - ma3.sse / sst))

    
if __name__ == "__main__":
    arima()
