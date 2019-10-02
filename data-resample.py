import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np 
 

# Import data
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
data = pd.read_csv('AEP_hourly.csv', parse_dates=['Datetime'], index_col='Datetime',date_parser=dateparse)

aep = data.resample('M').sum() 
pd.DataFrame(aep).to_csv('aep_monthly.csv')


