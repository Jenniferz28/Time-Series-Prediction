import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from math import sqrt
from sklearn.metrics import mean_squared_error

def data_import(filename, timeformat):
    dateparse = lambda dates: pd.datetime.strptime(dates, timeformat) 
    data = pd.read_csv(filename, parse_dates=['Datetime'], index_col='Datetime',date_parser=dateparse)
    # plt.plot(data, label='Dataset')
    # plt.show()
    return data

def data_split(data, train_percentage):
    train, test = data[0:len(data)*train_percentage/100], data[len(data)*train_percentage/100:]
    pd.DataFrame(train).to_csv('train.csv')
    pd.DataFrame(test).to_csv('demo_test.csv')
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.show()
    return train, test

def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')    # Perform Dickey-Fuller test:
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

    if dftest[0] < dftest[4]['1%']:                 # p-value < Critical value (1%)
        return True, dftest[2]
    else:
        return False, dftest[2]

def differencing(data):
    data_diff = data - data.shift()
    data_diff = data_diff.dropna()

    plt.plot(data_diff)
    plt.show()

    return data_diff

def acf_pacf(data,nlags):
    plot_pacf(data, method='ols', lags=nlags)
    plt.suptitle('lags: {}'.format(nlags))
    plt.show()

    plot_acf(data, lags=nlags)
    plt.suptitle('lags: {}'.format(nlags))
    plt.show()

def order_combination(p, d, q):
    minRMSE = 999999999.00
    x = 8
    # for x in range(p + 1):
    for y in range(1, q + 1):
        if(x > 0 or y > 1):
            rmse = arima_predict(train, test, 1, x, d, y) 
            print(x,d,y-1,rmse)
            if rmse < minRMSE :
                minRMSE = rmse 
                bestp = x
                bestq = y
    # print(bestp, d, bestq-1)
    return bestp, d, bestq

def arima_predict(train, test, n, p, d ,q):
    predictions = []
    q = q+1
    for t in range(0,len(test),n):
        result = ARMA(train, p, q, n)
        predictions = np.append(predictions, result)
        train = np.append(train, test[t:t+n])
    rmse = sqrt(mean_squared_error(test, predictions))
    # plot
    plt.plot(test.index, test.values, color='blue', label='Test')
    plt.plot(test.index, predictions, color='red', label='ARIMA')
    plt.show()
    pd.DataFrame(predictions).to_csv('arima_final.csv')
    return rmse

def moving_average(data, period):
    buffer = []
    for i in range(period, len(data)):
        buffer.append(np.copy(data[i - period : i]).mean())
    return buffer

def get_ma_errors(data, q):
    """
    q = the order of the moving-average
    """
    ma = moving_average(data, q)
    errors = []
    for n in range(q-1, len(ma)):
        error = data[n] - ma[n]
        errors.append(error)
    return errors

def get_lags_matrix(data, n):
    """
    n = number of lag used
    returns the matrix of lags
    """
    lags_matrix = []
    for i in range (n+1):
        lags_matrix.append(data[n-i:len(data)-i])
    return lags_matrix

def get_determinant(lags_matrix, n):
    """
    returns the determinant of matrix A, A0, A1, ..., An
    """
    n += 1 # plus one for y
    determinant = []
    # Matrix A
    matrix = [[0] * n for i in range(n)]
    for i in range(1, n):
        for j in range(1,n):
            matrix[i][j] = sum(np.multiply(lags_matrix[i],lags_matrix[j]))
    for i in range(1, n):
        matrix[0][i] = sum(lags_matrix[i])
        matrix[i][0] = sum(lags_matrix[i])
    matrix[0][0] = n-1
    determinant.append(np.linalg.det(matrix))

    # Matrix A0
    matrix0 = [[0] * n for i in range(n)]
    for i in range(1, n):
        for j in range(1,n):
            matrix0[i][j] = sum(np.multiply(lags_matrix[i],lags_matrix[j]))
    for i in range(0, n):
        matrix0[i][0] = sum(np.multiply(lags_matrix[0],lags_matrix[i]))
        matrix0[0][i] = sum(lags_matrix[i])
    determinant.append(np.linalg.det(matrix0))

    # Matrix A1..An
    for i in range(1,n):
        matrixA = np.copy(matrix)
        for j in range(n):
            matrixA[j][i] = matrix0[j][0]
        determinant.append(np.linalg.det(matrixA))

    return determinant

def get_coefficient(data, n):
    """
    returns the array of a, b1, b2, .. bn
    """
    lags_matrix = get_lags_matrix(data, n)
    determinant = get_determinant(lags_matrix, n)
    b = []
    for i in range(1,len(determinant)):
        b.append(np.nan_to_num(determinant[i]/determinant[0]))
    return b

def AR(train, p, step = 1):
    ar_coeff = get_coefficient(train,p)
    result = []
    for i in range(step):
        prediction = ar_coeff[0]
        for j in range(1,len(ar_coeff)):
            prediction+= ar_coeff[j]*train[-j]
        result = np.append(result,prediction)
        train = np.append(train,prediction)
    # pd.DataFrame(result).to_csv('AR.csv')
    return result

def MA(train, q, step):
    result = []
    for i in range(step):
        prediction = np.copy(train[-q:]).mean()
        result = np.append(result,prediction)
        train = np.append(train,prediction)
    # pd.DataFrame(result).to_csv('MA.csv')
    return result

def ARMA(train, p, q, step = 1):
    """
    p = the order (number of time lags)
    q = the order of the moving-average
    step = the number of future values to predict

    """
    if q==0:
        return AR(train, p, step)
    elif p==0:
        return MA(train, q, step)
    else: 
        ar_coeff = get_coefficient(train,p)
        # print ar_coeff
        ma_errors = get_ma_errors(train,q)
        ma_coeff = get_coefficient(ma_errors,q)
        result = []
        for i in range(step):
            prediction = ar_coeff[0]
            for j in range(1,len(ar_coeff)):
                prediction += ar_coeff[j]*train[-j]
            for j in range(1,len(ma_coeff)):
                prediction -= ma_coeff[j]*ma_errors[-j]
            result = np.append(result,prediction)
            train = np.append(train,prediction)
            ma_errors = get_ma_errors(train,q)
        return result

data = data_import('aep_monthly.csv','%Y-%m-%d')            # import data with given timeformat
train, test = data_split(data,70)                           # split train and test data with data train percentage is 70%
train = train.AEP_MW.values
test = test[:48]

# isStationary, nlags = test_stationarity(train)              # return stationarity test and number of lags used
# d = 0                                                       # initial number of d represent number of differencing
# train_diff = train
# while isStationary == False :                               # if not stationary
#     train_diff = np.diff(train_diff)                        # return differenced data
#     d += 1        
#     isStationary, nlags = test_stationarity(train_diff)  
# print "Differencing Process:", d 

# plt.plot(train)
# plt.show()

# acf_pacf(train,nlags)                                       # show acf and pacf graphic plot for given data series
# p, d, q = order_combination(8, 2, 30) 
rmse = arima_predict(train, test, 1, 8, 2 ,0)
# print rmse
