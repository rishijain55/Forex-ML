import pandas as pd
import numpy as np
from datetime import datetime

class DataParser:
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.read_csv(filename, header=None, skiprows=1)
        self.data.columns = ['ticker', 'date', 'time', 'open', 'high', 'low', 'close', 'vol']
        self.data['time'] = self.data['time'].astype(str).str.zfill(6)
        self.data['date'] = pd.to_datetime(self.data['date'],format='%Y%m%d')
        self.data['time'] = pd.to_datetime(self.data['time'], format='%H%M%S').dt.time
        # Combine 'date' and 'time' columns into a single datetime column
        self.data['date'] = pd.to_datetime(self.data['date'].astype(str) + ' ' + self.data['time'].astype(str))
        # Drop the 'date' and 'time' columns
        self.data = self.data.drop(columns=['time'])
        #drop ticker
        self.data = self.data.drop(columns=['ticker'])
        # Set the index to be the datetime column
        self.data = self.data.set_index('date')
        # Sort the data
        self.data = self.data.sort_index()


    def filter_by_date(self, start_date, end_date):
        self.data = self.data.loc[start_date:end_date]
        return self.data
    
    def sample_by_time(self, time_interval, origin):
        self.data = self.data.resample(time_interval, origin=origin).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'})
        self.data.reset_index(inplace=True)
        return self.data
    
    def get_close_price_numpy_array(self, x_start, x_end, y_ind_den, y_ind_num, interval):
        close_prices = self.data['close'].to_numpy()
        X = []
        y = []
        for i in range(0, len(close_prices),interval):
            X.append(close_prices[i+x_start:i+x_end])
            y.append(close_prices[i+y_ind_num]/close_prices[i+y_ind_den])
        X = np.array(X)
        y = np.array(y)
        nan_indices = np.where(np.isnan(X).any(axis=1))
        X = np.delete(X, nan_indices, axis=0)
        y = np.delete(y, nan_indices, axis=0)
        nan_indices = np.where(np.isnan(y))
        X = np.delete(X, nan_indices, axis=0)
        y = np.delete(y, nan_indices, axis=0)

        return X, y
    
    def tranform_X_to_returns(self, X):
        X_col = X.shape[1]
        X_new = X[:, 1:X_col]/X[:, 0:X_col-1]
        X_new = np.log(X_new)
        return X_new
    
    def transform_X_with_IWMA(self, X):
        X_new = np.zeros(X.shape)
        X_col = X.shape[1]
        for i in range(X_col):
            X_new[:,i] = np.sum(X[:,0:i+1], axis=1)/(i+1)
        return X_new
    
    def normalize_data(self, X,y):
        X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        y = (y - np.mean(y, axis=0))/np.std(y, axis=0)
        return X, y




