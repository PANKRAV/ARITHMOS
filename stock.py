import yfinance as yf
import numpy as np
import json
import scipy
import matplotlib as mat
import csv
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Stock():
    def __init__(self, name : str, days : int, step : int) :
        self.name = name
        self.stock = yf.download(name, period=f"{days}d", interval=f"{step}m")
        self.file = f"{self.name}_5d_30m.csv"
        self.close = self.stock['Close'].dropna().reset_index()
        self.fit = Model(self, 80)
        
        
        
        self.stock.to_csv(self.file)

    def __str__(self):
        return (str(self.stock.head()))
    



        
class Model():
    def __init__(self, stock : Stock, split_percent : int = 80):
        self.x = np.arange(len(stock.close))
        self.y = stock.close["BMW.DE"].values
        self.split_point = (split_percent*len(self.x))//100
        self.x_train = self.x[:self.split_point]
        self.x_test = self.x[self.split_point:]
        self.y_train = self.y[:self.split_point]
        self.y_test = self.y[self.split_point:]
        self.errors = []
        self.models= {}
        print(len(self.x_train)/len(self.x_test))
    

    def fit(self, deg: int) -> None :
        coeffs = np.polyfit(self.x_train, self.y_train, deg)
        model = np.poly1d(coeffs)
        y_pred = model(self.x_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        self.errors.append((deg, mae, mse))
        self.models[deg] = model
        
        



def main():
    bmw = Stock("BMW.DE", 5, 30)
    #print(bmw)
   # print(bmw.stock.to_csv(bmw.file))
    
    




if __name__ == "__main__":
    main()
    sys.exit(0)
    