import yfinance as yf
import numpy as np
import json
import scipy
import matplotlib as mat
import csv
import sys

class Stock():
    def __init__(self, name : str, days : int, step : int) :
        self.name = name
        self.stock = yf.download(name, period=f"{days}d", interval=f"{step}m")
        self.file = f"{self.name}_5d_30m.csv"
        self.close = self.stock['Close'].dropna().reset_index()
        self.fit = Model(self)
        
        
        
        self.stock.to_csv(self.file)

    def __str__(self):
        return (str(self.stock.head()))
    



        
class Model():
    def __init__(self, stock : Stock):
        self.x = np.arange(len(stock.close))
        self.y = stock.close.values
        print(self.y)
        



def main():
    bmw = Stock("BMW.DE", 5, 30)
    #print(bmw)
   # print(bmw.stock.to_csv(bmw.file))
    model = Model(bmw)
    




if __name__ == "__main__":
    main()
    sys.exit(0)
    