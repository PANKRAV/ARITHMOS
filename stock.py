import yfinance as yf
import numpy as np
import json
import scipy
import matplotlib.pyplot as plt
import csv
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Stock():
    def __init__(self, name : str, days : int, step : int) :
        self.name = name
        self.stock = yf.download(name, period=f"{days}d", interval=f"{step}m")
        self.file = f"{self.name}_5d_30m.csv"
        self.close = self.stock['Close'].dropna().reset_index()
        self.model = Model(self, 80)
        
        
        
        self.stock.to_csv(self.file)

    def __str__(self):
        return (str(self.stock.head()))

    def predict(self):
        ...
    



        
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
        self.errors.append((deg, mae, mse, coeffs))
        self.models[deg] = model
    

    def predict(self, model, N) :
        len_x = len(self.x)
        model = np.poly1d(model)
        prediction_x = np.arange(len_x, len_x + N)
        prediction_y = model(prediction_x)
        print(prediction_y)
        x_fit = np.concatenate([self.x, prediction_y])
        y_fit = model(x_fit)

        plt.figure(figsize=(10, 5))
        plt.plot(self.x, self.y, label="Πραγματικά Δεδομένα")
        plt.plot(x_fit, y_fit, label="Πολυωνυμική Καμπύλη")
        plt.plot(prediction_x, prediction_y, 'r--', label="Πρόβλεψη 24 ωρών (48 βήματα)")
        plt.xlabel("Χρονική στιγμή (30 λεπτά)")
        plt.ylabel("Τιμή μετοχής")
        plt.title("Πρόβλεψη επόμενης ημέρας")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        
        



def main():
    bmw = Stock("BMW.DE", 5, 30)
    #print(bmw)
    for deg in range(1, 4):
        bmw.model.fit(deg)

   #bmw.model.predict(bmw.model.errors[2][3], 48) καμπυλη πολυωνυμου δευτερου βαθμου

    best_model = min(bmw.model.errors, key=lambda t: t[1])
    bmw.model.predict(best_model[3], 48)
    
        
            
    print(bmw.model.errors)
    
    




if __name__ == "__main__":
    main()
    sys.exit(0)
    