import yfinance as yf
import numpy as np
import json
from scipy.integrate import simpson, trapezoid
import matplotlib.pyplot as plt
import csv
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
from functools import cache

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
        self.coeffs = {}
        print(len(self.x_train)/len(self.x_test))

    @cache
    def calculate_model(self, deg: int) -> np.poly1d:
        coeffs = np.polyfit(self.x_train, self.y_train, deg)
        self.coeffs[deg] = coeffs
        self.models[deg] = np.poly1d(coeffs)
    

    def fit(self, deg: int, model) -> None :
        y_pred = model(self.x_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        self.errors.append((deg, mae, mse, self.coeffs[deg]))
        self.models[deg] = model
    
#
    def predict(self, model, N) :
        len_x = len(self.x)
        model = np.poly1d(model)
        prediction_x = np.arange(len_x, len_x + N)
        prediction_y = model(prediction_x)
        print(prediction_y)
        x_fit = np.concatenate([self.x, prediction_y])
        y_fit = model(x_fit)

        plt.figure(figsize=(10, 5))
        plt.plot(self.x, self.y, label="Real World Data")
        plt.plot(x_fit, y_fit, label="Best Polynomial Fit")
        plt.plot(prediction_x, prediction_y, 'r--', label="Model Next Day Prediction")
        plt.xlabel("dt (30 minutes)")
        plt.ylabel("Stock Value")
        plt.title("Next Day Prediction")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def compute_meanvalues(self, deg):
        y_fit = self.models[deg](self.x)
        area_trapz = trapezoid(y_fit, self.x)
        area_simpson = simpson(y_fit, self.x)
        mean_trapz = area_trapz / len(self.x)
        mean_simpson = area_simpson / len(self.x)
        return mean_trapz, mean_simpson
    
    @staticmethod
    def aitken_neville(x_points, y_points, x):
        n = len(x_points)
        p = np.copy(y_points).astype(float)
        for k in range(1, n):
            for i in range(n - k):
                numerator = (x - x_points[i + k]) * p[i] + (x_points[i] - x) * p[i + 1]
                denominator = x_points[i] - x_points[i + k]
                p[i] = numerator / denominator
        return p[0]
    

    @staticmethod
    def newton_divided_diff(x_points, y_points):
        n = len(x_points)
        coef = np.copy(y_points).astype(float)
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                coef[i] = (coef[i] - coef[i - 1]) / (x_points[i] - x_points[i - j])
        return coef
    

    
    def detect_anomalies(self, window_size = 5) :
        anomalies = []

        for start in range(len(self.x) - window_size):
            x_window = self.x[start:start + window_size]
            y_window = self.y[start:start + window_size]

            x_next = self.x[start + window_size]
            actual = self.y[start + window_size]


            aitken_pred = self.aitken_neville(x_window, y_window, x_next)


            coef = self.newton_divided_diff(x_window, y_window)
            newton_pred = self.newton_polynomial(coef, x_window, x_next)

            delta = self.anomaly_threshold(y_window)

            deviation_aitken = abs(actual - aitken_pred)
            deviation_newton = abs(actual - newton_pred)

            anomaly_aitken = deviation_aitken > delta
            anomaly_newton = deviation_newton > delta

            anomalies.append(
                {
                "start_index": start,
                "actual": actual,
                "aitken_pred": aitken_pred,
                "newton_pred": newton_pred,
                "delta": delta,
                "deviation_aitken": deviation_aitken,
                "deviation_newton": deviation_newton,
                "anomaly_aitken": anomaly_aitken,
                "anomaly_newton": anomaly_newton,
            }
            )

        return anomalies   

    @staticmethod
    def newton_polynomial(coef, x_points, x):
        n = len(coef)
        result = coef[n - 1]
        for i in range(n - 2, -1, -1):
            result = result * (x - x_points[i]) + coef[i]
        return result
    
    @staticmethod
    def anomaly_threshold(values):
        range_val = np.max(values) - np.min(values)
        std_val = np.std(values)
        delta = 0.05 * range_val + 0.5 * std_val
        return delta

def main():
    bmw = Stock("BMW.DE", 5, 30)
    mean = {
        "Simp":{},
        "Trapz":{}
    }
    for deg in range(1, 4):
        bmw.model.calculate_model(deg)
        bmw.model.fit(deg, bmw.model.models[deg])
        val = bmw.model.compute_meanvalues(deg)
        mean["Simp"][deg] = val[0]
        mean["Trapz"][deg] = val[1]
    
    print(mean)

    #bmw.model.predict(bmw.model.errors[1][3], 48) #καμπυλη πολυωνυμου δευτερου βαθμου

    bestcalculate_model = min(bmw.model.errors, key=lambda t: t[1])
    bmw.model.predict(bestcalculate_model[3], 48)




    print(bmw.model.errors)


    anomalies = bmw.model.detect_anomalies()
    for a in anomalies:
        print(f"Διάστημα {a['start_index']} - {a['start_index'] + 4}:")
        print(f"  Πραγματική τιμή: {a['actual']:.4f}")
        print(f"  Πρόβλεψη Aitken: {a['aitken_pred']:.4f}, Ανωμαλία: {a['anomaly_aitken']}")
        print(f"  Πρόβλεψη Newton: {a['newton_pred']:.4f}, Ανωμαλία: {a['anomaly_newton']}")
        print(f"  Κατώφλι δ: {a['delta']:.4f}\n")
    
    




if __name__ == "__main__":
    main()
    sys.exit(0)
    