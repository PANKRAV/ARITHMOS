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

        

def main():
    pass



if __name__ == "__main__":
    main()
    sys.exit(0)
    