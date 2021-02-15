# -*- coding: utf-8 -*-
import numpy as np
import yaml
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from Modules.moving_mean import MovingMeanAlgorithm
from Modules.airPLS import airPLS_Algorithm


def show_figure(type):
    if type == "line":
        # plt.subplot(2,1,1)
        plt.plot(Time_470nm, zdFF, color='darkorchid')
        # plt.legend(loc='upper right')
        plt.ylabel("z dF/F")
        plt.title('Corrected signal')
        # plt.subplot(2,1,2)
        # plt.plot(Time_470nm, zInt_470nm, label="470nm", color='orangered')
        # plt.legend(loc='upper right')
        # plt.ylabel("Standardized intensity")
    elif type == "scatter":
        plt.scatter(zInt_410nm, zInt_470nm, s=2, color='darkseagreen')
        plt.plot(zInt_410nm, lr_model.predict(zInt_410nm.reshape(-1,1)), color='orangered')
        plt.title('Linear regression fit')
        plt.xlabel('410nm signal')
        plt.ylabel('470nm signal')
        plt.axes().set_aspect(0.3)

    plt.show()
    return 0


with open('config.yaml', 'r') as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)

example = config['raw_data']

with open(example) as f:
    reader = csv.reader(f) #コンストラクタ. 行を反復処理するイテレータとみなせる
    raw_data = np.array([row for row in reader]) #2次元配列

Raw_410nm = raw_data[:,1][2:].astype(np.float64)
Time_410nm = raw_data[:,2][2:].astype(np.float32)
Raw_470nm = raw_data[:,5][2:].astype(np.float64)
Time_470nm = raw_data[:,6][2:].astype(np.float32)

if len(Raw_410nm) != len(Raw_470nm):
    print("410nm and 470nm data lengths are different.")
    sys.exit()

# MeanInt_410nm = np.average(Raw_410nm)
# MeanInt_470nm = np.average(Raw_470nm)

# Moving Mean Algorithm
mma_window = config['mma_window']
MMA = MovingMeanAlgorithm(len(Raw_410nm), mma_window)
MovingMean_410nm = MMA(Raw_410nm)
MovingMean_470nm = MMA(Raw_470nm)

# adaptive iteratively reweiighted Penalized Least Squares algorithm
airPLS_lambda = config['airPLS_lambda']
airPLS = airPLS_Algorithm(airPLS_lambda)
airPLS_410nm = airPLS.airPLS(MovingMean_410nm)
airPLS_470nm = airPLS.airPLS(MovingMean_470nm)

# Baseline correction
BaseCorrec_410nm = MovingMean_410nm - airPLS_410nm
BaseCorrec_470nm = MovingMean_470nm - airPLS_470nm

# Standardization
zInt_410nm = (BaseCorrec_410nm - np.average(BaseCorrec_410nm)) / np.std(BaseCorrec_410nm)
zInt_470nm = (BaseCorrec_470nm - np.average(BaseCorrec_470nm)) / np.std(BaseCorrec_470nm)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(zInt_410nm.reshape(-1,1), zInt_470nm)
print('coefficient = ', lr_model.coef_[0])
print('intercept = ', lr_model.intercept_)

fitzInt_410nm = zInt_410nm * lr_model.coef_[0] + lr_model.intercept_


zdFF = zInt_470nm - fitzInt_410nm

_ = show_figure("line")
