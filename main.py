# -*- coding: utf-8 -*-
import numpy as np
import yaml
import csv
import matplotlib.pyplot as plt
from Modules.moving_mean import MovingMeanAlgorithm
from Modules.airPLS import airPLS_Algorithm


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

mma_window = config['mma_window']
MMA = MovingMeanAlgorithm(len(Raw_410nm), mma_window)
MovingMean_410nm = MMA(Raw_410nm)
MovingMean_470nm = MMA(Raw_470nm)


airPLS_lambda = config['airPLS_lambda']
airPLS = airPLS_Algorithm(airPLS_lambda)
airPLS_410nm = airPLS.airPLS(MovingMean_410nm)
airPLS_470nm = airPLS.airPLS(MovingMean_470nm)


BaseCorrec_410nm = MovingMean_410nm - airPLS_410nm
BaseCorrec_470nm = MovingMean_470nm - airPLS_470nm


zInt_410nm = (BaseCorrec_410nm - np.average(BaseCorrec_410nm)) / np.std(BaseCorrec_410nm)
zInt_470nm = (BaseCorrec_470nm - np.average(BaseCorrec_470nm)) / np.std(BaseCorrec_470nm)


plt.plot(Time_410nm, zInt_410nm)
plt.plot(Time_470nm, zInt_470nm)
plt.show()
