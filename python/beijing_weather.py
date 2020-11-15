#!/usr/bin/env python3

import urllib.request
from tempfile import gettempdir
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from generative_rf import FeatureGenerator


# download the data if it's not in /tmp
data_path = os.path.join(gettempdir(), "beijing_weather.csv")
if not os.path.exists(data_path):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
    urllib.request.urlretrieve(url, data_path)

cols = ["month", "hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "pm2.5"]
target = "TEMP"
features = [c for c in cols if c != target]


def read_chunks():
    chunksize = 4096
    n_rows = 43824
    n_chunks = 1 + (n_rows - 1) // chunksize
    chunks = pd.read_csv(data_path, header=0, usecols=cols, chunksize=chunksize)
    for df in tqdm(chunks, total=n_chunks):
        yield df

# the data generator we are going to evaluate
gen_rf = FeatureGenerator()
pred = []
ref = []

# train/predict on-the-fly
for i, df in enumerate(read_chunks()):
    df.fillna(df.mean(axis=0, skipna=True), inplace=True)
    X = df[features].values
    y = df[target].values
    if i > 0:
        # prediction
        pred += gen_rf.predict(X).tolist()
        ref += y.tolist()

    # training
    if i == 0:
        gen_rf.register(X, RandomForestRegressor().fit(X, y))
    else:
        # generate data from the forest
        X2 = gen_rf.generate(len(df) * 10)
        y2 = gen_rf.predict(X2)

        # merge with current data
        X_all = np.concatenate([X, X2], axis=0)
        y_all = np.concatenate([y, y2], axis=0)

        # train a new forest from all the data
        new_rf = RandomForestRegressor().fit(X_all, y_all)
        gen_rf.register(X, new_rf)

# visualization
def rolling_avg(arr):
    csum = np.cumsum(arr)
    return (csum[1000:] - csum[:-1000])/1000

x = np.arange(len(rolling_avg(ref)))
plt.plot(x, rolling_avg(ref), label="ground-truth")
plt.plot(x, rolling_avg(pred), label="Generative RF")
plt.xlabel("time")
plt.ylabel(target)
plt.legend()
plt.xticks([], [])
plt.title("Beijing Temperature")
for dpi in (70, 80, 90):
    plt.savefig("vs_ground_truth_%i.png" % dpi, dpi=dpi)
