#!/usr/bin/env python3

import urllib.request
from tempfile import gettempdir
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from generative_rf import FeatureGenerator
from generative_rf.baselines import RollingRF, SlowForgettingRF


class Named:
    def __init__(self, model, name: str) -> None:
        self._name = name
        self._model = model

    def __getattribute__(self, name: str):
        return getattr(object.__getattribute__(self, "_model"), name)

    def __repr__(self) -> str:
        return object.__getattribute__(self, "_name")


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

# train a regular RF
df = pd.concat([df.sample(frac=0.1) for df in read_chunks()], axis=0)
df.fillna(df.mean(axis=0, skipna=True), inplace=True)
df = df.sample(frac=1.0)
X = df[features].values
y = df[target].values
oracle_rf = Named(RandomForestRegressor().fit(X, y), name="Oracle RF")

# other baselines
good_baseline = SlowForgettingRF(0.8)
rolling_rfs = [
    RollingRF(0.8),
    RollingRF(0.6),
    good_baseline,
    SlowForgettingRF(0.6)
]

# the generator we are going to evaluate
gen_rf = Named(FeatureGenerator(), name="Generative RF")

# the performance of each model
errors = {rf: [] for rf in rolling_rfs+[oracle_rf, gen_rf]}

# train/predict on-the-fly
for i, df in enumerate(read_chunks()):
    df.fillna(df.mean(axis=0, skipna=True), inplace=True)
    X = df[features].values
    y = df[target].values
    if i > 0:
        # prediction
        for rf in errors.keys():
            error = np.abs(y - rf.predict(X))
            errors[rf] += error.tolist()

    # training
    new_rf = RandomForestRegressor().fit(X, y)
    for rf in rolling_rfs:
        rf.merge(new_rf)
    if i == 0:
        gen_rf.register(new_rf)
    else:
        # generate data from the forest
        X2, w2 = gen_rf.generate(20000)
        y2 = gen_rf.predict(X2)

        # merge with current data
        X_all = np.concatenate([X, X2], axis=0)
        y_all = np.concatenate([y, y2], axis=0)
        w = np.array([1] * X.shape[0] + [w2] * X2.shape[0])
        w *= len(w) / w.sum()

        # train a new forest from all the data
        new_rf = RandomForestRegressor().fit(X_all, y_all, sample_weight=w)
        gen_rf.register(new_rf).reinforce(X2, w2)

    # always call these functions at the end of each iteration
    gen_rf.reinforce(X).update_moments(X)

# visualization
def rolling_avg(arr):
    csum = np.cumsum(arr)
    return (csum[1000:] - csum[:-1000])/1000

# x-axis
x = np.arange(len(rolling_avg(errors[rolling_rfs[0]])))

# 1st plot
for rf in [oracle_rf]+rolling_rfs:
    plt.plot(x, rolling_avg(errors[rf]), label=str(rf))
plt.legend()
plt.title("Beijing Temperature")
plt.xlabel("time")
plt.ylabel(target+" L1 error")
plt.xticks([], [])
plt.savefig("cmp1.png")

# 2nd plot
plt.clf()
for rf in [good_baseline, oracle_rf, gen_rf]:
    plt.plot(x, rolling_avg(errors[rf]), label=str(rf))
plt.legend()
plt.title("Beijing Temperature")
plt.xlabel("time")
plt.ylabel(target+" L1 error")
plt.xticks([], [])
plt.savefig("cmp2.png")
