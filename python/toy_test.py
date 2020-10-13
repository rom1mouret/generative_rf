#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import make_s_curve
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from generative_rf import FeatureGenerator


s_curve = make_s_curve(n_samples=10000)[0][:, [0, 2]]
s_curve -= s_curve.mean(axis=0)
s_curve /= s_curve.std(axis=0)
noise = np.random.uniform(-2, 2, size=(500, 2))
y = [0] * len(s_curve) + [1] * len(noise)
X = np.concatenate([s_curve, noise], axis=0)

rf = RandomForestRegressor(n_estimators=100, max_depth=15).fit(X, y)
pred = rf.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred)
plt.xticks([], [])
plt.yticks([], [])
plt.title("Prediction with RandomForestRegressor")
plt.show()

generator = FeatureGenerator()
generator.register(X, rf)

data, weights = generator.generate(50000)
size = (weights - weights.min()) / (weights.max() - weights.min())
y = rf.predict(data)

plt.clf()
plt.scatter(data[:, 0], data[:, 1], c=y, s=5*size)
plt.xticks([], [])
plt.yticks([], [])
plt.title("Data sampled with RandomForestRegressor")
plt.show()
