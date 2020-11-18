## Dependencies

| Dependency   | Minimum version | Tested with     |
| ------------ | --------------- | --------------- |
| Python       | 3.5.0           | 3.5.2, 3.6.9    |
| scikit-learn | 0.17.0          | 0.22.1, 0.23.1  |
| Numpy        | 1.8.0           | 1.17.0, 1.19.1  |

Testing scripts require the following libraries, which are not mandatory to run GenerativeRF.

- tqdm
- matplotlib
- pandas

## Installation

```bash
sudo python3 setup.py install
```

or, to install it in your home directory,

```bash
python3 setup.py install --user
```

## Continual learning regression on a data stream

```python
from sklearn.ensemble import RandomForestRegressor
from generative_rf import FeatureGenerator

gen_rf = FeatureGenerator()
X, y = poll_data()  # poll_data() is your function
gen_rf.register(RandomForestRegressor().fit(X, y), real_data=X)

while True:
  X, y = poll_data()
  y_pred = gen_rf.predict(X)
  y_pred = gen_rf.predict_proba(X)
  # loss() and MAX_LOSS are provided by you
  if loss(y, y_pred).mean() > MAX_LOSS:
    # generate new data
    # please tailor approx_n for the problem at hand
    X2, w2 = gen_rf.generate(approx_n=20000)
    y2 = gen_rf.predict(X2)

    # merge with current data
    X_all = np.concatenate([X, X2], axis=0)
    y_all = np.concatenate([y, y2], axis=0)
    w = [1] * X.shape[0] + [w2] * X2.shape[0]

    # train a new forest from all the data
    new_rf = RandomForestRegressor().fit(X_all, y_all, sample_weight=w)
    gen_rf.register(new_rf).reinforce(X2, w2)
    # possible ways of using new_rf:
    # 1. compare predictions with the ground truth to detect anomalies
    # 2. serialize new_rf and use it wherever there is no ground truth available

  # this is called regardless of retraining
  gen_rf.reinforce(X).update_moments(X)

```

## Continual learning classification on a data stream

The only difference between regression and classification is the use of `predict_proba` and `class_sampling`.

```python
from sklearn.ensemble import RandomForestClassifier
from generative_rf import FeatureGenerator, class_sampling

gen_rf = FeatureGenerator()
X, y = poll_data()  # poll_data() is your function
gen_rf.register(RandomForestClassifier().fit(X, y)).reinforce(X)
gen_rf.update_moments(X)

while True:
  X, y = poll_data()
  y_pred = gen_rf.predict_proba(X)
  # loss() and MAX_LOSS are provided by you
  if loss(y, y_pred).mean() > MAX_LOSS:
    # generate new data
    # please tailor approx_n for the problem at hand
    X2, w2 = gen_rf.generate(approx_n=20000)
    proba = gen_rf.predict_proba(X2)
    X2, y2, sample_weights = class_sampling(X2, proba, w2)

    # merge with current data
    X_all = np.concatenate([X, X2], axis=0)
    y_all = np.concatenate([y, y2], axis=0)
    w = np.concatenate([[1]*len(y), sample_weights], axis=0)

    # train a new forest from all the data
    new_rf = RandomForestClassifier().fit(X_all, y_all, sample_weight=w)
    gen_rf.register(new_rf).reinforce(X2, w2)

  # this is called regardless of retraining
  gen_rf.reinforce(X).update_moments(X)
```
