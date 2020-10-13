import numpy as np
from copy import deepcopy


class RollingRF:
    def __init__(self, ratio: float) -> None:
        self._ratio = ratio
        self._rf = None

    def merge(self, new_rf, X: np.ndarray=None) -> None:
        if self._rf is None:
            self._rf = deepcopy(new_rf)
        else:
            keep = int(self._ratio * len(self._rf.estimators_))
            add = min(len(new_rf.estimators_), len(self._rf.estimators_) - keep)
            self._rf.estimators_ = new_rf.estimators_[:add] + self._rf.estimators_[-keep:]

    def predict(self, X) -> np.ndarray:
        return self._rf.predict(X)

    def __repr__(self) -> str:
        return "RollingRF(%.2f)" % self._ratio


class SlowForgettingRF(RollingRF):
    def merge(self, new_rf, X: np.ndarray=None) -> None:
        if self._rf is None:
            self._rf = deepcopy(new_rf)
        else:
            keep = int(self._ratio * len(self._rf.estimators_))
            add = min(len(new_rf.estimators_), len(self._rf.estimators_) - keep)
            keep_i = np.random.choice(len(self._rf.estimators_), size=keep, replace=False)
            self._rf.estimators_ = [
                self._rf.estimators_[i] for i in keep_i
            ] + new_rf.estimators_[:add]

    def __repr__(self) -> str:
        return "SlowForgettingRF(%.2f)" % self._ratio
