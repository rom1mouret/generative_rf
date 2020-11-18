from typing import Tuple, Union
import random
from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.tree._tree import TREE_LEAF
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def class_sampling(
        X: np.ndarray,
        proba: np.ndarray,
        sample_weights: np.ndarray=None,
        max_duplication: int=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Duplicate each feature vector for each class, while adjusting
    sample_weights to reflect the given probabilities.

    Parameters
    ----------
    X : numpy.ndarray
        The features.
    proba : numpy.ndarray
        Class probabilities for each feature vector. Shape: (n_samples, n_classes)
    sample_weights: numpy.ndarray
        Sample weights before duplication. Shape: (n_samples, )
    max_duplication: int
        Maximum number of times a row is duplicated.
        If there are two classes, each row will be duplicated once (duplication=1).
    Returns
    -------
    X : numpy.ndarray
        Feature rows purposefully duplicated. Shape: (n_samples, dim)
    sample_weight : numpy.ndarray
        Sample weights for training a new random forest. Shape: (n_samples, )
    """
    if sample_weights is None:
        sample_weights = np.ones(X.shape[0])

    dup = min(proba.shape[1], max_duplication + 1)
    argtop = (-proba).argsort(axis=1)[:, :dup]
    row_indices = np.arange(len(proba))[:, np.newaxis]
    rough_proba = proba[row_indices, argtop]
    rough_proba /= rough_proba.sum(axis=1, keepdims=True)

    rows = []
    y = []
    weights = []
    for k in range(dup):
        rows.append(X)
        y.append(argtop[:, k])
        weights.append(sample_weights * rough_proba[:, k])

    return np.vstack(rows), np.concatenate(y), np.concatenate(weights)


class CountingTree:
    """ Wrapper to count the number of samples in each branch of the tree.
        There is no reason for the user to directly use this class. """
    def __init__(self, base_tree: Union[DecisionTreeRegressor, DecisionTreeClassifier]) -> None:
        self._tree = base_tree
        self._count = np.zeros(self._tree.tree_.node_count)

    def count(self, X: np.ndarray, weight: float) -> None:
        # indicator is a sparse matrix of shape (n_samples, n_nodes)
        indicator = self._tree.decision_path(X)

        # iterate over the indicator matrix to count the samples in each branch
        rows, cols = indicator.nonzero()
        np.add.at(self._count, cols, weight)
        # equivalent to:
        # for j in cols:
        #     self._count[j] += weight

    def left_probability(self, node_index: int) -> float:
        left = self._count[self._tree.tree_.children_left[node_index]]
        right = self._count[self._tree.tree_.children_right[node_index]]
        return left / (left + right)


class FeatureGenerator:
    def __init__(self) -> None:
        # we need means and stds to initialize the generated data
        self._scaler = StandardScaler()
        self._total_samples = 0

    def generate(self, approx_n: int) -> Tuple[np.ndarray, float]:
        """ Generate data near the decision boundaries.
        Parameters
        ----------
        x : int
            The approximate number of samples to generate.

        Returns
        -------
        data : numpy.ndarray
            Generated features. Shape: (n_samples, dim)
        sample_weight : float
            Weight of each sample
        """
        # number of samples per tree
        n_per_tree = approx_n // self._rf.n_estimators
        n = self._rf.n_estimators * n_per_tree  # actual number of samples

        # default values (some features won't be set by the below algorithm)
        stds = np.sqrt(self._scaler.var_)
        X = np.random.normal(size=(n, self._dim)) * stds + self._scaler.mean_

        # generate n_per_tree samples from each tree
        for i_tree, estimator in enumerate(self._rf.estimators_):
            tree = estimator.tree_
            for i in range(n_per_tree):
                row_index = i_tree * n_per_tree + i
                node_index = 0
                right_bound = np.ones(self._dim) * np.inf
                left_bound = -right_bound

                # randomly pick one path in the tree
                while node_index != TREE_LEAF and \
                    tree.children_left[node_index] != tree.children_right[node_index]:
                    threshold = tree.threshold[node_index]
                    feature_i = tree.feature[node_index]

                    # probability of branching left or right
                    left_prob = self._counting_trees[i_tree].left_probability(node_index)

                    # we pick a value close to the threshold...
                    shift = 0.05 * np.abs(np.random.normal()) * stds[feature_i]
                    if random.random() <= left_prob:
                        value = threshold - shift
                    else:
                        value = threshold + shift
                    # ... but still within the known bounds
                    value = min(right_bound[feature_i], max(left_bound[feature_i], value))
                    # alternatively, we could keep the value already set, but I believe
                    # the chosen method restricts the value to be even closer to the
                    # decision boundary
                    X[row_index, feature_i] = value

                    # branching
                    if value <= threshold:
                        node_index = tree.children_left[node_index]
                        right_bound[feature_i] = min(right_bound[feature_i], threshold)
                    else:
                        node_index = tree.children_right[node_index]
                        left_bound[feature_i] = max(left_bound[feature_i], threshold)

        return X, self._total_samples / X.shape[0]

    def register(self, rf: Union[RandomForestRegressor, RandomForestClassifier]) -> "FeatureGenerator":
        """ Call this function after training your random forest, whether or not
        the forest was trained from generated data.

        Parameters
        ----------
        rf : RandomForestRegressor or RandomForestClassifier
             A newly trained random forest.
        """
        self._rf = rf
        self._dim = rf.n_features_
        self._counting_trees = [CountingTree(t) for t in rf.estimators_]

        return self

    def update_moments(self, X: np.ndarray) -> "FeatureGenerator":
        """ Updates mean and variance of the data distribution

        Parameters
        ----------
        X : numpy.ndarray
            Features the random forest is run on. Shape: (nrows, dim)
            Typically, X is an input prediction batch.
        """
        self._scaler.partial_fit(X)
        self._total_samples += X.shape[0]

        return self

    def reinforce(self, X: np.ndarray, weight: float=1) -> "FeatureGenerator":
        """ updates the sample counts at the tree node level.
        Always call this method after training

        Parameters
        ----------
        X : numpy.ndarray
            Features the random forest is run on. Shape: (nrows, dim)
            Typically the training data.
        weight: float
            X's sample weights
        """
        for wrapper in self._counting_trees:
            wrapper.count(X, weight)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ This is just for convenience.
        You can directly call scikit RandomForest's predict() instead."""
        return self._rf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ This is just for convenience.
        You can directly call scikit RandomForest's predict() instead.
        It only works if the model registered to the generator was a
        RandomForestClassifier.
        """
        return self._rf.predict_proba(X)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """ This is just for convenience.
        You can directly call scikit RandomForest's predict() instead.
        It only works if the model registered to the generator was a
        RandomForestClassifier.
        """
        return self._rf.predict_log_proba(X)
