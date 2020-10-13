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
        sample_weights: np.ndarray,
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
        self._count = np.zeros(self._tree.tree_.node_count, dtype=np.int32)

    def count(self, X: np.ndarray) -> None:
        # indicator is a sparse matrix of shape (n_samples, n_nodes)
        indicator = self._tree.decision_path(X)

        # iterate over the indicator matrix to count the samples in each branch
        rows, cols = indicator.nonzero()
        np.add.at(self._count, cols, 1)
        # equivalent to:
        # for j in cols:
        #     self._count[j] += 1

    def left_probability(self, node_index: int) -> float:
        left = self._count[self._tree.tree_.children_left[node_index]]
        right = self._count[self._tree.tree_.children_right[node_index]]
        return left / (left + right)


class FeatureGenerator:
    def __init__(self) -> None:
        # we need means and stds to initialize the generated data
        self._scaler = StandardScaler()

    def generate(self, approx_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Generate data near the decision boundaries.
        Parameters
        ----------
        x : int
            The approximate number of samples to generate.

        Returns
        -------
        data : numpy.ndarray
            Generated features. Shape: (n_samples, dim)
        sample_weight : numpy.ndarray
            Sample weights for training a new random forest. Shape: (n_samples, )
        """
        # number of samples per tree
        n_per_tree = approx_n // self._rf.n_estimators
        n = self._rf.n_estimators * n_per_tree  # actual number of samples

        # default values (some features won't be set)
        stds = np.sqrt(self._scaler.var_)
        X = np.random.normal(size=(n, self._dim)) * stds + self._scaler.mean_

        # this arrays counts the number of draws that fall into a different leaf
        weights = np.zeros(n)

        # generate n_per_tree samples from each tree
        for i_tree, estimator in enumerate(self._rf.estimators_):
            tree = estimator.tree_
            leaf_to_row_index = defaultdict(list)  # to calculate weights
            for i in range(n_per_tree):
                row_index = i_tree * n_per_tree + i
                node_index = 0
                right_bound = np.ones(self._dim) * np.inf
                left_bound = -right_bound

                # randomly pick one path in the tree
                while node_index != TREE_LEAF and \
                    tree.children_left[node_index] != tree.children_right[node_index]:
                    prev_node_index = node_index
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
                    X[row_index, feature_i] = value

                    # branching
                    if value <= threshold:
                        node_index = tree.children_left[node_index]
                        right_bound[feature_i] = min(right_bound[feature_i], threshold)
                    else:
                        node_index = tree.children_right[node_index]
                        left_bound[feature_i] = max(left_bound[feature_i], threshold)

                # see below how this is used to calculate weights:
                leaf_to_row_index[prev_node_index].append(row_index)

            for indices in leaf_to_row_index.values():
                # if two draws fall into the same leaf, their weight is 1/2
                w = 1 / len(indices)
                for k in indices:
                    weights[k] = w

        return X, weights

    def register(
            self, X: np.ndarray,
            rf: Union[RandomForestRegressor, RandomForestClassifier]) -> None:
        """ Call this function after training your random forest, whether or not
        the forest was trained from generated data.

        Parameters
        ----------
        X : numpy.ndarray
            The features the random forest was trained from. Shape: (nrows, dim)
            This includes generated data, if the forest was trained from such
            data.
        rf : RandomForestRegressor or RandomForestClassifier
             A trained random forest.
        """
        self._dim = X.shape[1]
        self._rf = rf
        self._counting_trees = [CountingTree(t) for t in rf.estimators_]
        self.reinforce(X)

    def reinforce(self, X: np.ndarray) -> None:
        """ Call this function if you are not using X for training (perhaps
        because the current RF works well enough on X), but you still
        want to nudge the generator towards X.

        Parameters
        ----------
        X : numpy.ndarray
            Features the random forest is run on. Shape: (nrows, dim)
            Typically prediction data.
        """
        self._scaler.partial_fit(X)
        for wrapper in self._counting_trees:
            wrapper.count(X)

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
