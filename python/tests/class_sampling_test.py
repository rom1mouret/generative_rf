import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from generative_rf import class_sampling


class TestClassSampling(unittest.TestCase):

    def test_simple(self):
        X = np.zeros((3, 4))
        prob = np.array([
            [10, 5, 20, 5],
            [10, 10, 0, 0],
            [0, 5, 12, 20]
        ], dtype=np.float32)
        weights = np.ones(X.shape[0])
        X2, y, weights2 = class_sampling(X, prob, weights, max_duplication=1)
        self.assertEqual(X2.shape[0], 2 * X.shape[0])
        self.assertEqual(y.shape[0], X2.shape[0])
        self.assertEqual(weights2.shape[0], X2.shape[0])
        self.assertEqual(sorted(y.tolist()), sorted([0, 2, 0, 1, 2, 3]))
        self.assertAlmostEqual(weights.sum(), weights2.sum())

        X2, y, weights2 = class_sampling(X, prob, weights, max_duplication=2)
        self.assertEqual(X2.shape[0], 3 * X.shape[0])
        self.assertEqual(y.shape[0], X2.shape[0])
        self.assertEqual(weights2.shape[0], X2.shape[0])
        self.assertAlmostEqual(weights.sum(), weights2.sum())

    def test_with_rf(self):
        X = np.random.normal(size=(80, 2))
        y = [0] * (X.shape[0]//2) + [1] * (X.shape[0] - X.shape[0]//2)
        rf = RandomForestClassifier().fit(X, y)
        proba = rf.predict_proba(X)

        weights = np.ones(X.shape[0])
        _, y, _ = class_sampling(X, proba, weights, max_duplication=1)
        self.assertAlmostEqual(y.mean(), 0.5)


if __name__ == '__main__':
    unittest.main()
