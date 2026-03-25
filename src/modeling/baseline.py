import numpy as np


class MarginalFrequencyBaseline:
    def __init__(self):
        self.frequencies = None

    def fit(self, X, y):
        self.frequencies = y.sum(axis=0) / y.shape[0]
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        return np.tile(self.frequencies, (n_samples, 1))
