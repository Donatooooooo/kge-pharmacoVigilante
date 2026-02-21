import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import SEED


def split_data(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"- Split: {X_train.shape[0]} train / {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test, scaler


class MarginalFrequencyBaseline:
    def __init__(self):
        self.frequencies = None

    def fit(self, X, y):
        self.frequencies = y.sum(axis=0) / y.shape[0]
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        return np.tile(self.frequencies, (n_samples, 1))
