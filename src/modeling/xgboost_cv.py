import pickle

import numpy as np
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from src.config import SEED, XGB


class XGBoost:
    def __init__(self, n_splits=10):
        self.n_splits = n_splits
        self.fold_models = []
        self.y_prob_oof = None

    def _make_pipeline(self):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    OneVsRestClassifier(
                        XGBClassifier(
                            n_estimators=100,
                            max_depth=3,
                            learning_rate=0.1,
                            min_child_weight=3,
                            eval_metric="logloss",
                            tree_method="hist",
                            device="cuda",
                            random_state=SEED,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        )

    @staticmethod
    def _get_proba(pipe, X):
        if hasattr(pipe, "predict_proba"):
            try:
                p = pipe.predict_proba(X)
                if isinstance(p, list):
                    return np.column_stack([col[:, 1] if col.ndim > 1 else col for col in p])
                elif p.ndim == 3:
                    return p[:, :, 1]
                return p
            except Exception:
                pass
        if hasattr(pipe, "decision_function"):
            try:
                return pipe.decision_function(X)
            except Exception:
                pass
        return None

    def fit(self, X, y):
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=SEED)
        self.y_prob_oof = np.zeros(y.shape, dtype=float)
        self.fold_models = []

        folds = list(enumerate(cv.split(X), 1))
        for _, (train_idx, test_idx) in tqdm(folds, desc=f"{self.n_splits}-Fold CV"):
            pipe = self._make_pipeline()
            pipe.fit(X[train_idx], y[train_idx])

            y_prob = self._get_proba(pipe, X[test_idx])
            if y_prob is not None:
                self.y_prob_oof[test_idx] = y_prob

            self.fold_models.append(pipe)
        return self

    def predict_proba(self, X):
        probas = [self._get_proba(p, X) for p in self.fold_models]
        return np.mean([p for p in probas if p is not None], axis=0)


def train_cv(X, y, n_splits=10):
    model = XGBoost(n_splits=n_splits)
    model.fit(X, y)

    XGB.mkdir(parents=True, exist_ok=True)

    model_path = XGB / "xgboost_cv.pkl"
    with open(str(model_path), "wb") as f:
        pickle.dump(model, f)
    return model.y_prob_oof, model
