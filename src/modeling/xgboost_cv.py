import pickle
import warnings

import numpy as np
import optuna
import json
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import SEED, XGB, REPORTS_DIR


class XGBoostTrainer:

    def __init__(self, device="cuda"):
        self.device = device
        self.model = None

    def _make_pipeline(self, **xgb_params):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    OneVsRestClassifier(
                        XGBClassifier(
                            eval_metric="logloss",
                            tree_method="hist",
                            device=self.device,
                            random_state=SEED,
                            n_jobs=-1,
                            **xgb_params,
                        )
                    ),
                ),
            ]
        )

    def _objective(self, trial, X, y, n_splits):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.05, 0.1]),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5]),
        }
        
        
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        scores = []

        for step, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            pipe = self._make_pipeline(**params)
            pipe.fit(X[train_idx], y[train_idx])

            y_proba = self.get_proba(pipe, X[val_idx])
            score = log_loss(y[val_idx], y_proba)
            scores.append(score)

            trial.report(np.mean(scores), step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    def search_and_train(self, X, y, n_splits=5, n_trials=30):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )

        study.optimize(
            lambda trial: self._objective(trial, X, y, n_splits),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        output = REPORTS_DIR / "xgboost_hyperparams.json"
        with open(str(output), "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=2, ensure_ascii=False)

        self.model = self._make_pipeline(**study.best_params)
        self.model.fit(X, y)

        self._save_model()
        return self.model

    def predict_proba(self, X):
        return self.get_proba(self.model, X)

    @staticmethod
    def get_proba(pipe, X):
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

    def _save_model(self):
        XGB.mkdir(parents=True, exist_ok=True)
        with open(str(XGB), "wb") as f:
            pickle.dump(self.model, f)
        print(f"  Model saved -> {XGB}")

    @staticmethod
    def load():
        with open(str(XGB), "rb") as f:
            model = pickle.load(f)
        trainer = XGBoostTrainer()
        trainer.model = model
        return trainer
