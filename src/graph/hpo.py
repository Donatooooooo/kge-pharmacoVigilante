import logging
import os
from pathlib import Path
import warnings

import mlflow
import numpy as np
import pandas as pd
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
from pykeen.utils import set_random_seed
from sklearn.model_selection import train_test_split
import torch

warnings.filterwarnings("ignore")
logging.getLogger("pykeen").setLevel(logging.ERROR)

from src.config import (
    HPO_ROTATE_DIR,
    INCLUDE_SIDE_EFFECTS,
    MLFLOW,
    SEED,
    TARGET_RELATION,
    TRIPLES_TSV,
    get_rotate_hpo_config,
)

INFO = " "


class HpoPipeline:
    def __init__(self, tsv_path):
        self.tsv_path = tsv_path
        self.data = None
        self.triples_factory = None
        self.training = None
        self.validation = None
        self.testing = None

        self._load_triples()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_triples(self, delimiter="\t"):
        self.data = pd.read_csv(
            self.tsv_path,
            delimiter=delimiter,
            names=["subject", "relation", "object"],
            encoding="utf-16",
        )
        self.data = self.data.fillna("").astype(str)

        n_entities = len(set(self.data["subject"].tolist() + self.data["object"].tolist()))
        n_relations = len(self.data["relation"].unique())

        print(f"- Loaded {len(self.data):,} triples")
        print(f"   ├─ Entities: {n_entities:,}")
        print(f"   └─ Relations: {n_relations}")

        self.triples_factory = TriplesFactory.from_labeled_triples(
            self.data.values, create_inverse_triples=True
        )

    def create_dataset(self, train_ratio, val_ratio, test_ratio):
        if INCLUDE_SIDE_EFFECTS:
            self._create_strategic_split(train_ratio, val_ratio, test_ratio)
        else:
            self._create_random_split(train_ratio, val_ratio, test_ratio)

    def _create_random_split(self, train_ratio, val_ratio, test_ratio):
        self.training, self.testing, self.validation = self.triples_factory.split(
            [train_ratio, test_ratio, val_ratio]
        )
        print("\n- Dataset splits:")
        print(f"   ├─ Training:   {self.training.num_triples:,} triples")
        print(f"   ├─ Validation: {self.validation.num_triples:,} triples")
        print(f"   └─ Testing:    {self.testing.num_triples:,} triples")

    def _create_strategic_split(self, _, val_ratio, test_ratio):
        se_mask = self.data["relation"] == TARGET_RELATION
        se_triples = self.data[se_mask].values
        other_triples = self.data[~se_mask].values

        se_train, se_temp = train_test_split(
            se_triples,
            test_size=(val_ratio + test_ratio),
            random_state=SEED,
        )
        se_val, se_test = train_test_split(
            se_temp,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=SEED,
        )

        train_triples = np.vstack([other_triples, se_train])

        self.training = TriplesFactory.from_labeled_triples(
            train_triples,
            entity_to_id=self.triples_factory.entity_to_id,
            relation_to_id=self.triples_factory.relation_to_id,
            create_inverse_triples=True,
        )
        self.validation = TriplesFactory.from_labeled_triples(
            se_val,
            entity_to_id=self.triples_factory.entity_to_id,
            relation_to_id=self.triples_factory.relation_to_id,
            create_inverse_triples=True,
        )
        self.testing = TriplesFactory.from_labeled_triples(
            se_test,
            entity_to_id=self.triples_factory.entity_to_id,
            relation_to_id=self.triples_factory.relation_to_id,
            create_inverse_triples=True,
        )

        print(f"\n- Dataset splits for '{TARGET_RELATION}' prediction:")
        print(f"   ├─ Training:   {self.training.num_triples:,} triples (all relations)")
        print(
            f"   ├─ Validation: {self.validation.num_triples:,} triples (only {TARGET_RELATION})"
        )
        print(f"   └─ Testing:    {self.testing.num_triples:,} triples (only {TARGET_RELATION})")

    def _get_data_args(self):
        return self.training, self.testing, self.validation, self.device

    def _log_to_mlflow(self, model_name, result, save_model_dir):
        triples_stem = Path(self.tsv_path).stem
        exp = f"{model_name}_{triples_stem}"
        if INCLUDE_SIDE_EFFECTS:
            exp += "_SideEffects"

        best_trial = result.study.best_trial

        if not mlflow.get_experiment_by_name(exp):
            mlflow.create_experiment(exp)
        mlflow.set_experiment(exp)

        with mlflow.start_run():
            mlflow.set_tag("n_trials", len(result.study.trials))
            mlflow.set_tag("best_trial", best_trial.number)

            if INCLUDE_SIDE_EFFECTS:
                mlflow.set_tag("info", INFO)
                mlflow.set_tag("target_relation", TARGET_RELATION)

            for key, value in best_trial.params.items():
                mlflow.log_param(key, value)

            mlflow.log_metric("best_score", best_trial.value)
            for key, value in best_trial.user_attrs.items():
                if "hits" in key or "rank" in key:
                    if "realistic" in key:
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)

            try:
                trial_dir = Path(save_model_dir) / str(best_trial.number)

                results_json = trial_dir / "results.json"
                if results_json.exists():
                    mlflow.log_artifact(str(results_json))

                model_pkl = trial_dir / "trained_model.pkl"
                if model_pkl.exists():
                    mlflow.log_artifact(str(model_pkl))
                else:
                    for pkl in trial_dir.glob("*.pkl"):
                        mlflow.log_artifact(str(pkl))
            except Exception as e:
                print(f"   └─ MLflow artifact logging failed: {e}")

    def run_hpo(self):
        self.create_dataset(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

        results = {}
        data_args = self._get_data_args()
        configs = {
            "RotatE": (get_rotate_hpo_config, HPO_ROTATE_DIR),
        }

        for model_name, (get_config, save_dir) in configs.items():
            print(f"- [HPO] {model_name}")
            result = hpo_pipeline(**get_config(*data_args))
            results[model_name] = result
            if MLFLOW:
                self._log_to_mlflow(model_name, result, save_dir)


if __name__ == "__main__":
    set_random_seed(SEED)

    if MLFLOW:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    hpo = HpoPipeline(str(TRIPLES_TSV))
    hpo.run_hpo()
