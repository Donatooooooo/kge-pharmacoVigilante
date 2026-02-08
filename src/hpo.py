import logging, warnings
warnings.filterwarnings("ignore")
logging.getLogger('pykeen').setLevel(logging.WARNING)

import pandas as pd
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
import torch, json

from config import (
    HPO_COMPLEX_DIR,
    HPO_RESULTS_JSON,
    HPO_ROTATE_DIR,
    HPO_TRANSE_DIR,
    TRIPLES_TSV,
    N_TRIALS
)


class HpoPipeline:
    def __init__(self, tsv_path):
        self.tsv_path = tsv_path
        self.dataset = None
        self.model = None
        self.results = None

        self.triples_factory = self.load_triples()

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    
        self.device = device

    def load_triples(self, delimiter = "\t"):
        data = pd.read_csv(
            self.tsv_path,
            delimiter=delimiter,
            names=["subject", "relation", "object"],
            encoding="utf-16",
        )
        data = data.fillna("").astype(str)
        print(data["relation"].unique())
        print(f"- Loaded {len(data)} triples")
        print(f"   ├─ Entities: {len(set(data['subject'].tolist() + data['object'].tolist()))}")
        print(f"   └─ Relations: {len(data['relation'].unique())}")

        self.triples_factory = TriplesFactory.from_labeled_triples(
            data.values, create_inverse_triples=True
        )
        return self.triples_factory

    def create_dataset(self, train_ratio, test_ratio, val_ratio):
        self.training, self.testing, self.validation = self.triples_factory.split(
            [train_ratio, test_ratio, val_ratio]
        )
        return

    def get_transe_hpo_config(self):
        return {
            "model": "TransE",
            "training": self.training,
            "testing": self.testing,
            "validation": self.validation,
            "negative_sampler": "basic",
            "n_trials":  N_TRIALS,
            "save_model_directory": str(HPO_TRANSE_DIR),
            "loss": "MarginRankingLoss",
            "loss_kwargs_ranges": dict(
                margin=dict(type=float, low=0.5, high=1.5),
            ),
            "lr_scheduler": "ExponentialLR",
            "lr_scheduler_kwargs_ranges": dict(
                gamma=dict(type=float, low=0.90, high=0.99),
            ),
            "model_kwargs": dict(
                entity_initializer="xavier_uniform_",
                relation_initializer="xavier_uniform_",
            ),
            "model_kwargs_ranges": dict(
                embedding_dim=dict(type=int, low=32, high=128, q=32),
                scoring_fct_norm=dict(type=int, low=1, high=2),
            ),
            "optimizer": "Adam",
            "optimizer_kwargs_ranges": dict(
                lr=dict(type=float, low=1e-4, high=1e-2, log=True),
            ),
            "training_kwargs": dict(
                batch_size=256,
                num_epochs=500,
            ),
            "negative_sampler_kwargs_ranges": dict(
                num_negs_per_pos=dict(type=int, low=1, high=30, log=True),
            ),
            "stopper": "early",
            "stopper_kwargs": dict(frequency=25, patience=15, relative_delta=0.001),
            "device": self.device,
        }

    def get_complex_hpo_config(self):
        return {
            "model": "ComplEx",
            "training": self.training,
            "testing": self.testing,
            "validation": self.validation,
            "negative_sampler": "basic",
            "n_trials":  N_TRIALS,
            "save_model_directory": str(HPO_COMPLEX_DIR),
            "loss": "SoftplusLoss",
            "lr_scheduler": "ExponentialLR",
            "lr_scheduler_kwargs_ranges": dict(
                gamma=dict(type=float, low=0.93, high=0.99),
            ),
            "model_kwargs": dict(
                entity_initializer="xavier_uniform_",
                relation_initializer="xavier_uniform_",
            ),
            "model_kwargs_ranges": dict(
                embedding_dim=dict(type=int, low=32, high=128, q=32),
            ),
            "regularizer": "LpRegularizer",
            "regularizer_kwargs": dict(p=2.0),
            "regularizer_kwargs_ranges": dict(
                weight=dict(type=float, low=1e-5, high=1e-3, log=True),
            ),
            "optimizer": "Adam",
            "optimizer_kwargs_ranges": dict(
                lr=dict(type=float, low=1e-4, high=1e-2, log=True),
            ),
            "training_kwargs": dict(
                batch_size=256,
                num_epochs=500,
            ),
            "negative_sampler_kwargs_ranges": dict(
                num_negs_per_pos=dict(type=int, low=1, high=30, log=True),
            ),
            "stopper": "early",
            "stopper_kwargs": dict(frequency=30, patience=20, relative_delta=0.0005),
            "device": self.device,
        }

    def get_rotate_hpo_config(self):
        return {
            "model": "RotatE",
            "training": self.training,
            "testing": self.testing,
            "validation": self.validation,
            "negative_sampler": "basic",
            "n_trials":  N_TRIALS,
            "save_model_directory": str(HPO_ROTATE_DIR),
            "loss": "NSSALoss",
            "loss_kwargs_ranges": dict(
                margin=dict(type=float, low=6.0, high=24.0),
                adversarial_temperature=dict(type=float, low=0.5, high=2.0),
            ),
            "lr_scheduler": "ExponentialLR",
            "lr_scheduler_kwargs_ranges": dict(
                gamma=dict(type=float, low=0.93, high=0.99),
            ),
            "model_kwargs_ranges": dict(
                embedding_dim=dict(type=int, low=32, high=128, q=32),
            ),
            "optimizer": "Adam",
            "optimizer_kwargs_ranges": dict(
                lr=dict(type=float, low=1e-4, high=1e-2, log=True),
            ),
            "training_kwargs": dict(
                batch_size=256,
                num_epochs=500,
            ),
            "negative_sampler_kwargs_ranges": dict(
                num_negs_per_pos=dict(type=int, low=1, high=30, log=True),
            ),
            "stopper": "early",
            "stopper_kwargs": dict(frequency=25, patience=15, relative_delta=0.001),
            "device": self.device,
        }

    def run_hpo(self):

        self.create_dataset(
            train_ratio = 0.7, 
            test_ratio = 0.2, 
            val_ratio = 0.1
        )

        # Searching Block
        results = {}
        print("\n- [HPO] TransE")
        results["TransE"] = hpo_pipeline(**self.get_transe_hpo_config())

        print("\n- [HPO] ComplEx")
        results["ComplEx"] = hpo_pipeline(**self.get_complex_hpo_config())

        print("\n- [HPO] RotatE")
        results["RotatE"] = hpo_pipeline(**self.get_rotate_hpo_config())


        # Decision Block
        print("\n- [HPO] Pipeline completed")
        model_details = {}
        try:
            for model_name, result in results.items():
                completed = [t for t in result.study.trials if t.state.name == "COMPLETE"]
                if completed:
                    best_score = result.study.best_trial.value
                    print(f"{model_name}: Best Score = {best_score:.4f}")
                    model_details[model_name] = {
                        "best_score": best_score,
                        "best_params": result.study.best_trial.params,
                        "n_trials": len(result.study.trials),
                    }

            if model_details:
                best_model_name = max(model_details, key=lambda m: model_details[m]["best_score"])
                best_info = model_details[best_model_name]

                output = {
                    "best_model": best_model_name,
                    "best_score": best_info["best_score"],
                    "best_params": best_info["best_params"],
                    "all_models": model_details,
                }

                with open(HPO_RESULTS_JSON, "w") as f:
                    json.dump(output, f, indent=4)

                print(f"\n- [HPO] Best model: {best_model_name} with (score={best_info['best_score']:.4f})")
                print(f"\n  Best params: {best_info['best_params']}")

        except Exception as e:
            print(f"[HPO] Savings FAILED, {e}")


if __name__ == "__main__":
    graph = str(TRIPLES_TSV)
    hpo = HpoPipeline(graph).run_hpo()
