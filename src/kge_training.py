import logging
import warnings

import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import torch

from config import MODELS_DIR, TRAINED_MODEL_PKL, TRIPLES_TSV

warnings.filterwarnings("ignore")
logging.getLogger("pykeen").setLevel(logging.ERROR)


class LinkPredictor:
    def __init__(self, tsv_path: str):
        self.tsv_path = tsv_path
        self.model = None
        self.results = None

        self.triples_factory = self.load_triples()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_triples(self, delimiter: str = "\t"):
        data = pd.read_csv(
            self.tsv_path,
            delimiter=delimiter,
            names=["subject", "relation", "object"],
            encoding="utf-16",
        )
        data = data.fillna("").astype(str)
        print(data["relation"].unique())
        print(f"- Loaded {len(data):,} triples")
        print(f"   ├─ Entities: {len(set(data['subject'].tolist() + data['object'].tolist()))}")
        print(f"   └─ Relations: {len(data['relation'].unique())}")

        self.triples_factory = TriplesFactory.from_labeled_triples(
            data.values, create_inverse_triples=True
        )
        return self.triples_factory

    def create_dataset(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1
    ):
        self.training, self.validation, self.testing = self.triples_factory.split(
            [train_ratio, val_ratio, test_ratio]
        )
        return

    def train_model(self, model_name="TransE"):
        self.results = pipeline(
            model=model_name,
            training=self.training,
            validation=self.validation,
            testing=self.testing,
            model_kwargs={
                "embedding_dim": 300,
                "scoring_fct_norm": 2,
                "entity_initializer": "normal",
            },
            loss="MarginRankingLoss",
            loss_kwargs={
                "margin": 1.098831035927509,
            },
            negative_sampler="basic",
            negative_sampler_kwargs={
                "num_negs_per_pos": 5,
            },
            optimizer="Adam",
            optimizer_kwargs={
                "lr": 0.007330148646008595,
            },
            lr_scheduler="ExponentialLR",
            lr_scheduler_kwargs={
                "gamma": 0.8531368147004285,
            },
            training_kwargs={
                "num_epochs": 70,
                "batch_size": 1024,
                "use_tqdm": True,
            },
            stopper="early",
            stopper_kwargs=dict(frequency=20, patience=10, relative_delta=0.002),
            device=self.device,
        )

        print("\n- Model Trained")
        print("  └─ Evaluation on test set:")
        print(f"      Mean Rank: {self.results.get_metric('mean_rank'):.4f}")
        print(f"      Mean Reciprocal Rank: {self.results.get_metric('mean_reciprocal_rank'):.4f}")
        print(f"      Hits@1: {self.results.get_metric('hits_at_1'):.4f}")
        print(f"      Hits@3: {self.results.get_metric('hits_at_3'):.4f}")
        print(f"      Hits@10: {self.results.get_metric('hits_at_10'):.4f}")

        return self.results

    def save_model(self):
        self.results.save_to_directory(str(MODELS_DIR))
        print(f"- Model saved to {MODELS_DIR}")

    def load_model(self):
        self.model = torch.load(str(TRAINED_MODEL_PKL), weights_only=False)
        print("- Loaded model")


def run():
    tsv_file = str(TRIPLES_TSV)

    predictor = LinkPredictor(tsv_file)
    predictor.create_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    predictor.train_model(model_name="TransE")
    predictor.save_model()

    return predictor


if __name__ == "__main__":
    run()
