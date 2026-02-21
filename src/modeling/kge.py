import logging
import warnings

import numpy as np
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from sklearn.model_selection import train_test_split
import torch

from config import (
    MODEL_SE,
    MODELS_DIR,
    PARAMS,
    SEED,
    TARGET_RELATION,
)

warnings.filterwarnings("ignore")
logging.getLogger("pykeen").setLevel(logging.ERROR)


class LinkPredictor:
    def __init__(self, tsv_path, side_effects=False):
        self.tsv_path = tsv_path
        self.side_effects = side_effects
        self.params = PARAMS[side_effects]
        self.model = None
        self.results = None
        self.data = None
        self.triples_factory = self.load_triples()
        self.device = self._get_device()

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_triples(self, delimiter="\t"):
        self.data = pd.read_csv(
            self.tsv_path,
            delimiter=delimiter,
            names=["subject", "relation", "object"],
            encoding="utf-16",
        )
        self.data = self.data.fillna("").astype(str)
        print(f"- Loaded {len(self.data):,} triples")
        print(
            f"   ├─ Entities: {len(set(self.data['subject'].tolist() + self.data['object'].tolist()))}"
        )
        print(f"   └─ Relations: {len(self.data['relation'].unique())}")
        self.triples_factory = TriplesFactory.from_labeled_triples(
            self.data.values, create_inverse_triples=True
        )
        return self.triples_factory

    def create_dataset(self, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        if self.side_effects:
            self._create_strategic_split(val_ratio, test_ratio)
        else:
            self.training, self.validation, self.testing = self.triples_factory.split(
                [train_ratio, val_ratio, test_ratio]
            )

    def _create_strategic_split(self, val_ratio, test_ratio):
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

    def train_model(self, model_name="RotatE"):
        p = self.params
        self.results = pipeline(
            model=model_name,
            training=self.training,
            validation=self.validation,
            testing=self.testing,
            model_kwargs={
                "embedding_dim": p["embedding_dim"],
                "entity_initializer": "xavier_uniform_",
                "relation_initializer": "init_phases",
            },
            loss="NSSALoss",
            loss_kwargs={
                "margin": p["margin"],
                "adversarial_temperature": p["adversarial_temperature"],
            },
            negative_sampler="bernoulli",
            negative_sampler_kwargs={
                "num_negs_per_pos": p["num_negs_per_pos"],
            },
            optimizer="Adam",
            optimizer_kwargs={
                "lr": p["lr"],
            },
            lr_scheduler="ExponentialLR",
            lr_scheduler_kwargs={
                "gamma": p["gamma"],
            },
            training_kwargs={
                "num_epochs": 500,
                "batch_size": p["batch_size"],
                "use_tqdm": True,
            },
            stopper="early",
            stopper_kwargs=dict(
                frequency=p["stopper_frequency"],
                patience=p["stopper_patience"],
                relative_delta=0.0005,
            ),
            device=self.device,
        )
        print("\n- Model Trained")
        print("  └─ Evaluation on test set:")
        print(f"      Mean Rank: {self.results.get_metric('mean_rank'):.4f}")
        print(f"      Mean Reciprocal Rank: {self.results.get_metric('mean_reciprocal_rank'):.4f}")
        print(f"      Hits@1: {self.results.get_metric('hits_at_1'):.4f}")
        print(f"      Hits@3: {self.results.get_metric('hits_at_3'):.4f}")
        print(f"      Hits@10: {self.results.get_metric('hits_at_10'):.4f}")

        self.results.save_to_directory(str(MODELS_DIR / p["save_dir"]))
        return self.results

    def load_model(self):
        self.model = torch.load(str(MODEL_SE), weights_only=False)
