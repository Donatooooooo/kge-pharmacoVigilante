from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
from pykeen.triples import TriplesFactory
from sklearn.model_selection import train_test_split
import torch

from src.config import (
    EMB_DATASET,
    MODEL,
    SEED,
    SIDE_EFFETCS_MEDICINE_CSV,
    TARGET_RELATION,
    TRIPLES_TSV,
    TRIPLES_TSV_SE,
)
from src.graph.build_graph import process_drug


def drug_name_to_entity(drug):
    drug, _ = process_drug(drug)
    return drug.replace("  ", "").replace(" ", "")


class EmbeddingExtractor:
    def __init__(self):
        self.model = None
        self.entity_to_id = None

    def load_model(self):
        self.model = torch.load(str(MODEL), weights_only=False).eval()

    def load_entity_mapping(self):
        data = pd.read_csv(
            str(TRIPLES_TSV),
            delimiter="\t",
            names=["subject", "relation", "object"],
            encoding="utf-16",
        )
        data = data.fillna("").astype(str)

        tf = TriplesFactory.from_labeled_triples(data.values, create_inverse_triples=True)
        self.entity_to_id = tf.entity_to_id
        self.drug_entities = set(data["subject"].unique())

    def extract_embeddings(self):
        with torch.no_grad():
            raw = self.model.entity_representations[0]().cpu().numpy()

        if np.iscomplexobj(raw):
            self.all_embeddings = np.concatenate([raw.real, raw.imag], axis=1)
        else:
            self.all_embeddings = raw

    def load_side_effects(self):
        df = pd.read_csv(str(SIDE_EFFETCS_MEDICINE_CSV))
        df = df.drop_duplicates(subset=["name"])

        se_cols = [c for c in df.columns if c.startswith("sideEffect")]

        all_se = set()
        for col in se_cols:
            all_se.update(df[col].dropna().astype(str).str.strip().unique())
        all_se.discard("")
        self.side_effect_names = sorted(all_se)

        self.drug_side_effects = {}
        for _, row in df.iterrows():
            effects = set()
            for col in se_cols:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    effects.add(str(val).strip())
            self.drug_side_effects[row["name"]] = effects

    # ------------------------------------------------------------------

    def _build_split_masks(self, drug_names, side_effect_names, y_full):
        se_data = pd.read_csv(
            str(TRIPLES_TSV_SE),
            delimiter="\t",
            names=["subject", "relation", "object"],
            encoding="utf-16",
        )
        se_data = se_data.fillna("").astype(str)

        se_mask = se_data["relation"] == TARGET_RELATION
        se_triples = se_data[se_mask].values

        val_ratio = 0.1
        test_ratio = 0.2

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

        drug_to_indices = defaultdict(list)
        for idx, name in enumerate(drug_names):
            tsv_entity = drug_name_to_entity(name).replace("-", "")
            drug_to_indices[tsv_entity].append(idx)

        se_to_idx = {}
        for idx, se_name in enumerate(side_effect_names):
            tsv_entity = se_name.lower().replace(" ", "_").replace("-", "")
            se_to_idx[tsv_entity] = idx

        y_train = y_full.copy()
        test_mask = np.zeros_like(y_full)

        unmatched_test = 0
        unmatched_val = 0

        for triple in se_test:
            drug_entity, _, se_entity = triple
            drug_indices = drug_to_indices.get(drug_entity, [])
            se_idx = se_to_idx.get(se_entity)

            if drug_indices and se_idx is not None:
                for d_idx in drug_indices:
                    test_mask[d_idx, se_idx] = 1
                    y_train[d_idx, se_idx] = 0
            else:
                unmatched_test += 1

        for triple in se_val:
            drug_entity, _, se_entity = triple
            drug_indices = drug_to_indices.get(drug_entity, [])
            se_idx = se_to_idx.get(se_entity)

            if drug_indices and se_idx is not None:
                for d_idx in drug_indices:
                    y_train[d_idx, se_idx] = 0
            else:
                unmatched_val += 1

        overlap = int((y_train * test_mask).sum())
        assert overlap == 0, f"LEAK: {overlap} pairs in both train and test!"

        print("\n  Split alignment:")
        print(f"  ├─ TSV SE triples:       {len(se_triples)}")
        print(f"  ├─ Train SE triples:     {len(se_train)}")
        print(f"  ├─ Val SE triples:       {len(se_val)}")
        print(f"  ├─ Test SE triples:      {len(se_test)}")
        print(f"  ├─ y_train positives:    {int(y_train.sum())}")
        print(f"  ├─ test_mask positives:  {int(test_mask.sum())}")
        print(f"  ├─ y_full positives:     {int(y_full.sum())}")
        print(f"  ├─ Unmatched test:       {unmatched_test}")
        print(f"  └─ Unmatched val:        {unmatched_val}")

        return y_train, test_mask

    # ------------------------------------------------------------------

    def build_dataset(self):
        self.load_model()
        self.load_entity_mapping()
        self.extract_embeddings()
        self.load_side_effects()

        drugs, X_list, y_list = [], [], []
        unmatched = 0

        for original_name, effects in self.drug_side_effects.items():
            entity_name = drug_name_to_entity(original_name)

            if entity_name in self.entity_to_id:
                eid = self.entity_to_id[entity_name]
                emb = self.all_embeddings[eid]
                label = [1 if se in effects else 0 for se in self.side_effect_names]
                drugs.append(original_name)
                X_list.append(emb)
                y_list.append(label)
            else:
                unmatched += 1

        X = np.array(X_list)
        y_full = np.array(y_list)

        y_train, test_mask = self._build_split_masks(drugs, self.side_effect_names, y_full)

        dataset = {
            "drug_names": drugs,
            "X": X,
            "y": y_train,
            "y_full": y_full,
            "test_mask": test_mask,
            "side_effect_names": self.side_effect_names,
            "embedding_dim": X.shape[1],
        }

        with open(str(EMB_DATASET), "wb") as f:
            pickle.dump(dataset, f)

        print(f"\n  Dataset saved -> {EMB_DATASET}")
        print(f"  ├─ Matched drugs:          {len(drugs)}")
        print(f"  ├─ X (embeddings):         {X.shape}")
        print(f"  ├─ y_train (labels):       {y_train.shape}  (positives: {int(y_train.sum())})")
        print(
            f"  ├─ test_mask:              {test_mask.shape}  (positives: {int(test_mask.sum())})"
        )
        print(f"  ├─ Avg side effects/drug:  {y_full.sum(axis=1).mean():.1f}")
        print(f"  └─ Active labels (≥1 drug): {(y_full.sum(axis=0) > 0).sum()}")


if __name__ == "__main__":
    EmbeddingExtractor().build_dataset()
