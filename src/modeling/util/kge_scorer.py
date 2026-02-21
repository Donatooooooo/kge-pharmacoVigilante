"""Score drug-SE pairs using a trained KGE_SE model."""

import gzip

import numpy as np
import torch

from src.config import MODEL_SE, TARGET_RELATION
from src.graph.embeddings import drug_name_to_entity


def _load_entity_mapping(mapping_dir):
    """Load entity_to_id and relation_to_id from saved training_triples."""
    entity_to_id = {}
    with gzip.open(str(mapping_dir / "entity_to_id.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity_to_id[parts[1]] = int(parts[0])

    relation_to_id = {}
    with gzip.open(str(mapping_dir / "relation_to_id.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                relation_to_id[parts[1]] = int(parts[0])

    return entity_to_id, relation_to_id


def score_all_pairs(drug_names, se_names, batch_size=512):
    """
    Score all (drug, has_side_effect, SE) pairs using the KGE_SE model.

    Returns:
        scores: np.ndarray of shape (n_drugs, n_se) with KGE scores.
                Unmatched entities get -inf (worst possible rank).
    """
    # Load model
    model = torch.load(str(MODEL_SE), weights_only=False).eval()
    device = next(model.parameters()).device

    # Load entity/relation mappings from saved training triples
    mapping_dir = MODEL_SE.parent / "training_triples"
    entity_to_id, relation_to_id = _load_entity_mapping(mapping_dir)

    rel_id = relation_to_id[TARGET_RELATION]

    # Map drug names to model entity IDs
    drug_ids = []
    drug_valid = []
    for name in drug_names:
        tsv_entity = drug_name_to_entity(name).replace("-", "")
        eid = entity_to_id.get(tsv_entity)
        drug_ids.append(eid if eid is not None else -1)
        drug_valid.append(eid is not None)

    # Map SE names to model entity IDs
    se_ids = []
    se_valid = []
    for se_name in se_names:
        tsv_entity = se_name.lower().replace(" ", "_").replace("-", "")
        eid = entity_to_id.get(tsv_entity)
        se_ids.append(eid if eid is not None else -1)
        se_valid.append(eid is not None)

    n_drugs = len(drug_names)
    n_se = len(se_names)
    scores = np.full((n_drugs, n_se), -np.inf)

    matched_drugs = sum(drug_valid)
    matched_se = sum(se_valid)
    print("  KGE_SE scoring:")
    print(f"  ├─ Matched drugs: {matched_drugs}/{n_drugs}")
    print(f"  ├─ Matched SEs:   {matched_se}/{n_se}")

    # Build valid SE tensor once
    valid_se_ids = [sid for sid, v in zip(se_ids, se_valid) if v]
    valid_se_col_indices = [i for i, v in enumerate(se_valid) if v]
    valid_se_tensor = torch.tensor(valid_se_ids, dtype=torch.long, device=device)
    n_valid_se = len(valid_se_tensor)

    with torch.no_grad():
        for drug_idx in range(n_drugs):
            if not drug_valid[drug_idx]:
                continue

            did = drug_ids[drug_idx]

            # Build triples: (drug, has_side_effect, se) for all valid SEs
            h = torch.full((n_valid_se,), did, dtype=torch.long, device=device)
            r = torch.full((n_valid_se,), rel_id, dtype=torch.long, device=device)
            t = valid_se_tensor

            # Score in batches
            drug_scores = []
            for start in range(0, n_valid_se, batch_size):
                end = min(start + batch_size, n_valid_se)
                hrt = torch.stack([h[start:end], r[start:end], t[start:end]], dim=1)
                batch_scores = model.score_hrt(hrt).cpu().numpy().flatten()
                drug_scores.append(batch_scores)

            drug_scores = np.concatenate(drug_scores)

            for score_idx, se_col_idx in enumerate(valid_se_col_indices):
                scores[drug_idx, se_col_idx] = drug_scores[score_idx]

    print(f"  └─ Score matrix shape: {scores.shape}")

    return scores
