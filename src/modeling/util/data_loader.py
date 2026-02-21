import pickle

from src.config import EMB_DATASET


def load_dataset(min_label_count=2):
    with open(str(EMB_DATASET), "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]
    y_full = data.get("y_full", y)
    test_mask = data.get("test_mask")
    drug_names = data["drug_names"]
    se_names = data["side_effect_names"]

    print(f"- Loaded: {X.shape[0]} drugs, {X.shape[1]}d embeddings, {y.shape[1]} labels")

    counts = y.sum(axis=0)
    valid = counts >= min_label_count
    y = y[:, valid]
    y_full = y_full[:, valid]
    se_names = [s for s, v in zip(se_names, valid) if v]

    if test_mask is not None:
        test_mask = test_mask[:, valid]

    print(f"- Labels with >= {min_label_count} samples (in train): {sum(valid)}/{len(valid)}")

    if test_mask is not None:
        print(f"- Test pairs after filtering: {int(test_mask.sum())}")

    return X, y, y_full, test_mask, drug_names, se_names
