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

    counts = y.sum(axis=0)
    valid = counts >= min_label_count
    y = y[:, valid]
    y_full = y_full[:, valid]
    se_names = [s for s, v in zip(se_names, valid) if v]

    if test_mask is not None:
        test_mask = test_mask[:, valid]

    return X, y, y_full, test_mask, drug_names, se_names
