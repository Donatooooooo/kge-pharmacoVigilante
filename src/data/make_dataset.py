import pandas as pd

from config import (
    PROCESSED_MEDICINE_CSV,
    RAW_COMPOSITION_CSV,
    RAW_MEDICINE_CSV,
    SIDE_EFFETCS_MEDICINE_CSV,
)


def to_lower(column):
    return column.str.lower()


def make_dataset():
    medicine = pd.read_csv(RAW_MEDICINE_CSV, low_memory=False)
    composition = pd.read_csv(RAW_COMPOSITION_CSV)

    medicine["name"] = to_lower(medicine["name"])
    composition["name"] = to_lower(composition["name"])

    merged = pd.merge(medicine, composition, left_on="name", right_on="name", how="inner")

    medicine_dataset = merged[
        [
            "name",
            "pack_size_label",
            "short_composition1",
            "short_composition2",
            "substitute0",
            "substitute1",
            "substitute2",
            "substitute3",
            "Therapeutic Class",
        ]
    ]

    side_effects = merged[["name"] + [c for c in merged.columns if c.startswith("sideEffect")]]

    medicine_dataset.to_csv(PROCESSED_MEDICINE_CSV, index=False)
    side_effects.to_csv(SIDE_EFFETCS_MEDICINE_CSV, index=False)
    return medicine_dataset


if __name__ == "__main__":
    medicine_dataset = make_dataset()
    print(f"""
    - Dataset created: {medicine_dataset.shape}
        """)
