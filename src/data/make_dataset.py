from labels import normalize_side_effects
import numpy as np
import pandas as pd
from patterns import normalize_chemical_class, normalize_use

from src.config import (
    MIN_CC_FREQ,
    MIN_SE_FREQ,
    MIN_USE_FREQ,
    PROCESSED_MEDICINE_CSV,
    RAW_COMPOSITION_CSV,
    RAW_MEDICINE_CSV,
    SIDE_EFFETCS_MEDICINE_CSV,
)


def filter_se(df, se_cols, min_frequency=MIN_SE_FREQ):
    se_stack = df[se_cols].stack()
    se_counts = se_stack.value_counts()
    valid_se = set(se_counts[se_counts >= min_frequency].index)

    # Drop duplicates se in a raw
    arr = df[se_cols].values
    for i in range(len(arr)):
        seen = set()
        for j in range(len(arr[i])):
            if pd.notna(arr[i, j]):
                if arr[i, j] in seen:
                    arr[i, j] = None
                else:
                    seen.add(arr[i, j])
    df[se_cols] = arr

    # Drop rare se under a treshold
    df[se_cols] = df[se_cols].where(df[se_cols].isin(valid_se))

    return df


def clean_se(df, se_cols):
    to_exclude = {
        "no common side effects seen",
        "limited data available",
    }

    df[se_cols] = (
        df[se_cols]
        .apply(lambda x: x.str.lower())
        .apply(lambda x: x.str.replace(r"\s*\([^)]*\)", "", regex=True))
        .apply(lambda x: x.str.replace(r"[\-_/]", " ", regex=True))
        .apply(lambda x: x.str.replace(r"\s+", " ", regex=True))
        .apply(lambda x: x.str.strip())
        .replace(to_exclude, pd.NA)
    )
    return df


def normalize_se_labels(df, se_cols):
    mapping = normalize_side_effects(df)
    df[se_cols] = df[se_cols].apply(lambda col: col.map(mapping).fillna(col))
    return df


def compact_se_columns(df, se_cols):
    arr = df[se_cols].values

    mask = pd.notna(arr)
    out = np.full(arr.shape, np.nan, dtype=object)

    for i in range(len(arr)):
        valid = arr[i, mask[i]]
        out[i, : len(valid)] = valid

    max_se = mask.sum(axis=1).max()

    se_compact = pd.DataFrame(
        out[:, :max_se], columns=[f"sideEffect{i}" for i in range(max_se)], index=df.index
    )
    return pd.concat([df.drop(columns=se_cols), se_compact], axis=1)


def clean_uses(df, use_cols):
    arr = df[use_cols].values

    for i in range(len(arr)):
        seen = set()
        for j in range(len(arr[i])):
            cell = arr[i, j]
            if not pd.notna(cell) or not cell:
                continue
            pairs = normalize_use(cell)
            if not pairs:
                arr[i, j] = None
                continue

            _, condition = pairs[0]
            if condition in seen:
                arr[i, j] = None
            else:
                arr[i, j] = cell
                seen.add(condition)

    df[use_cols] = arr
    return df


def clean_chemical_class(df):
    df["Chemical Class"] = df["Chemical Class"].apply(normalize_chemical_class)
    return df


def filter_chemical_class(df, min_frequency=MIN_CC_FREQ):
    counts = df["Chemical Class"].value_counts()
    valid = set(counts[counts >= min_frequency].index)
    df["Chemical Class"] = df["Chemical Class"].where(df["Chemical Class"].isin(valid))
    return df


def filter_uses(df, use_cols, min_frequency=MIN_USE_FREQ):
    freq = {}
    for col in use_cols:
        for val in df[col].dropna():
            for _, condition in normalize_use(val):
                freq[condition] = freq.get(condition, 0) + 1

    valid = {c for c, n in freq.items() if n >= min_frequency}

    for col in use_cols:
        df[col] = df[col].apply(
            lambda v: v if pd.notna(v) and any(c in valid for _, c in normalize_use(v)) else np.nan
        )
    return df


def make_dataset():
    medicine = pd.read_csv(RAW_MEDICINE_CSV, low_memory=False)
    composition = pd.read_csv(RAW_COMPOSITION_CSV)

    medicine["name"] = medicine["name"].str.lower()
    composition["name"] = composition["name"].str.lower()
    merged = pd.merge(medicine, composition, on="name", how="inner")

    # Preprocessing
    se_cols = [c for c in merged.columns if c.startswith("sideEffect")]
    use_cols = [c for c in merged.columns if c.startswith("use")]
    str_cols = merged.select_dtypes(include="object").columns

    merged[str_cols] = merged[str_cols].apply(
        lambda x: x.str.lower().str.strip().replace("  ", " ")
    )

    merged = (
        merged.drop_duplicates()
        .drop_duplicates(subset="name", keep="first")
        .pipe(clean_chemical_class)
        .pipe(filter_chemical_class)
        .pipe(clean_uses, use_cols)
        .pipe(filter_uses, use_cols)
        .pipe(clean_se, se_cols)
        .pipe(filter_se, se_cols)
        .dropna(subset=se_cols, how="all")
        .dropna(subset=["Chemical Class"], how="all")
        .pipe(normalize_se_labels, se_cols)
        .pipe(compact_se_columns, se_cols)
    )

    # Dataset splitting
    medicine_dataset = merged[
        [
            "name",
            "pack_size_label",
            "short_composition1",
            "short_composition2",
            "substitute0",
            "use0",
            "Chemical Class",
            "Therapeutic Class",
        ]
    ]

    side_effects = merged[["name"] + [c for c in merged.columns if c.startswith("sideEffect")]]

    medicine_dataset.to_csv(PROCESSED_MEDICINE_CSV, index=False)
    side_effects.to_csv(SIDE_EFFETCS_MEDICINE_CSV, index=False)

    return (medicine_dataset.shape, side_effects.shape)


if __name__ == "__main__":
    dfs = make_dataset()

    print(f"""
        Datasets created:
        medicine: {dfs[0]}
        side_effects: {dfs[1]}
    """)
