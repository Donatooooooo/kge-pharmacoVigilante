import json

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.modeling.baseline import MarginalFrequencyBaseline
from src.modeling.util.data_loader import load_dataset
from src.modeling.util.evaluation import (
    convert_numpy,
    evaluate_ranking_masked,
    evaluate_stratified_masked,
    print_metrics,
)
from src.modeling.util.kge_scorer import score_all_pairs
from src.modeling.xgboost_cv import train_cv


def get_frequency_strata(y_train):
    counts = y_train.sum(axis=0)

    strata = {
        "Rare (1-10)": np.where((counts >= 1) & (counts <= 10))[0],
        "Uncommon (11-50)": np.where((counts > 10) & (counts <= 50))[0],
        "Moderate (51-200)": np.where((counts > 50) & (counts <= 200))[0],
        "Common (>200)": np.where(counts > 200)[0],
    }

    return strata, counts


def run(n_splits=10):

    print("Loading dataset...")
    X, y_train, y_full, test_mask, drug_names, se_names = load_dataset()

    n_train = int(y_train.sum())
    n_test = int(test_mask.sum())
    n_full = int(y_full.sum())
    overlap = int((y_train * test_mask).sum())

    strata, label_counts = get_frequency_strata(y_train)

    print(f"\n{'=' * 60}")
    print("  LABEL FREQUENCY DISTRIBUTION (Training Set)")
    print(f"{'=' * 60}")
    for stratum_name, indices in strata.items():
        print(f"  {stratum_name}: {len(indices)} labels")

    #  Baseline
    print(f"\n{'=' * 60}")
    print("  TRAINING MODELS")
    print(f"{'=' * 60}")

    print("\nFitting Marginal Frequency Baseline on training set...")
    baseline = MarginalFrequencyBaseline()
    baseline.fit(X, y_train)
    y_scores_baseline = baseline.predict_proba(X)

    #  Score KGE_SE
    print("\nScoring KGE_SE model...")
    y_scores_kge = score_all_pairs(drug_names, se_names)

    #  Train XGBoost
    print(f"\nTraining XGBoost ({n_splits}-Fold CV on training labels)...")
    y_scores_xgb, _ = train_cv(X, y_train, n_splits=n_splits)

    #  Evaluation all 3 on held-out test pairs
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION ON HELD-OUT TEST PAIRS (N={n_test})")
    print(f"{'=' * 60}")

    metrics_baseline = evaluate_ranking_masked(y_scores_baseline, test_mask)
    metrics_xgb = evaluate_ranking_masked(y_scores_xgb, test_mask)
    metrics_kge = evaluate_ranking_masked(y_scores_kge, test_mask)

    print_metrics("Marginal Frequency Baseline", metrics_baseline)
    print_metrics("XGBoost (OVR)", metrics_xgb)
    print_metrics("KGE_SE (RotatE)", metrics_kge)

    print(f"\n{'=' * 60}")
    print("  MODEL COMPARISONS (Global)")
    print(f"{'=' * 60}")

    models = {
        "Baseline": metrics_baseline,
        "XGBoost": metrics_xgb,
        "KGE_SE": metrics_kge,
    }

    for name_a, met_a in models.items():
        for name_b, met_b in models.items():
            if name_a >= name_b:
                continue
            delta_mrr = met_b["MRR"] - met_a["MRR"]
            rel = (delta_mrr / met_a["MRR"] * 100) if met_a["MRR"] > 0 else 0
            print(f"  {name_b} vs {name_a}: Δ MRR = {delta_mrr:+.4f} ({rel:+.1f}%)")

    print(f"\n{'=' * 60}")
    print("  STRATIFIED EVALUATION (Test Pairs, strata by training freq)")
    print(f"{'=' * 60}")

    strat_baseline = evaluate_stratified_masked(y_scores_baseline, test_mask, strata, label_counts)
    strat_xgb = evaluate_stratified_masked(y_scores_xgb, test_mask, strata, label_counts)
    strat_kge = evaluate_stratified_masked(y_scores_kge, test_mask, strata, label_counts)

    comparison_rows = []
    for stratum_name in strata.keys():
        row_b = strat_baseline[strat_baseline["Stratum"] == stratum_name]
        row_x = strat_xgb[strat_xgb["Stratum"] == stratum_name]
        row_k = strat_kge[strat_kge["Stratum"] == stratum_name]

        if len(row_b) == 0 or len(row_x) == 0 or len(row_k) == 0:
            continue

        comparison_rows.append({
            "Stratum": stratum_name,
            "N Labels": int(row_x["N Labels"].values[0]),
            "N Test Positives": int(row_x["N Positives"].values[0]),
            "Baseline MRR": float(row_b["MRR"].values[0]),
            "XGBoost MRR": float(row_x["MRR"].values[0]),
            "KGE_SE MRR": float(row_k["MRR"].values[0]),
            "Baseline MR": float(row_b["MR"].values[0]),
            "XGBoost MR": float(row_x["MR"].values[0]),
            "KGE_SE MR": float(row_k["MR"].values[0]),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    print("\n  MRR by Stratum:")
    print(comparison_df[["Stratum", "N Labels", "N Test Positives",
                          "Baseline MRR", "XGBoost MRR", "KGE_SE MRR"]].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:+.1f}"
    ))

    print(f"\n{'=' * 60}")
    print("  KEY INSIGHT")
    print(f"{'=' * 60}")

    rare_row = next((r for r in comparison_rows if "Rare" in r["Stratum"]), None)
    common_row = next((r for r in comparison_rows if "Common" in r["Stratum"]), None)

    if rare_row and common_row:
        print("\n  RARE side effects (1-10 training occurrences):")
        print(f"    Baseline MRR: {rare_row['Baseline MRR']:.4f}")
        print(f"    XGBoost MRR:  {rare_row['XGBoost MRR']:.4f}")
        print(f"    KGE_SE MRR:   {rare_row['KGE_SE MRR']:.4f}")

        print("\n  COMMON side effects (>200 training occurrences):")
        print(f"    Baseline MRR: {common_row['Baseline MRR']:.4f}")
        print(f"    XGBoost MRR:  {common_row['XGBoost MRR']:.4f}")
        print(f"    KGE_SE MRR:   {common_row['KGE_SE MRR']:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "ranking_comparison.json"

    report = convert_numpy({
        "split_info": {
            "description": (
                f"Training: 70% SE triples (N={n_train}), "
                f"Evaluation: 20% held-out test (N={n_test})"
            ),
            "train_se_associations": n_train,
            "test_se_pairs": n_test,
            "full_se_associations": n_full,
            "train_test_overlap": overlap,
        },
        "cv_folds": n_splits,
        "global_metrics": {
            "baseline": metrics_baseline,
            "xgboost": metrics_xgb,
            "kge_se": metrics_kge,
        },
        "stratified_comparison": comparison_rows,
        "label_distribution": {
            name: len(indices) for name, indices in strata.items()
        },
    })

    with open(str(output), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n- Results saved to {output}")

    return {
        "global": {
            "baseline": metrics_baseline,
            "xgboost": metrics_xgb,
            "kge_se": metrics_kge,
        },
        "stratified": comparison_df,
    }


if __name__ == "__main__":
    run()
