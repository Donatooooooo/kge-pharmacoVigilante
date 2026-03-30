import json

import numpy as np
import pandas as pd

from src.config import MODEL_SE, REPORTS_DIR, TRIPLES_TSV_SE, XGB
from src.modeling.baseline import MarginalFrequencyBaseline
from src.modeling.util.data_loader import load_dataset
from src.modeling.util.evaluation import (
    convert_numpy,
    evaluate_ranking_masked,
    evaluate_stratified_masked,
    paired_ttest_reciprocal_rank,
    print_metrics,
)
from src.modeling.util.kge_scorer import score_all_pairs
from src.modeling.xgboost_cv import XGBoostTrainer
from src.modeling.kge import LinkPredictor


def get_frequency_strata(y_train):
    counts = y_train.sum(axis=0)

    strata = {
        "Rare (1-50)": np.where((counts >= 1) & (counts <= 50))[0],
        "Uncommon (51-120)": np.where((counts > 50) & (counts <= 120))[0],
        "Moderate (121-500)": np.where((counts > 120) & (counts <= 500))[0],
        "Common (>500)": np.where(counts > 500)[0],
    }

    return strata, counts


def run():
    X, y_train, y_full, test_mask, drug_names, se_names = load_dataset()

    n_train = int(y_train.sum())
    n_test = int(test_mask.sum())
    n_full = int(y_full.sum())
    overlap = int((y_train * test_mask).sum())

    strata, label_counts = get_frequency_strata(y_train)

    # Baseline
    print("- Fitting Marginal Frequency Baseline on training set")
    baseline = MarginalFrequencyBaseline()
    baseline.fit(X, y_train)
    y_scores_baseline = baseline.predict_proba(X)

    # Approach A
    if not MODEL_SE.exists():
        print("- Training KGE side effects")
        kge = LinkPredictor(TRIPLES_TSV_SE, side_effects=True)
        kge.create_dataset()
        kge.train_model()

    print("\n- Scoring KGE side effects model")
    y_scores_kge = score_all_pairs(drug_names, se_names)

    # Approach B
    if XGB.exists():
        xgb = XGBoostTrainer.load()
    else:
        print(f"- Training XGBoost")
        xgb = XGBoostTrainer()
        xgb.search_and_train(X, y_train)

    print(f"- Scoring XGBoost")
    y_scores_xgb = xgb.predict_proba(X)


    # Evaluation
    metrics_baseline = evaluate_ranking_masked(y_scores_baseline, test_mask)
    metrics_xgb = evaluate_ranking_masked(y_scores_xgb, test_mask)
    metrics_kge = evaluate_ranking_masked(y_scores_kge, test_mask)

    print(f"- Evaluation on held-out test pairs (n = {n_test})")
    print_metrics("Baseline", metrics_baseline)
    print_metrics("Approach A", metrics_kge)
    print_metrics("Approach B", metrics_xgb)

    ttest_result = paired_ttest_reciprocal_rank(y_scores_xgb, y_scores_kge, test_mask)
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

    print("\n- Stratified evaluation")
    print(comparison_df[["Stratum", "N Labels", "N Test Positives",
                          "Baseline MRR", "XGBoost MRR", "KGE_SE MRR"]].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:+.1f}"
    ))

    report = convert_numpy({
        "split_info": {
            "train_se_associations": n_train,
            "test_se_pairs": n_test,
            "full_se_associations": n_full,
            "train_test_overlap": overlap,
        },
        "hpo": "5-fold CV",
        "paired_ttest_xgb_vs_kge": ttest_result,
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

    output = REPORTS_DIR / "ranking_comparison.json"
    with open(str(output), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"- Results saved {output}")
    return

if __name__ == "__main__":
    run()
