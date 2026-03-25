import numpy as np
import pandas as pd
from scipy.stats import norm


def mean_rank_from_flat(ranks):
    """MR from flat list of ranks."""
    if len(ranks) == 0:
        return float("nan")
    return float(np.mean(ranks))


def mrr_from_flat(ranks):
    """MRR from flat list of ranks (treats each as independent)."""
    if len(ranks) == 0:
        return float("nan")
    return float(np.mean([1.0 / r for r in ranks]))


def hits_at_k_flat(ranks, k):
    """Hits@K: fraction of test triples with rank <= k."""
    if len(ranks) == 0:
        return float("nan")
    return float(np.mean(np.asarray(ranks) <= k))


# =========================================================================
#  Masked evaluation (held-out test pairs only)
# =========================================================================
def compute_ranks_masked(y_scores, test_mask):
    """
    For each drug with at least one positive in test_mask, rank ALL SEs
    by score (descending) and return the ranks of the test-positive SEs.
    """
    n_samples, n_labels = y_scores.shape
    all_ranks = []

    for i in range(n_samples):
        test_positives = test_mask[i] == 1
        if not test_positives.any():
            all_ranks.append(np.array([]))
            continue

        sorted_indices = np.argsort(-y_scores[i])
        ranks = np.empty(n_labels, dtype=int)
        ranks[sorted_indices] = np.arange(1, n_labels + 1)

        all_ranks.append(ranks[test_positives])

    return all_ranks


def evaluate_ranking_masked(y_scores, test_mask):
    ranks_list = compute_ranks_masked(y_scores, test_mask)
    flat_ranks = np.concatenate([r for r in ranks_list if len(r) > 0])

    return {
        "MR": mean_rank_from_flat(flat_ranks),
        "MRR": mrr_from_flat(flat_ranks),
        "Hits@1": hits_at_k_flat(flat_ranks, 1),
        "Hits@3": hits_at_k_flat(flat_ranks, 3),
        "Hits@5": hits_at_k_flat(flat_ranks, 5),
        "Hits@10": hits_at_k_flat(flat_ranks, 10),
    }


def paired_ztest_reciprocal_rank(y_scores_a, y_scores_b, test_mask):
    """Paired z-test on per-drug mean reciprocal rank between two models."""
    ranks_a = compute_ranks_masked(y_scores_a, test_mask)
    ranks_b = compute_ranks_masked(y_scores_b, test_mask)

    per_drug_rr_a = np.array([np.mean(1.0 / r) for r in ranks_a if len(r) > 0])
    per_drug_rr_b = np.array([np.mean(1.0 / r) for r in ranks_b if len(r) > 0])

    diff = per_drug_rr_a - per_drug_rr_b
    n = len(diff)
    z_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    cohen_d = diff.mean() / diff.std(ddof=1)

    return {
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "cohen_d": float(cohen_d),
        "mean_rr_a": float(per_drug_rr_a.mean()),
        "mean_rr_b": float(per_drug_rr_b.mean()),
        "n_drugs": n,
    }


def compute_ranks_per_label_masked(y_scores, test_mask):
    n_samples, n_labels = y_scores.shape
    ranks_per_label = {i: [] for i in range(n_labels)}

    for sample_idx in range(n_samples):
        sorted_indices = np.argsort(-y_scores[sample_idx])
        ranks = np.empty(n_labels, dtype=int)
        ranks[sorted_indices] = np.arange(1, n_labels + 1)

        for label_idx in range(n_labels):
            if test_mask[sample_idx, label_idx] == 1:
                ranks_per_label[label_idx].append(ranks[label_idx])

    return ranks_per_label


def evaluate_stratified_masked(
    y_scores,
    test_mask,
    strata,
    label_counts,
):
    """
    Stratified evaluation using only held-out test pairs.
    Strata defined by training set frequencies (label_counts from y_train).
    """
    ranks_per_label = compute_ranks_per_label_masked(y_scores, test_mask)

    results = []
    for stratum_name, label_indices in strata.items():
        if len(label_indices) == 0:
            continue

        all_ranks = []
        total_positives = 0
        for label_idx in label_indices:
            all_ranks.extend(ranks_per_label[label_idx])
            total_positives += len(ranks_per_label[label_idx])

        if len(all_ranks) == 0:
            continue

        results.append(
            {
                "Stratum": stratum_name,
                "N Labels": len(label_indices),
                "N Positives": total_positives,
                "MR": mean_rank_from_flat(all_ranks),
                "MRR": mrr_from_flat(all_ranks),
                "Hits@10": hits_at_k_flat(all_ranks, 10),
            }
        )

    return pd.DataFrame(results)


def print_metrics(name, metrics):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  MR:       {metrics['MR']:.2f}")
    print(f"  MRR:      {metrics['MRR']:.4f}")
    print(f"  Hits@1:   {metrics['Hits@1']:.4f}")
    print(f"  Hits@3:   {metrics['Hits@3']:.4f}")
    print(f"  Hits@5:   {metrics['Hits@5']:.4f}")
    print(f"  Hits@10:  {metrics['Hits@10']:.4f}")


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
