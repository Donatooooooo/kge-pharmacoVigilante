import numpy as np
import pandas as pd


def mean_rank(ranks_list):
    """Mean Rank (MR): Average rank across all true positives."""
    all_ranks = np.concatenate([r for r in ranks_list if len(r) > 0])
    if len(all_ranks) == 0:
        return float("nan")
    return float(np.mean(all_ranks))


def mean_reciprocal_rank(ranks_list):
    """MRR: Average of 1/rank for the best-ranked true positive per sample."""
    reciprocals = []
    for ranks in ranks_list:
        if len(ranks) > 0:
            best_rank = np.min(ranks)
            reciprocals.append(1.0 / best_rank)

    if len(reciprocals) == 0:
        return float("nan")
    return float(np.mean(reciprocals))


def hits_at_k(ranks_list, k):
    """Hits@K: Fraction of samples with at least one TP in top K."""
    hits = 0
    valid_samples = 0

    for ranks in ranks_list:
        if len(ranks) > 0:
            valid_samples += 1
            if np.min(ranks) <= k:
                hits += 1

    if valid_samples == 0:
        return float("nan")
    return hits / valid_samples


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


def compute_ranks_per_label(y_true, y_scores):
    """
    Compute ranks for each label separately.

    Returns:
        Dictionary mapping label_idx -> list of ranks for that label
    """
    n_samples, n_labels = y_true.shape
    ranks_per_label = {i: [] for i in range(n_labels)}

    for sample_idx in range(n_samples):
        sorted_indices = np.argsort(-y_scores[sample_idx])

        ranks = np.empty(n_labels, dtype=int)
        ranks[sorted_indices] = np.arange(1, n_labels + 1)

        for label_idx in range(n_labels):
            if y_true[sample_idx, label_idx] == 1:
                ranks_per_label[label_idx].append(ranks[label_idx])

    return ranks_per_label


def compute_ranks(y_true, y_scores):
    """
    Compute the rank of each true positive side effect for each drug.
    """
    n_samples, n_labels = y_true.shape
    all_ranks = []

    for i in range(n_samples):
        sorted_indices = np.argsort(-y_scores[i])
        ranks = np.empty(n_labels, dtype=int)
        ranks[sorted_indices] = np.arange(1, n_labels + 1)

        true_positive_mask = y_true[i] == 1
        if true_positive_mask.any():
            all_ranks.append(ranks[true_positive_mask])
        else:
            all_ranks.append(np.array([]))

    return all_ranks


def evaluate_ranking(y_true, y_scores):
    """Compute all ranking metrics."""
    ranks_list = compute_ranks(y_true, y_scores)

    return {
        "MR": mean_rank(ranks_list),
        "MRR": mean_reciprocal_rank(ranks_list),
        "Hits@1": hits_at_k(ranks_list, 1),
        "Hits@3": hits_at_k(ranks_list, 3),
        "Hits@5": hits_at_k(ranks_list, 5),
        "Hits@10": hits_at_k(ranks_list, 10),
    }


def evaluate_stratified(
    y_true,
    y_scores,
    strata,
    label_counts,
):

    ranks_per_label = compute_ranks_per_label(y_true, y_scores)

    results = []
    for stratum_name, label_indices in strata.items():
        if len(label_indices) == 0:
            continue

        # Collect all ranks for labels in this stratum
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
            }
        )

    return pd.DataFrame(results)


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
    """Compute ranking metrics using only test_mask positives."""
    ranks_list = compute_ranks_masked(y_scores, test_mask)

    return {
        "MR": mean_rank(ranks_list),
        "MRR": mean_reciprocal_rank(ranks_list),
        "Hits@1": hits_at_k(ranks_list, 1),
        "Hits@3": hits_at_k(ranks_list, 3),
        "Hits@5": hits_at_k(ranks_list, 5),
        "Hits@10": hits_at_k(ranks_list, 10),
    }


def compute_ranks_per_label_masked(y_scores, test_mask):
    """Compute ranks for each label, but only for test-positive pairs."""
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
