#!/usr/bin/env python3
"""Multi-asset correlation matrix analysis with hierarchical clustering.

Computes correlation matrices (Pearson and Spearman) across FX/metals
instruments, performs hierarchical clustering to identify instrument groups,
and reports diversification metrics.

Usage:
    python scripts/correlation_matrix.py --demo

Dependencies:
    uv pip install pandas numpy scipy

To use with real data, load your own multi-instrument return DataFrame
(e.g. from an MT5 export via quantlab's data loaders) in place of
generate_demo_data().
"""

import argparse
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# ── Configuration ───────────────────────────────────────────────────
DAYS = 180


# ── Data Generation ────────────────────────────────────────────────
def generate_demo_data(n_assets: int = 8, n_days: int = 180) -> pd.DataFrame:
    """Generate synthetic correlated FX/metals return data for demo mode.

    Creates instruments with an illustrative FX/metals correlation structure:
    - A broad-USD factor (USD-quote pairs load negatively, USD-base pairs
      load positively)
    - A risk-on/off factor (risk-sensitive currencies vs. safe havens)
    - A metals factor shared by gold/silver
    - Idiosyncratic noise per instrument

    This is illustrative only — measure actual correlation structure on
    your own instruments rather than assuming these loadings.

    Args:
        n_assets: Number of synthetic instruments.
        n_days: Number of days of data.

    Returns:
        DataFrame of daily returns with instrument columns.
    """
    rng = np.random.default_rng(42)
    names = [
        "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
        "USDCAD", "USDCHF", "XAUUSD", "XAGUSD",
    ][:n_assets]

    # Broad factors
    usd_factor = rng.normal(0.0002, 0.006, n_days)    # broad USD strength
    risk_factor = rng.normal(0.0000, 0.005, n_days)   # risk-on/off
    metals_factor = rng.normal(0.0003, 0.009, n_days)  # gold/silver co-driver

    # Instrument-specific loadings on factors
    loadings = {
        "EURUSD": {"usd": -0.80, "risk": 0.10, "metals": 0.00, "idio": 0.005},
        "GBPUSD": {"usd": -0.75, "risk": 0.15, "metals": 0.00, "idio": 0.006},
        "AUDUSD": {"usd": -0.65, "risk": 0.55, "metals": 0.10, "idio": 0.007},
        "USDJPY": {"usd": 0.70, "risk": -0.35, "metals": 0.00, "idio": 0.005},
        "USDCAD": {"usd": 0.60, "risk": -0.20, "metals": -0.15, "idio": 0.006},
        "USDCHF": {"usd": 0.75, "risk": -0.25, "metals": 0.00, "idio": 0.005},
        "XAUUSD": {"usd": -0.50, "risk": -0.20, "metals": 0.80, "idio": 0.009},
        "XAGUSD": {"usd": -0.40, "risk": -0.10, "metals": 0.75, "idio": 0.014},
    }

    returns_data: dict[str, np.ndarray] = {}
    for name in names:
        l = loadings[name]
        asset_return = (
            l["usd"] * usd_factor
            + l["risk"] * risk_factor
            + l["metals"] * metals_factor
            + rng.normal(0, l["idio"], n_days)
        )
        returns_data[name] = asset_return

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(returns_data, index=dates)


# ── Correlation Analysis ────────────────────────────────────────────
def compute_correlation_matrices(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlation matrices.

    Args:
        returns: DataFrame of asset returns.

    Returns:
        Tuple of (pearson_corr, spearman_corr) DataFrames.
    """
    pearson = returns.corr(method="pearson")
    spearman = returns.corr(method="spearman")
    return pearson, spearman


def find_extreme_pairs(
    corr_matrix: pd.DataFrame, n: int = 5
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """Find the most and least correlated pairs.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        n: Number of pairs to return.

    Returns:
        Tuple of (strongest_pairs, weakest_pairs), each a list of
        (asset_a, asset_b, correlation) tuples.
    """
    pairs: list[tuple[str, str, float]] = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    strongest = pairs_sorted[:n]
    weakest = pairs_sorted[-n:]
    return strongest, weakest


# ── Hierarchical Clustering ────────────────────────────────────────
def cluster_assets(
    corr_matrix: pd.DataFrame, max_clusters: int = 5
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Perform hierarchical clustering on assets based on correlation.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        max_clusters: Maximum number of clusters.

    Returns:
        Tuple of (linkage_matrix, distance_matrix, cluster_labels).
    """
    # Convert correlation to distance
    dist_matrix = np.sqrt(2.0 * (1.0 - corr_matrix.values))
    np.fill_diagonal(dist_matrix, 0.0)

    # Ensure symmetry (numerical precision)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0

    condensed = squareform(dist_matrix)
    linkage_matrix = linkage(condensed, method="ward")

    # Determine optimal number of clusters (up to max_clusters)
    labels = fcluster(linkage_matrix, t=max_clusters, criterion="maxclust")
    return linkage_matrix, dist_matrix, labels.tolist()


def print_dendrogram_text(
    linkage_matrix: np.ndarray, labels: list[str]
) -> None:
    """Print a text-based representation of the clustering hierarchy.

    Args:
        linkage_matrix: Linkage matrix from scipy.
        labels: Asset names.
    """
    n = len(labels)
    clusters: dict[int, str] = {i: labels[i] for i in range(n)}

    print("\n  Clustering Hierarchy (Ward linkage):")
    print("  " + "-" * 55)
    for i, row in enumerate(linkage_matrix):
        left, right, dist, count = int(row[0]), int(row[1]), row[2], int(row[3])
        left_label = clusters.get(left, f"C{left}")
        right_label = clusters.get(right, f"C{right}")
        merged_label = f"({left_label} + {right_label})"
        clusters[n + i] = merged_label
        print(f"  d={dist:.3f}: {left_label}  <->  {right_label}")
    print()


# ── Diversification Metrics ────────────────────────────────────────
def diversification_metrics(corr_matrix: pd.DataFrame) -> dict[str, float]:
    """Compute portfolio diversification metrics from correlation matrix.

    Args:
        corr_matrix: Correlation matrix DataFrame.

    Returns:
        Dictionary with diversification metrics.
    """
    n = corr_matrix.shape[0]
    corr_values = corr_matrix.values

    # Average pairwise correlation (off-diagonal)
    mask = ~np.eye(n, dtype=bool)
    avg_corr = corr_values[mask].mean()

    # Effective N from average correlation
    if avg_corr < 1.0:
        effective_n_simple = 1.0 / (1.0 / n + (1.0 - 1.0 / n) * avg_corr)
    else:
        effective_n_simple = 1.0

    # Effective N from eigenvalue entropy
    eigenvalues = np.linalg.eigvalsh(corr_values)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    proportions = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(proportions * np.log(proportions))
    effective_n_eigen = np.exp(entropy)

    # First eigenvalue dominance (market factor strength)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    first_eigen_pct = eigenvalues_sorted[0] / eigenvalues_sorted.sum() * 100

    return {
        "n_assets": n,
        "avg_pairwise_corr": avg_corr,
        "effective_n_simple": effective_n_simple,
        "effective_n_eigenvalue": effective_n_eigen,
        "first_eigenvalue_pct": first_eigen_pct,
        "max_corr": corr_values[mask].max(),
        "min_corr": corr_values[mask].min(),
    }


# ── Eigenvalue Analysis ────────────────────────────────────────────
def eigenvalue_analysis(corr_matrix: pd.DataFrame) -> None:
    """Print eigenvalue decomposition of correlation matrix.

    Args:
        corr_matrix: Correlation matrix DataFrame.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total = eigenvalues.sum()
    cumulative = 0.0

    print("\n  Eigenvalue Decomposition:")
    print("  " + "-" * 55)
    print(f"  {'Factor':<10} {'Eigenvalue':>10} {'% Var':>8} {'Cum %':>8}")
    print("  " + "-" * 55)
    for i, ev in enumerate(eigenvalues):
        pct = ev / total * 100
        cumulative += pct
        label = "Market" if i == 0 else f"Factor {i}"
        print(f"  {label:<10} {ev:>10.3f} {pct:>7.1f}% {cumulative:>7.1f}%")
    print()


# ── Display Functions ───────────────────────────────────────────────
def print_correlation_heatmap(corr_matrix: pd.DataFrame, title: str) -> None:
    """Print a text-based correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        title: Title for the heatmap.
    """
    labels = corr_matrix.columns.tolist()
    max_label = max(len(l) for l in labels)

    print(f"\n  {title}")
    print("  " + "-" * (max_label + 2 + len(labels) * 7))

    # Header
    header = " " * (max_label + 2)
    for label in labels:
        header += f"{label:>7}"
    print(f"  {header}")

    # Rows
    for i, row_label in enumerate(labels):
        row_str = f"  {row_label:>{max_label}}  "
        for j in range(len(labels)):
            val = corr_matrix.iloc[i, j]
            if i == j:
                row_str += "   1.00"
            elif val >= 0.8:
                row_str += f"  {val:5.2f}"  # Strong positive
            elif val >= 0.5:
                row_str += f"  {val:5.2f}"
            elif val <= -0.3:
                row_str += f" {val:5.2f}"
            else:
                row_str += f"  {val:5.2f}"
        print(row_str)
    print()

    # Legend
    print("  Interpretation: >0.8 strong | 0.5-0.8 moderate | <0.5 weak")


def print_cluster_report(
    labels: list[str], clusters: list[int]
) -> None:
    """Print cluster membership report.

    Args:
        labels: Asset names.
        clusters: Cluster assignment for each asset.
    """
    print("\n  Asset Clusters:")
    print("  " + "-" * 40)
    cluster_map: dict[int, list[str]] = {}
    for asset, cluster in zip(labels, clusters):
        cluster_map.setdefault(cluster, []).append(asset)

    for cluster_id in sorted(cluster_map.keys()):
        members = cluster_map[cluster_id]
        print(f"  Cluster {cluster_id}: {', '.join(members)}")
    print()


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    """Run multi-asset correlation analysis."""
    parser = argparse.ArgumentParser(
        description="Multi-asset correlation matrix analysis"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use synthetic demo data (currently the only supported mode)"
    )
    parser.add_argument(
        "--days", type=int, default=DAYS,
        help=f"Days of history to generate (default: {DAYS})"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  MULTI-ASSET CORRELATION ANALYSIS")
    print("=" * 65)

    print("\n  Mode: DEMO (synthetic FX/metals data)")
    print("  Swap generate_demo_data() for your own MT5/quantlab return")
    print("  DataFrame to run this on real instruments.")
    returns = generate_demo_data(n_days=args.days)
    asset_names = returns.columns.tolist()

    print(f"\n  Assets: {len(asset_names)}")
    print(f"  Observations: {len(returns)}")
    print(f"  Date range: {returns.index[0].strftime('%Y-%m-%d')} to "
          f"{returns.index[-1].strftime('%Y-%m-%d')}")

    # Compute correlation matrices
    pearson_corr, spearman_corr = compute_correlation_matrices(returns)

    # Display heatmaps
    print_correlation_heatmap(pearson_corr, "Pearson Correlation Matrix")
    print_correlation_heatmap(spearman_corr, "Spearman Rank Correlation Matrix")

    # Strongest and weakest pairs
    strongest, weakest = find_extreme_pairs(pearson_corr)
    print("\n  Strongest Correlated Pairs (Pearson):")
    print("  " + "-" * 40)
    for a, b, corr in strongest:
        print(f"  {a:>6} / {b:<6}  r = {corr:+.3f}")

    print("\n  Weakest Correlated Pairs (Pearson):")
    print("  " + "-" * 40)
    for a, b, corr in weakest:
        print(f"  {a:>6} / {b:<6}  r = {corr:+.3f}")

    # Eigenvalue analysis
    eigenvalue_analysis(pearson_corr)

    # Hierarchical clustering
    linkage_mat, dist_mat, clusters = cluster_assets(pearson_corr)
    print_dendrogram_text(linkage_mat, asset_names)
    print_cluster_report(asset_names, clusters)

    # Diversification metrics
    metrics = diversification_metrics(pearson_corr)
    print("\n  Diversification Metrics:")
    print("  " + "-" * 50)
    print(f"  Number of assets:             {metrics['n_assets']}")
    print(f"  Avg pairwise correlation:     {metrics['avg_pairwise_corr']:.3f}")
    print(f"  Effective N (simple):         {metrics['effective_n_simple']:.1f}")
    print(f"  Effective N (eigenvalue):     {metrics['effective_n_eigenvalue']:.1f}")
    print(f"  Market factor dominance:      {metrics['first_eigenvalue_pct']:.1f}%")
    print(f"  Max pairwise correlation:     {metrics['max_corr']:.3f}")
    print(f"  Min pairwise correlation:     {metrics['min_corr']:.3f}")

    # Assessment
    print("\n  Assessment:")
    print("  " + "-" * 50)
    avg = metrics["avg_pairwise_corr"]
    if avg > 0.7:
        print("  WARNING: High average correlation — portfolio is poorly diversified.")
        print("  Consider adding uncorrelated instruments (different currency blocs, metals, B3).")
    elif avg > 0.4:
        print("  Moderate diversification. Some correlated clusters present.")
        print("  Consider reducing within-cluster allocations.")
    else:
        print("  Good diversification. Assets provide independent return streams.")

    mkt = metrics["first_eigenvalue_pct"]
    if mkt > 70:
        print(f"  Market factor explains {mkt:.0f}% of variance — all assets move together.")
    elif mkt > 50:
        print(f"  Market factor explains {mkt:.0f}% — significant common driver.")

    print("\n  Note: This is analysis output, not financial advice.")
    print("=" * 65)


if __name__ == "__main__":
    main()
