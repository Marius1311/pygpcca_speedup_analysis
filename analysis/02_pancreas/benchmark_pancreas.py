"""Benchmark: NM vs L-BFGS-B on the pancreas dataset (real data).

Compares Nelder-Mead and L-BFGS-B optimization for GPCCA macrostate
computation on the CellRank pancreas dataset (~2500 cells) using the
PseudotimeKernel with palantir_pseudotime.

Usage
-----
    pixi run python analysis/02_pancreas/benchmark_pancreas.py [--mini]

The ``--mini`` flag runs a quick test with 1 repeat and 2 values of m.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import cellrank as cr

warnings.filterwarnings("ignore", category=UserWarning)

PYTHON = sys.executable
METHODS = ["Nelder-Mead", "L-BFGS-B"]


def load_and_setup():
    """Load pancreas data and compute PseudotimeKernel transition matrix.

    Returns
    -------
    g
        GPCCA estimator with Schur decomposition precomputed.
    adata
        The AnnData object.
    """
    print("Loading pancreas dataset...")
    adata = cr.datasets.pancreas(kind="raw")
    print(f"  Shape: {adata.shape}")
    print(f"  Clusters: {sorted(adata.obs['clusters'].unique())}")

    # Need neighbors for the PseudotimeKernel
    import scanpy as sc

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)

    print("Computing PseudotimeKernel transition matrix...")
    pk = cr.kernels.PseudotimeKernel(adata, time_key="palantir_pseudotime")
    pk.compute_transition_matrix()

    print("Initializing GPCCA estimator and computing Schur decomposition...")
    g = cr.estimators.GPCCA(pk)
    g.compute_schur(n_components=20)

    return g, adata


def name_macrostates(memberships, clusters, n_cells=30):
    """Assign names to macrostates based on cluster overlap.

    Parameters
    ----------
    memberships
        Macrostate membership matrix, shape (n_cells, m).
    clusters
        Cluster labels for each cell.
    n_cells
        Number of top cells to use for naming.

    Returns
    -------
    names
        List of macrostate names.
    """
    names = []
    for j in range(memberships.shape[1]):
        col = memberships[:, j]
        top_idx = np.argsort(col)[-n_cells:]
        top_clusters = clusters.iloc[top_idx]
        name = top_clusters.value_counts().index[0]
        names.append(name)
    return names


def run_single(g, m, optimizer, cluster_key="clusters"):
    """Run a single macrostate computation and return results.

    Parameters
    ----------
    g
        GPCCA estimator with precomputed Schur decomposition.
    m
        Number of macrostates.
    optimizer
        Optimization method name.
    cluster_key
        Key in adata.obs for cluster labels.

    Returns
    -------
    result
        Dict with keys: time, crispness, memberships, names.
    """
    t0 = time.perf_counter()
    g.compute_macrostates(
        n_states=m,
        cluster_key=cluster_key,
        optimizer=optimizer,
    )
    elapsed = time.perf_counter() - t0

    crispness = g._gpcca.optimal_crispness
    memberships = np.array(g.macrostates_memberships)
    names = list(g.macrostates_memberships.names)
    names = [str(n) for n in names]
    macrostates = g.macrostates.copy()  # discrete assignment (pd.Series, NaN for unassigned)

    return {
        "time": elapsed,
        "crispness": crispness,
        "memberships": memberships,
        "names": names,
        "macrostates": macrostates,
    }


def correlate_memberships(res_nm, res_lb):
    """Match macrostates by name and compute membership correlations.

    Parameters
    ----------
    res_nm
        Result dict from NM run.
    res_lb
        Result dict from L-BFGS-B run.

    Returns
    -------
    correlations
        Dict mapping macrostate name to Pearson correlation.
    unmatched_nm
        Names in NM but not in L-BFGS-B.
    unmatched_lb
        Names in L-BFGS-B but not in NM.
    """
    nm_names = res_nm["names"]
    lb_names = res_lb["names"]
    nm_memb = res_nm["memberships"]
    lb_memb = res_lb["memberships"]

    # Build name → column index maps
    nm_map = {name: i for i, name in enumerate(nm_names)}
    lb_map = {name: i for i, name in enumerate(lb_names)}

    common = set(nm_names) & set(lb_names)
    correlations = {}
    for name in sorted(common):
        r, _ = pearsonr(nm_memb[:, nm_map[name]], lb_memb[:, lb_map[name]])
        correlations[name] = r

    unmatched_nm = sorted(set(nm_names) - set(lb_names))
    unmatched_lb = sorted(set(lb_names) - set(nm_names))

    return correlations, unmatched_nm, unmatched_lb


def compute_jaccard(res_nm, res_lb):
    """Compute per-macrostate Jaccard index of discrete assignments.

    For each macrostate name present in both results, compute the Jaccard
    index of the sets of cells assigned to that macrostate (top-30 cells).

    Parameters
    ----------
    res_nm
        Result dict from NM run (must contain 'macrostates' key).
    res_lb
        Result dict from L-BFGS-B run (must contain 'macrostates' key).

    Returns
    -------
    jaccards
        Dict mapping macrostate name to Jaccard index.
    unmatched_nm
        Names in NM but not in L-BFGS-B.
    unmatched_lb
        Names in L-BFGS-B but not in NM.
    """
    ms_nm = res_nm["macrostates"]
    ms_lb = res_lb["macrostates"]

    nm_names = set(ms_nm.dropna().unique())
    lb_names = set(ms_lb.dropna().unique())
    common = sorted(nm_names & lb_names)

    jaccards = {}
    for name in common:
        cells_nm = set(ms_nm.index[ms_nm == name])
        cells_lb = set(ms_lb.index[ms_lb == name])
        intersection = len(cells_nm & cells_lb)
        union = len(cells_nm | cells_lb)
        jaccards[name] = intersection / union if union > 0 else 0.0

    unmatched_nm = sorted(nm_names - lb_names)
    unmatched_lb = sorted(lb_names - nm_names)

    return jaccards, unmatched_nm, unmatched_lb


def run_experiment(g, m_values, n_repeats=3):
    """Run the full experiment across m values and repeats.

    Parameters
    ----------
    g
        GPCCA estimator with precomputed Schur decomposition.
    m_values
        List of m values to test.
    n_repeats
        Number of repeats per (m, method) combination.

    Returns
    -------
    results
        Nested dict: results[m][method] = list of result dicts.
    """
    results = {}
    for m in m_values:
        results[m] = {method: [] for method in METHODS}
        for rep in range(n_repeats):
            for method in METHODS:
                print(f"  m={m}, {method}, repeat {rep + 1}/{n_repeats}...", end=" ", flush=True)
                res = run_single(g, m, method)
                print(f"{res['time']:.2f}s, crispness={res['crispness']:.4f}")
                results[m][method].append(res)
    return results


def format_runtime_table(results):
    """Format Table A: runtime comparison.

    Parameters
    ----------
    results
        Results dict from run_experiment.

    Returns
    -------
    table
        Markdown table string.
    """
    lines = ["## Table A: Runtime and Crispness", ""]
    lines.append("| m | NM time (s) | L-BFGS-B time (s) | Speedup | NM crispness | L-BFGS-B crispness |")
    lines.append("|--:|------------:|-------------------:|--------:|-------------:|-------------------:|")

    for m in sorted(results.keys()):
        nm_times = [r["time"] for r in results[m]["Nelder-Mead"]]
        lb_times = [r["time"] for r in results[m]["L-BFGS-B"]]
        nm_crisp = [r["crispness"] for r in results[m]["Nelder-Mead"]]
        lb_crisp = [r["crispness"] for r in results[m]["L-BFGS-B"]]

        nm_t = f"{np.mean(nm_times):.2f} ± {np.std(nm_times):.2f}"
        lb_t = f"{np.mean(lb_times):.2f} ± {np.std(lb_times):.2f}"
        speedup = np.mean(nm_times) / np.mean(lb_times)
        nm_c = f"{np.mean(nm_crisp):.4f}"
        lb_c = f"{np.mean(lb_crisp):.4f}"

        lines.append(f"| {m} | {nm_t} | {lb_t} | {speedup:.1f}× | {nm_c} | {lb_c} |")

    return "\n".join(lines)


def format_correlation_table(results):
    """Format Table B: membership correlations.

    Parameters
    ----------
    results
        Results dict from run_experiment.

    Returns
    -------
    table
        Markdown table string.
    """
    lines = ["## Table B: Macrostate Membership Correlations", ""]

    for m in sorted(results.keys()):
        # Use the first repeat for correlation
        res_nm = results[m]["Nelder-Mead"][0]
        res_lb = results[m]["L-BFGS-B"][0]

        correlations, unmatched_nm, unmatched_lb = correlate_memberships(res_nm, res_lb)

        lines.append(f"### m = {m}")
        lines.append("")
        lines.append(f"NM macrostates: {res_nm['names']}")
        lines.append(f"L-BFGS-B macrostates: {res_lb['names']}")
        lines.append("")

        if correlations:
            lines.append("| Macrostate | Pearson r |")
            lines.append("|:-----------|----------:|")
            for name, r in correlations.items():
                lines.append(f"| {name} | {r:.4f} |")
            lines.append("")

        if unmatched_nm:
            lines.append(f"Unmatched (NM only): {unmatched_nm}")
        if unmatched_lb:
            lines.append(f"Unmatched (L-BFGS-B only): {unmatched_lb}")
        lines.append("")

    return "\n".join(lines)


def format_jaccard_table(results):
    """Format Table C: per-macrostate Jaccard indices of discrete assignments.

    Parameters
    ----------
    results
        Results dict from run_experiment.

    Returns
    -------
    table
        Markdown table string.
    """
    lines = ["## Table C: Discrete Assignment Jaccard Indices (top-30 cells)", ""]

    for m in sorted(results.keys()):
        res_nm = results[m]["Nelder-Mead"][0]
        res_lb = results[m]["L-BFGS-B"][0]

        jaccards, unmatched_nm, unmatched_lb = compute_jaccard(res_nm, res_lb)

        lines.append(f"### m = {m}")
        lines.append("")

        if jaccards:
            lines.append("| Macrostate | Jaccard | |NM ∩ LB| | |NM ∪ LB| |")
            lines.append("|:-----------|--------:|---------:|---------:|")
            ms_nm = res_nm["macrostates"]
            ms_lb = res_lb["macrostates"]
            for name in sorted(jaccards.keys()):
                cells_nm = set(ms_nm.index[ms_nm == name])
                cells_lb = set(ms_lb.index[ms_lb == name])
                isect = len(cells_nm & cells_lb)
                union = len(cells_nm | cells_lb)
                lines.append(f"| {name} | {jaccards[name]:.3f} | {isect} | {union} |")
            lines.append("")

        if unmatched_nm:
            lines.append(f"Unmatched (NM only): {unmatched_nm}")
        if unmatched_lb:
            lines.append(f"Unmatched (L-BFGS-B only): {unmatched_lb}")
        lines.append("")

    return "\n".join(lines)


def main():
    mini = "--mini" in sys.argv

    if mini:
        m_values = [4, 8]
        n_repeats = 1
        print("=== MINI MODE: m_values={}, n_repeats={} ===\n".format(m_values, n_repeats))
    else:
        m_values = [3, 4, 5, 6, 8, 10, 12, 15]
        n_repeats = 3
        print("=== FULL MODE: m_values={}, n_repeats={} ===\n".format(m_values, n_repeats))

    g, adata = load_and_setup()

    print("\nRunning experiments...")
    results = run_experiment(g, m_values, n_repeats)

    print("\n" + "=" * 60)
    table_a = format_runtime_table(results)
    print(table_a)
    print()
    table_b = format_correlation_table(results)
    print(table_b)
    table_c = format_jaccard_table(results)
    print(table_c)


if __name__ == "__main__":
    main()
