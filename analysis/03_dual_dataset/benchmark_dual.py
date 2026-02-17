"""Benchmark: NM(1 start) vs L-BFGS-B(10 starts) on pancreas + bone marrow.

Compares the current CellRank default (single NM) against the anticipated
new default (L-BFGS-B with 10 starts, eps=0.1, seed=0) on two datasets.

Usage
-----
    pixi run python analysis/03_dual_dataset/benchmark_dual.py
"""
import sys
import time
import warnings
import multiprocessing

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import cellrank as cr
import scanpy as sc

warnings.filterwarnings("ignore", category=UserWarning)

# --- Two configurations to compare ---
CONFIGS = {
    "NM (1 start)": dict(optimizer="Nelder-Mead"),
    "L-BFGS-B (10, eps=0.1)": dict(optimizer="L-BFGS-B", n_starts=10, perturbation_scale=0.1, seed=0),
}

M_VALUES = [3, 4, 5, 6, 8, 10, 12, 15]


def setup_pancreas():
    """Load pancreas data and return GPCCA estimator."""
    print("Loading pancreas dataset...")
    adata = cr.datasets.pancreas(kind="raw")
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    pk = cr.kernels.PseudotimeKernel(adata, time_key="palantir_pseudotime")
    pk.compute_transition_matrix()
    g = cr.estimators.GPCCA(pk)
    g.compute_schur(n_components=20)
    print(f"  Pancreas: {adata.shape[0]} cells, {adata.obs['clusters'].nunique()} clusters")
    return g


def setup_bone_marrow():
    """Load bone marrow data and return GPCCA estimator."""
    print("Loading bone marrow dataset...")
    adata = cr.datasets.bone_marrow()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    pk = cr.kernels.PseudotimeKernel(adata, time_key="palantir_pseudotime")
    pk.compute_transition_matrix()
    g = cr.estimators.GPCCA(pk)
    g.compute_schur(n_components=20)
    print(f"  Bone marrow: {adata.shape[0]} cells, {adata.obs['clusters'].nunique()} clusters")
    return g


def run_single(g, m, config_kwargs):
    """Run a single macrostate computation."""
    t0 = time.perf_counter()
    g.compute_macrostates(n_states=m, cluster_key="clusters", **config_kwargs)
    elapsed = time.perf_counter() - t0
    crispness = g._gpcca.optimal_crispness
    memberships = np.array(g.macrostates_memberships)
    names = [str(n) for n in g.macrostates_memberships.names]
    macrostates = g.macrostates.copy()
    return {
        "time": elapsed,
        "crispness": crispness,
        "memberships": memberships,
        "names": names,
        "macrostates": macrostates,
    }


def correlate_memberships(res_a, res_b):
    """Match macrostates by name and compute membership correlations."""
    a_map = {name: i for i, name in enumerate(res_a["names"])}
    b_map = {name: i for i, name in enumerate(res_b["names"])}
    common = sorted(set(res_a["names"]) & set(res_b["names"]))
    correlations = {}
    for name in common:
        r, _ = pearsonr(
            res_a["memberships"][:, a_map[name]],
            res_b["memberships"][:, b_map[name]],
        )
        correlations[name] = r
    unmatched_a = sorted(set(res_a["names"]) - set(res_b["names"]))
    unmatched_b = sorted(set(res_b["names"]) - set(res_a["names"]))
    return correlations, unmatched_a, unmatched_b


def compute_jaccard(res_a, res_b):
    """Compute per-macrostate Jaccard index of discrete assignments."""
    ms_a = res_a["macrostates"]
    ms_b = res_b["macrostates"]
    common = sorted(set(ms_a.dropna().unique()) & set(ms_b.dropna().unique()))
    jaccards = {}
    for name in common:
        cells_a = set(ms_a.index[ms_a == name])
        cells_b = set(ms_b.index[ms_b == name])
        isect = len(cells_a & cells_b)
        union = len(cells_a | cells_b)
        jaccards[name] = isect / union if union > 0 else 0.0
    unmatched_a = sorted(set(ms_a.dropna().unique()) - set(ms_b.dropna().unique()))
    unmatched_b = sorted(set(ms_b.dropna().unique()) - set(ms_a.dropna().unique()))
    return jaccards, unmatched_a, unmatched_b


def run_dataset(dataset_name, g, m_values):
    """Run all configs across all m values for one dataset."""
    config_names = list(CONFIGS.keys())
    results = {}  # results[m][config_name] = result_dict

    for m in m_values:
        results[m] = {}
        for cname in config_names:
            print(f"  {dataset_name} m={m:2d} {cname:30s}", end="  ", flush=True)
            try:
                res = run_single(g, m, CONFIGS[cname])
                print(f"crispness={res['crispness']:.4f}  time={res['time']:.2f}s")
                results[m][cname] = res
            except ValueError as e:
                print(f"FAILED: {e}")
                results[m][cname] = None

    return results


def print_table_a(dataset_name, results):
    """Print Table A: runtime and crispness comparison."""
    c1, c2 = list(CONFIGS.keys())
    print(f"\n## {dataset_name} — Table A: Runtime and Crispness\n")
    print(f"| m | {c1} time | {c1} crisp | {c2} time | {c2} crisp | Speedup |")
    print("|--:|----------:|-----------:|----------:|-----------:|--------:|")

    for m in sorted(results.keys()):
        r1 = results[m][c1]
        r2 = results[m][c2]
        if r1 is None or r2 is None:
            t1 = r1["time"] if r1 else "FAIL"
            c1v = f"{r1['crispness']:.4f}" if r1 else "FAIL"
            t2 = r2["time"] if r2 else "FAIL"
            c2v = f"{r2['crispness']:.4f}" if r2 else "FAIL"
            print(f"| {m} | {t1} | {c1v} | {t2} | {c2v} | — |")
        else:
            speedup = r1["time"] / r2["time"] if r2["time"] > 0 else float("inf")
            crisp_delta = r2["crispness"] - r1["crispness"]
            marker = "▲" if crisp_delta > 0.001 else ("▼" if crisp_delta < -0.001 else "=")
            print(
                f"| {m:2d} | {r1['time']:5.2f}s | {r1['crispness']:.4f} "
                f"| {r2['time']:5.2f}s | {r2['crispness']:.4f} {marker} "
                f"| {speedup:5.1f}× |"
            )


def print_table_b(dataset_name, results):
    """Print Table B: membership correlations."""
    c1, c2 = list(CONFIGS.keys())
    print(f"\n## {dataset_name} — Table B: Membership Correlations\n")

    all_rows = []
    for m in sorted(results.keys()):
        r1 = results[m][c1]
        r2 = results[m][c2]
        if r1 is None or r2 is None:
            continue
        correlations, unmatched_1, unmatched_2 = correlate_memberships(r1, r2)
        for name, r in correlations.items():
            all_rows.append((m, name, r))
        for name in unmatched_1:
            all_rows.append((m, f"{name} (only {c1})", None))
        for name in unmatched_2:
            all_rows.append((m, f"{name} (only {c2})", None))

    print("| m | Macrostate | Pearson r |")
    print("|--:|:-----------|----------:|")
    for m, name, r in all_rows:
        r_str = f"{r:.4f}" if r is not None else "—"
        print(f"| {m:2d} | {name} | {r_str} |")


def print_table_c(dataset_name, results):
    """Print Table C: Jaccard indices."""
    c1, c2 = list(CONFIGS.keys())
    print(f"\n## {dataset_name} — Table C: Jaccard Indices (discrete assignments)\n")

    all_rows = []
    for m in sorted(results.keys()):
        r1 = results[m][c1]
        r2 = results[m][c2]
        if r1 is None or r2 is None:
            continue
        jaccards, unmatched_1, unmatched_2 = compute_jaccard(r1, r2)
        for name, j in sorted(jaccards.items()):
            all_rows.append((m, name, j))
        for name in unmatched_1:
            all_rows.append((m, f"{name} (only {c1})", None))
        for name in unmatched_2:
            all_rows.append((m, f"{name} (only {c2})", None))

    print("| m | Macrostate | Jaccard |")
    print("|--:|:-----------|--------:|")
    for m, name, j in all_rows:
        j_str = f"{j:.3f}" if j is not None else "—"
        print(f"| {m:2d} | {name} | {j_str} |")


def main():
    datasets = [
        ("Pancreas (~2500 cells)", setup_pancreas),
        ("Bone Marrow (~5800 cells)", setup_bone_marrow),
    ]

    all_results = {}
    for dname, setup_fn in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dname}")
        print(f"{'='*70}")
        g = setup_fn()
        all_results[dname] = run_dataset(dname, g, M_VALUES)

    # Print all tables
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    for dname in all_results:
        print_table_a(dname, all_results[dname])
        print_table_b(dname, all_results[dname])
        print_table_c(dname, all_results[dname])

    print("\nDone!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
