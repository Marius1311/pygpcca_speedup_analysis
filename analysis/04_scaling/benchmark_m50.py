"""Quick scaling test: L-BFGS-B vs NM at m=50 on pancreas and bone marrow."""

import multiprocessing
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def run_m50(dataset_name, g):
    m = 50
    print(f"\n{'='*60}")
    print(f"  {dataset_name}: m={m}")
    print(f"{'='*60}")

    # L-BFGS-B, 10 starts
    print(f"\n  L-BFGS-B, 10 starts, eps=0.1 ...")
    t0 = time.perf_counter()
    g.compute_macrostates(
        n_states=m,
        cluster_key="clusters",
        optimizer="L-BFGS-B",
        n_starts=10,
        perturbation_scale=0.1,
        seed=0,
    )
    elapsed_lb = time.perf_counter() - t0
    crisp_lb = g._gpcca.crispness_values[-1]
    n_macro_lb = len(g.macrostates.cat.categories)
    print(f"  Time:       {elapsed_lb:.2f}s")
    print(f"  Crispness:  {crisp_lb:.4f}")
    print(f"  #Macrostates: {n_macro_lb}")

    # NM, 1 start
    print(f"\n  NM, 1 start ...")
    t0 = time.perf_counter()
    g.compute_macrostates(
        n_states=m,
        cluster_key="clusters",
        optimizer="Nelder-Mead",
        n_starts=1,
    )
    elapsed_nm = time.perf_counter() - t0
    crisp_nm = g._gpcca.crispness_values[-1]
    n_macro_nm = len(g.macrostates.cat.categories)
    print(f"  Time:       {elapsed_nm:.2f}s")
    print(f"  Crispness:  {crisp_nm:.4f}")
    print(f"  #Macrostates: {n_macro_nm}")

    print(f"\n  === {dataset_name} m={m} ===")
    print(f"  L-BFGS-B(10): {elapsed_lb:6.1f}s  crisp={crisp_lb:.4f}")
    print(f"  NM(1):        {elapsed_nm:6.1f}s  crisp={crisp_nm:.4f}")
    ratio = crisp_lb / crisp_nm if crisp_nm > 0 else float("inf")
    print(f"  Crispness ratio: {ratio:.2f}x")
    print(f"  Speedup: {elapsed_nm / elapsed_lb:.1f}x")


def setup_pancreas():
    import cellrank as cr
    import scanpy as sc

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
    g.compute_schur(n_components=51)
    print(f"  Pancreas: {adata.n_obs} cells, {adata.obs['clusters'].nunique()} clusters")
    return g


def setup_bone_marrow():
    import cellrank as cr
    import scanpy as sc

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
    g.compute_schur(n_components=51)
    print(f"  Bone marrow: {adata.n_obs} cells, {adata.obs['clusters'].nunique()} clusters")
    return g


if __name__ == "__main__":
    multiprocessing.freeze_support()

    g_panc = setup_pancreas()
    run_m50("Pancreas (~2500 cells)", g_panc)
    del g_panc

    g_bm = setup_bone_marrow()
    run_m50("Bone Marrow (~5800 cells)", g_bm)
    del g_bm

    print("\nDone!")
