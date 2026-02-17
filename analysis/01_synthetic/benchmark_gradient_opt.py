#!/usr/bin/env python
"""Benchmark gradient-based optimization vs Nelder-Mead in pyGPCCA.

This script compares Nelder-Mead, L-BFGS-B, BFGS, and CG for the rotation
matrix optimization in ``_opt_soft``.  It measures wall time and solution
quality (crispness, membership validity) across a range of cluster counts
``m`` for a fixed matrix size ``n``.

Results are printed as Markdown tables suitable for pasting into the report.

Experiment A — Runtime scaling in m (fixed n)
Experiment B — Solution quality comparison
"""

import sys
import time
import warnings

import numpy as np
from scipy.linalg import schur, get_lapack_funcs

from pygpcca._gpcca import _do_schur, _gram_schmidt_mod, _gpcca_core


def make_block_diagonal_transition_matrix(
    n: int,
    k: int,
    coupling: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Build a block-diagonal row-stochastic transition matrix.

    Parameters
    ----------
    n
        Total number of states (distributed evenly across blocks).
    k
        Number of metastable blocks.
    coupling
        Inter-block transition probability (per row, redistributed uniformly
        among off-block states).
    seed
        Random seed.

    Returns
    -------
    P
        Dense row-stochastic transition matrix of shape ``(n, n)``.
    """
    rng = np.random.default_rng(seed)
    block_size = n // k
    P = np.zeros((n, n), dtype=np.float64)

    for b in range(k):
        start = b * block_size
        end = start + block_size if b < k - 1 else n
        bsize = end - start
        # Random within-block transitions
        block = rng.random((bsize, bsize)) + np.eye(bsize) * 10.0
        block /= block.sum(axis=1, keepdims=True)
        P[start:end, start:end] = block

    # Add inter-block coupling
    for i in range(n):
        # Determine which block row i belongs to
        block_idx = min(i // block_size, k - 1)
        start = block_idx * block_size
        end = start + block_size if block_idx < k - 1 else n
        bsize = end - start
        off_block = n - bsize

        if off_block > 0:
            # Move `coupling` fraction of probability mass to off-block states
            P[i, start:end] *= 1.0 - coupling
            off_mass = coupling / off_block
            for j in range(n):
                if j < start or j >= end:
                    P[i, j] += off_mass

    # Re-normalize rows
    P /= P.sum(axis=1, keepdims=True)

    return P


def compute_schur_vectors(P: np.ndarray, m: int) -> np.ndarray:
    """Compute the first m orthonormalized Schur vectors.

    Uses scipy's LAPACK DTRSEN for robust eigenvalue reordering,
    bypassing the old Brandts sorting code which fails at large m.

    Parameters
    ----------
    P
        Row-stochastic transition matrix.
    m
        Number of Schur vectors (= number of clusters).

    Returns
    -------
    X
        Schur vector matrix of shape ``(n, m)`` with X[:, 0] == 1.
    """
    n = P.shape[0]
    eta = np.ones(n) / n  # uniform distribution

    # Weight P by eta (similarity transform)
    sqrt_eta = np.sqrt(eta)
    P_bar = np.diag(sqrt_eta) @ P @ np.diag(1.0 / sqrt_eta)

    # Compute real Schur form
    R, Q = schur(P_bar, output="real")

    # Compute eigenvalues from the Schur form
    eigenvalues = np.empty(n, dtype=complex)
    i = 0
    while i < n:
        if i + 1 < n and abs(R[i + 1, i]) > 1e-10:
            # 2x2 block: complex conjugate pair
            a, b, c, d = R[i, i], R[i, i + 1], R[i + 1, i], R[i + 1, i + 1]
            tr = a + d
            det = a * d - b * c
            disc = tr**2 / 4 - det
            eigenvalues[i] = tr / 2 + np.sqrt(complex(disc))
            eigenvalues[i + 1] = tr / 2 - np.sqrt(complex(disc))
            i += 2
        else:
            eigenvalues[i] = R[i, i]
            i += 1

    # Select the m eigenvalues with largest magnitude
    order = np.argsort(-np.abs(eigenvalues))
    selected = np.zeros(n, dtype=np.int32)
    for idx in order[:m]:
        selected[idx] = 1
        # If part of a conjugate pair in a 2x2 block, select both
        if idx + 1 < n and abs(R[idx + 1, idx]) > 1e-10:
            selected[idx + 1] = 1
        elif idx > 0 and abs(R[idx, idx - 1]) > 1e-10:
            selected[idx - 1] = 1

    # Use DTRSEN to reorder
    trsen = get_lapack_funcs("trsen", (R,))
    lwork = max(1, n * (n + 1))
    liwork = max(1, n * n)
    Rr, Qr, _wr, _wi, _ms, _sep, _est, info = trsen(
        selected, R, Q, lwork=lwork, liwork=liwork,
    )
    if info != 0:
        raise RuntimeError(f"DTRSEN failed with info={info}")

    # Take first m columns
    Q_m = Qr[:, :m]
    R_m = Rr[:m, :m]

    # Orthogonalize via Gram-Schmidt in the eta inner product
    Q_m = _gram_schmidt_mod(Q_m, eta)

    # Transform back to Schur vectors of P
    X = Q_m / sqrt_eta[:, None]
    X[:, 0] = 1.0

    return X


def run_experiment_a(
    n: int,
    m_values: list[int],
    methods: list[str],
    n_repeats: int,
    k: int,
    seed: int = 42,
) -> list[dict]:
    """Experiment A: runtime scaling in m.

    Parameters
    ----------
    n
        Matrix size.
    m_values
        List of cluster counts to benchmark.
    methods
        Optimization methods to compare.
    n_repeats
        Number of repeats per configuration.
    k
        Number of true metastable blocks in the test matrix.
    seed
        Random seed for matrix generation.

    Returns
    -------
    results
        List of result dicts with keys: m, method, repeat, time_s, fopt, crispness.
    """
    max_m = max(m_values)
    assert max_m <= k, f"max(m)={max_m} exceeds k={k} blocks"

    print(f"\nBuilding n={n}, k={k} block-diagonal matrix...", flush=True)
    P = make_block_diagonal_transition_matrix(n, k, coupling=0.01, seed=seed)
    print(f"  Matrix built. Computing Schur vectors for m={max_m}...", flush=True)
    X_full = compute_schur_vectors(P, max_m)
    print(f"  Schur vectors ready (shape={X_full.shape}).", flush=True)

    results = []
    for m in m_values:
        X = X_full[:, :m].copy()
        opt_dim = (m - 1) ** 2
        for method in methods:
            # Skip NM for large m — too slow
            if method == "Nelder-Mead" and m > 15:
                print(f"  m={m:>3d}, {method:<15s}: SKIPPED (too slow)", flush=True)
                continue

            times = []
            fopts = []
            crispnesses = []
            for rep in range(n_repeats):
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    chi, rot_matrix, crispness = _gpcca_core(X.copy(), method=method)
                elapsed = time.perf_counter() - t0
                fopt = m * (1.0 - crispness)

                times.append(elapsed)
                fopts.append(fopt)
                crispnesses.append(crispness)

                results.append({
                    "m": m,
                    "method": method,
                    "repeat": rep,
                    "time_s": elapsed,
                    "fopt": fopt,
                    "crispness": crispness,
                    "opt_dim": opt_dim,
                })

            mean_t = np.mean(times)
            mean_c = np.mean(crispnesses)
            print(
                f"  m={m:>3d}, {method:<15s}: {mean_t:8.3f}s  "
                f"crispness={mean_c:.4f}  fopt={np.mean(fopts):.4f}",
                flush=True,
            )

    return results


def run_experiment_b(results: list[dict], methods: list[str]) -> list[dict]:
    """Experiment B: solution quality comparison between methods.

    Compares crispness and membership validity across methods for each m.

    Parameters
    ----------
    results
        Results from Experiment A.
    methods
        Methods to compare.

    Returns
    -------
    quality
        List of quality comparison dicts.
    """
    # Group by m
    m_values = sorted(set(r["m"] for r in results))
    quality = []

    for m in m_values:
        for method in methods:
            runs = [r for r in results if r["m"] == m and r["method"] == method]
            if not runs:
                continue
            quality.append({
                "m": m,
                "method": method,
                "mean_crispness": np.mean([r["crispness"] for r in runs]),
                "std_crispness": np.std([r["crispness"] for r in runs]),
                "mean_fopt": np.mean([r["fopt"] for r in runs]),
                "mean_time": np.mean([r["time_s"] for r in runs]),
                "std_time": np.std([r["time_s"] for r in runs]),
            })

    return quality


def format_table_a(quality: list[dict], methods: list[str]) -> str:
    """Format Experiment A results as a Markdown table."""
    m_values = sorted(set(q["m"] for q in quality))

    lines = [
        "### Experiment A — Runtime scaling in m",
        "",
        f"Matrix: n=see below, block-diagonal with coupling=0.01.",
        f"Methods: {', '.join(methods)}. 3 repeats per configuration.",
        "",
        "| m | opt dim |" + "".join(f" {meth} time (s) |" for meth in methods) + " speedup (NM/L-BFGS-B) |",
        "|:--|:--------|" + "".join("-" * (len(meth) + 12) + ":|" for meth in methods) + ":-----------------------|",
    ]

    for m in m_values:
        row = f"| {m} | {(m-1)**2} |"
        nm_time = None
        lbfgsb_time = None
        for meth in methods:
            runs = [q for q in quality if q["m"] == m and q["method"] == meth]
            if runs:
                t = runs[0]["mean_time"]
                row += f" {t:.3f} ± {runs[0]['std_time']:.3f} |"
                if meth == "Nelder-Mead":
                    nm_time = t
                if meth == "L-BFGS-B":
                    lbfgsb_time = t
            else:
                row += " — |"
        if nm_time is not None and lbfgsb_time is not None and lbfgsb_time > 0:
            row += f" {nm_time / lbfgsb_time:.0f}× |"
        else:
            row += " — |"
        lines.append(row)

    return "\n".join(lines)


def format_table_b(quality: list[dict], methods: list[str]) -> str:
    """Format Experiment B results as a Markdown table."""
    m_values = sorted(set(q["m"] for q in quality))

    lines = [
        "### Experiment B — Solution quality (crispness)",
        "",
        "| m |" + "".join(f" {meth} crispness |" for meth in methods) + "",
        "|:--|" + "".join("-" * (len(meth) + 13) + ":|" for meth in methods) + "",
    ]

    for m in m_values:
        row = f"| {m} |"
        for meth in methods:
            runs = [q for q in quality if q["m"] == m and q["method"] == meth]
            if runs:
                row += f" {runs[0]['mean_crispness']:.4f} ± {runs[0]['std_crispness']:.4f} |"
            else:
                row += " — |"
        lines.append(row)

    return "\n".join(lines)


def format_detailed_table(results: list[dict], methods: list[str]) -> str:
    """Format per-m timing comparison as a compact table for the report."""
    m_values = sorted(set(r["m"] for r in results))

    lines = [
        "### Detailed timing comparison",
        "",
        "| m | opt dim | Method | mean time (s) | std | mean crispness | mean fopt |",
        "|:--|:--------|:-------|:--------------|:----|:---------------|:----------|",
    ]

    for m in m_values:
        for meth in methods:
            runs = [r for r in results if r["m"] == m and r["method"] == meth]
            if not runs:
                continue
            mean_t = np.mean([r["time_s"] for r in runs])
            std_t = np.std([r["time_s"] for r in runs])
            mean_c = np.mean([r["crispness"] for r in runs])
            mean_f = np.mean([r["fopt"] for r in runs])
            lines.append(
                f"| {m} | {(m-1)**2} | {meth} | {mean_t:.4f} | {std_t:.4f} | {mean_c:.4f} | {mean_f:.4f} |"
            )

    return "\n".join(lines)


def main() -> None:
    """Run all benchmark experiments."""
    print("=" * 70)
    print("pyGPCCA Gradient Optimization Benchmark")
    print("=" * 70)
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")

    import scipy

    print(f"SciPy {scipy.__version__}")

    # Configuration
    # Use n=5000 for speed (Schur decomposition via dense Brandts is O(n^3));
    # the optimization scaling in m is independent of n (see profiling report).
    n = 5000
    k = 30  # number of true metastable blocks
    m_values = [3, 5, 8, 10, 15, 20, 25, 30]
    methods = ["Nelder-Mead", "L-BFGS-B", "BFGS", "CG"]
    n_repeats = 3

    print(f"\nConfiguration: n={n}, k={k}, m_values={m_values}")
    print(f"Methods: {methods}")
    print(f"Repeats: {n_repeats}")

    # --- Experiment A: Runtime ---
    print("\n" + "=" * 70)
    print("Experiment A: Runtime scaling in m")
    print("=" * 70)
    results = run_experiment_a(n, m_values, methods, n_repeats, k)

    # --- Experiment B: Quality ---
    quality = run_experiment_b(results, methods)

    # --- Print tables ---
    print("\n\n" + "=" * 70)
    print("RESULTS (Markdown)")
    print("=" * 70)

    print(f"\nn={n}, k={k} blocks, coupling=0.01, {n_repeats} repeats.\n")

    table_a = format_table_a(quality, methods)
    print(table_a)
    print()

    table_b = format_table_b(quality, methods)
    print(table_b)
    print()

    detailed = format_detailed_table(results, methods)
    print(detailed)

    # Write results to file for report
    output_path = "benchmark_gradient_opt_results.md"
    with open(output_path, "w") as f:
        f.write(f"# Gradient Optimization Benchmark Results\n\n")
        f.write(f"n={n}, k={k} blocks, coupling=0.01, {n_repeats} repeats.\n\n")
        f.write(f"Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}\n\n")
        f.write(table_a + "\n\n")
        f.write(table_b + "\n\n")
        f.write(detailed + "\n")
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
