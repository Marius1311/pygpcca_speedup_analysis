# pyGPCCA Speedup Analysis

Benchmarks for pyGPCCA optimization improvements: gradient-based optimization (L-BFGS-B), multi-start via SO(k) rotation perturbation, and Krylov scaling.

## Setup

All repos must live in the same parent directory:

```
Projects/
├── pygpcca_speedup_analysis/   # this repo
├── pyGPCCA/                    # feature/gradient-optimization branch
└── cellrank/                   # with optimizer passthrough
```

```bash
pixi install
pixi run test   # verify setup
```

The pixi environment provides PETSc/SLEPc for Krylov-based Schur decomposition (needed for m > ~20).

## Benchmarks

| Directory | What it tests | Datasets |
|-----------|--------------|----------|
| `analysis/01_synthetic/` | NM vs L-BFGS-B/BFGS/CG runtime & crispness scaling in m | Synthetic block-diagonal (n=5000, k=30) |
| `analysis/02_pancreas/` | NM vs L-BFGS-B on real data (Tables A/B/C) | CellRank pancreas (~2500 cells) |
| `analysis/03_dual_dataset/` | NM(1 start) vs L-BFGS-B(10 starts, eps=0.1) | Pancreas + bone marrow (~5800 cells) |
| `analysis/04_scaling/` | Feasibility at m=50 with Krylov | Pancreas + bone marrow |

Run any benchmark:

```bash
pixi run python analysis/01_synthetic/benchmark_gradient_opt.py
pixi run python analysis/02_pancreas/benchmark_pancreas.py [--mini]
pixi run python analysis/03_dual_dataset/benchmark_dual.py
pixi run python analysis/04_scaling/benchmark_m50.py
```

## Key findings so far

- **L-BFGS-B matches or beats NM** at every m tested on two real datasets
- Gap grows with m: ties at m≤5, **+50–135% crispness at m≥10**
- NM fails outright at bone marrow m=12 (degenerate solution); L-BFGS-B succeeds
- Multi-start (10 starts, eps=0.1) reliably finds better basins at negligible wall-time cost
- Dense Schur decomposition limits m to ~20; Krylov (SLEPc) needed beyond that
