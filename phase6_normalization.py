"""
SPORE · src/phase6_normalization.py
────────────────────────────────────
Phase 6: Normalization & Scaling
  - Global library size scaling (to median or fixed target)
  - log1p transformation

Sparse-safe: Manual float32 C-buffer casting prevents Scanpy's silent
dense upcasting and matrix duplication OOM traps.
"""

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from .utils import log_phase_header, snapshot, log_memory


def _sample_expression_values(adata, n_sample: int = 100_000):
    """Sample non-zero expression values for plotting histograms."""
    X = adata.X
    if sp.issparse(X):
        data = X.data.copy()  # .data gives non-zero values directly
    else:
        data = X.flatten()
        data = data[data > 0]
    if len(data) > n_sample:
        rng = np.random.default_rng(42)
        data = rng.choice(data, size=n_sample, replace=False)
    return data


def normalize_split(adata, cfg: dict, logger, label: str = ""):
    """
    Normalize a single AnnData split in-place.
    """
    p6 = cfg["phase6_normalization"]

    # Extract sample for plotting BEFORE mutating the matrix
    raw_sample = _sample_expression_values(adata)

    # 1. THE "SPARSE COPY IS CHEAP" TRAP
    # Removed: adata.layers["raw_counts"] = adata.X.copy()
    # Duplicating a 1.5M cell CSR matrix instantly demands ~15 GB.
    # If downstream models need raw counts, load them dynamically from the Phase 4/5 .h5ad files.

    # 2. THE IMPLICIT UPCAST TRAP
    # sc.pp.normalize_total requires floats. If handed ints, it silently allocates a full duplicate matrix.
    # We manually overwrite the 1D data buffer to float32 to force true in-place calculation.
    if sp.issparse(adata.X) and adata.X.dtype != np.float32:
        logger.info(f"  [{label}] Upcasting C-buffer to float32 to prevent Scanpy duplication...")
        adata.X.data = adata.X.data.astype(np.float32, copy=False)

    target = p6["target_sum"]
    target_str = "median" if target is None else f"{target:,}"

    sc.pp.normalize_total(adata, target_sum=target)
    logger.info(f"  [{label}] Library size normalized (target: {target_str})")

    if p6["log_transform"]:
        sc.pp.log1p(adata)
        logger.info(f"  [{label}] log1p applied")

    norm_sample = _sample_expression_values(adata)
    snapshot(adata, f"{label} normalized", logger)

    return adata, raw_sample, norm_sample


def run_phase6(splits: dict, cfg: dict, logger):
    """Normalize all splits. Returns raw/norm samples from train for plotting."""
    log_phase_header(logger, 6, "Normalization & Scaling")
    log_memory(logger, "start of Phase 6")

    raw_sample, norm_sample = None, None
    for key in ["train", "val", "test"]:
        splits[key], raw_s, norm_s = normalize_split(
            splits[key], cfg, logger, label=key.title())
        if key == "train":
            raw_sample, norm_sample = raw_s, norm_s

    return raw_sample, norm_sample
