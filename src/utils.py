"""
SPORE · src/utils.py
────────────────────
Config loading, logging, memory monitoring, and shared helpers.
"""

import yaml
import logging
import os
import gc
import psutil
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from datetime import datetime


def load_config(config_path: str = "spore_config.yaml") -> dict:
    """Load and validate the SPORE YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"])
    cfg["paths"]["_root"] = root
    cfg["paths"]["_raw_h5ad"] = root / cfg["paths"]["raw_h5ad"]
    cfg["paths"]["_processed"] = root / cfg["paths"]["processed_dir"]
    cfg["paths"]["_splits"] = root / cfg["paths"]["splits_dir"]
    cfg["paths"]["_figures"] = root / cfg["paths"]["figures_dir"]
    cfg["paths"]["_output_graphs"] = root / cfg["paths"]["output_graphs_dir"]
    cfg["paths"]["_logs"] = root / cfg["paths"]["log_dir"]

    for key in ["_processed", "_splits", "_figures", "_output_graphs", "_logs"]:
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)

    return cfg


def setup_logger(cfg: dict, name: str = "SPORE") -> logging.Logger:
    """Configure a logger that writes to both console and a timestamped log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(message)s",
                            datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = cfg["paths"]["_logs"] / f"spore_run_{timestamp}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s"))
    logger.addHandler(fh)

    logger.info(f"SPORE log initialized → {log_path}")
    return logger


def log_phase_header(logger: logging.Logger, phase, title: str):
    """Print a clean phase header to the log and console."""
    bar = "═" * 60
    logger.info(bar)
    logger.info(f"  PHASE {phase} · {title}")
    logger.info(bar)


def snapshot(adata, label: str, logger: logging.Logger):
    """Log shape snapshot with memory and sparsity info."""
    n_cells, n_genes = adata.shape
    mem = get_memory_usage()
    sparse_tag = ""
    if sp.issparse(adata.X):
        nnz = adata.X.nnz
        total = n_cells * n_genes
        sparsity = (1 - nnz / total) * 100 if total > 0 else 0
        sparse_tag = f"  (sparse, {sparsity:.1f}% zeros)"
    logger.info(f"  [{label}] → {n_cells:,} cells  ×  {n_genes:,} genes{sparse_tag}  |  RAM: {mem}")


# ═══════════════════════════════════════════════════════════════════════════
#  MEMORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def get_memory_usage() -> str:
    """Return current process RSS memory as a human-readable string."""
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    if mem_bytes >= 1e9:
        return f"{mem_bytes / 1e9:.1f} GB"
    return f"{mem_bytes / 1e6:.0f} MB"


def log_memory(logger: logging.Logger, label: str = ""):
    """Log the current memory usage."""
    mem = get_memory_usage()
    tag = f" ({label})" if label else ""
    logger.info(f"  💾 Memory{tag}: {mem}")


def force_gc(logger: logging.Logger = None):
    """Force garbage collection and optionally log it."""
    collected = gc.collect()
    if logger:
        mem = get_memory_usage()
        logger.info(f"  🗑️  GC collected {collected} objects  |  RAM: {mem}")


def ensure_sparse(adata, logger: logging.Logger = None):
    """
    Convert adata.X to CSR sparse if it isn't already.
    This is the single most important memory optimization for large datasets.

    A 2M × 10K dense float32 matrix = ~80 GB.
    The same at 95% sparsity in CSR      = ~4 GB.
    """
    if not sp.issparse(adata.X):
        if logger:
            n_cells, n_genes = adata.shape
            dense_gb = (n_cells * n_genes * 4) / 1e9
            logger.info(f"  ⚡ Converting dense → CSR sparse "
                        f"(dense would be ~{dense_gb:.1f} GB)")
        adata.X = sp.csr_matrix(adata.X)
        force_gc(logger)
        if logger:
            nnz = adata.X.nnz
            sparse_gb = (nnz * 12) / 1e9
            logger.info(f"  ⚡ Sparse: {nnz:,} non-zeros (~{sparse_gb:.1f} GB)")
    return adata


def safe_subset(adata, cell_mask=None, gene_mask=None, logger=None):
    """
    Memory-safe subsetting of AnnData.

    The standard pattern adata[mask].copy() creates a full view THEN copies,
    temporarily holding ~2x the data in RAM. For a 60 GB object on 128 GB,
    that's an instant OOM.

    This function:
      1. Bypass Check (Fast-Path): Instantly returns the original object if masks are 100% True, preventing 1:1 duplicate OOM crashes during "do-nothing" filters.
      2. Slices the sparse matrix directly via integer indexing.
      3. Subsets .obs / .var DataFrames.
      4. Constructs a new AnnData from components.
      5. Deletes the old adata if requested.
    """
    import anndata as ad

    if cell_mask is None:
        cell_mask = np.ones(adata.n_obs, dtype=bool)
    if gene_mask is None:
        gene_mask = np.ones(adata.n_vars, dtype=bool)

    # ── THE FIX: Fast-Path Bypass ──────────────────────────────────────────
    # If no cells or genes are being removed, skip slicing entirely
    if cell_mask.all() and gene_mask.all():
        if logger:
            logger.info("  No filtering required. Bypassing slice to save RAM.")
        return adata
    # ───────────────────────────────────────────────────────────────────────

    cell_idx = np.where(cell_mask)[0]
    gene_idx = np.where(gene_mask)[0]

    if logger:
        logger.info(f"  Subsetting: {cell_idx.shape[0]:,} cells × {gene_idx.shape[0]:,} genes")

    # Slice sparse matrix
    X_new = adata.X[cell_idx, :][:, gene_idx] # Added a comma slice for safer SciPy routing
    if sp.issparse(X_new):
        X_new = X_new.tocsr()

    obs_new = adata.obs.iloc[cell_idx].copy()
    var_new = adata.var.iloc[gene_idx].copy()

    # Carry over layers (subset them too)
    layers = {}
    if adata.layers:
        for lname, ldata in adata.layers.items():
            lsub = ldata[cell_idx, :][:, gene_idx]
            if sp.issparse(lsub):
                lsub = lsub.tocsr()
            layers[lname] = lsub

    # Carry over .uns
    uns = adata.uns.copy() if hasattr(adata, 'uns') and adata.uns else {}

    # Build new AnnData
    adata_new = ad.AnnData(X=X_new, obs=obs_new, var=var_new, layers=layers, uns=uns)

    return adata_new