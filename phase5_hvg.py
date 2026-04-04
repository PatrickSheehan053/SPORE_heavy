"""
SPORE · src/phase5_hvg.py
──────────────────────────
Phase 5: Dimensionality Reduction (Optional)
  - Seurat v3 HVG selection on training split (raw counts)
  - Perturbation target rescue override
  - Apply unified gene mask to all splits

Memory-safe: computes HVG dispersion on a lightweight sub-sample to 
prevent X**2 duplication spikes. Uses $O(1)$ C-buffer shifting for masking.
"""

import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from .utils import log_phase_header, snapshot, log_memory, force_gc

def safe_in_memory_gene_subset(adata, keep_mask, logger):
    """
    Destructively filters columns (genes) of a CSR matrix IN-PLACE. 
    Uses a row-by-row C-buffer shift to strictly enforce O(1) memory allocation,
    completely avoiding the massive RAM bloat of vectorized subsetting.
    """
    if keep_mask.all():
        return adata
        
    n_kept_genes = keep_mask.sum()
    
    if sp.issparse(adata.X) and adata.X.format == "csr":
        indptr = adata.X.indptr
        indices = adata.X.indices
        data = adata.X.data
        
        col_mapping = np.zeros(adata.X.shape[1], dtype=np.int32) - 1
        col_mapping[keep_mask] = np.arange(n_kept_genes, dtype=np.int32)
        
        new_indptr = np.zeros_like(indptr)
        write_ptr = 0
        
        for i in range(len(indptr) - 1):
            start = indptr[i]
            end = indptr[i+1]
            new_indptr[i] = write_ptr
            
            if start < end:
                r_ind = indices[start:end]
                r_dat = data[start:end]
                
                m_cols = col_mapping[r_ind]
                mask = m_cols >= 0
                nnz_row = mask.sum()
                
                if nnz_row > 0:
                    indices[write_ptr : write_ptr + nnz_row] = m_cols[mask]
                    data[write_ptr : write_ptr + nnz_row] = r_dat[mask]
                    
                write_ptr += nnz_row
                
        new_indptr[-1] = write_ptr
        
        new_X = sp.csr_matrix(
            (data[:write_ptr], indices[:write_ptr], new_indptr), 
            shape=(adata.n_obs, n_kept_genes)
        )
    else:
        logger.warning("  Matrix is not CSR. Falling back to default slicing.")
        new_X = adata.X[:, keep_mask]

    new_var = adata.var.iloc[keep_mask].copy()
    new_obs = adata.obs.copy()
    new_obsm = {k: v.copy() for k, v in adata.obsm.items()} if hasattr(adata, 'obsm') else {}
    new_varm = {k: v[keep_mask] for k, v in adata.varm.items()} if hasattr(adata, 'varm') else {}
    new_uns = adata.uns.copy() if hasattr(adata, 'uns') else {}
        
    del adata
    force_gc(logger)
    
    return ad.AnnData(
        X=new_X, obs=new_obs, var=new_var,
        obsm=new_obsm, varm=new_varm, uns=new_uns
    )


def _get_perturbation_targets(adata, cfg: dict) -> set:
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    targets = set(adata.obs[pert_col].unique()) - {ctrl_label}
    return targets & set(adata.var_names)


def select_hvg(train_adata, cfg: dict, logger):
    log_phase_header(logger, 5, "Dimensionality Reduction (HVG)")
    p5 = cfg["phase5_hvg"]

    if not p5["enabled"]:
        logger.info("  Phase 5 SKIPPED (disabled in config)")
        return list(train_adata.var_names), []

    n_top = p5["n_top_genes"]
    method = p5["method"]

    logger.info(f"  HVG method: {method} | Target: {n_top:,} genes")
    
    # ── Memory-Safe HVG Subsampling ──────────────────────────────────────────
    # Calculating LOESS variance across 1.5M cells triggers an X**2 duplication OOM.
    # We sample 100k cells to calculate the rankings, which is mathematically robust.
    MAX_HVG_CELLS = 100_000
    if train_adata.n_obs > MAX_HVG_CELLS:
        logger.info(f"  Downsampling to {MAX_HVG_CELLS:,} cells to calculate variance safely...")
        rng = np.random.default_rng(42)
        idx = rng.choice(train_adata.n_obs, MAX_HVG_CELLS, replace=False)
        calc_adata = train_adata[idx].copy()
    else:
        calc_adata = train_adata.copy()

    sc.pp.highly_variable_genes(calc_adata, n_top_genes=n_top, flavor=method, check_values=False)

    hvg_mask = calc_adata.var["highly_variable"]
    hvg_genes = set(calc_adata.var_names[hvg_mask])
    logger.info(f"  Seurat v3 selected: {len(hvg_genes):,} HVGs")

    # ADD THESE TWO LINES: Save the full variance stats before we delete the matrix
    # Safely grab whichever dispersion/variance columns Scanpy generated based on the flavor
    var_cols = [c for c in calc_adata.var.columns if c in ['means', 'dispersions_norm', 'variances_norm', 'highly_variable']]
    hvg_stats = calc_adata.var[var_cols].copy()
    train_adata.uns['hvg_stats'] = hvg_stats

    del calc_adata
    force_gc(logger)

    # ── Target Rescue ────────────────────────────────────────────────────────
    rescued = []
    if p5["rescue_perturbation_targets"]:
        targets = _get_perturbation_targets(train_adata, cfg)
        missing = targets - hvg_genes

        if missing:
            rescued = sorted(missing)
            hvg_genes |= missing
            logger.info(f"  ⚡ Target rescue: {len(rescued)} targets added to HVG set")
            if len(rescued) <= 20:
                logger.info(f"    Rescued: {', '.join(rescued)}")
            else:
                logger.info(f"    Rescued (first 20): {', '.join(rescued[:20])} ...")
        else:
            logger.info(f"  Target rescue: all targets already in HVG set")

    ordered_hvgs = [g for g in train_adata.var_names if g in hvg_genes]
    logger.info(f"  Final feature set: {len(ordered_hvgs):,} genes")

    return ordered_hvgs, rescued


def apply_hvg_mask(adata, hvg_genes: list, label: str, logger):
    hvg_set = set(hvg_genes)
    gene_mask = np.array([g in hvg_set for g in adata.var_names])
    n_valid = gene_mask.sum()
    logger.info(f"  [{label}] Applying HVG mask: {n_valid:,}/{adata.n_vars:,} genes")

    adata_new = safe_in_memory_gene_subset(adata, keep_mask=gene_mask, logger=logger)
    return adata_new


def run_phase5(splits: dict, cfg: dict, logger):
    hvg_genes, rescued = select_hvg(splits["train"], cfg, logger)

    log_memory(logger, "before HVG masking")
    for key in ["train", "val", "test"]:
        splits[key] = apply_hvg_mask(splits[key], hvg_genes, key.title(), logger)

    return hvg_genes, rescued
