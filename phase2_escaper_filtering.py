"""
SPORE · src/phase2_escaper_filtering.py
────────────────────────────────────────
Phase 2: Escaper Filtering
  - Cell-level escaper removal (CRISPRi efficacy validation)
  - Perturbation-level triage (minimum population size)

Memory-safe: uses vectorized in-place memory mutation to literally
rewrite the SciPy CSR arrays without allocating duplicate matrices.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import gc
from .utils import log_phase_header, snapshot, log_memory, force_gc


def safe_in_memory_subset(adata, keep_mask, logger):
    """
    Destructively modifies the underlying scipy.sparse CSR matrix IN-PLACE. 
    Rebuilds the AnnData wrapper from scratch to satisfy Scanpy's strict
    dimension validation without allocating duplicate matrices.
    """
    if keep_mask.all():
        logger.info("  No filtering required. Bypassing slice to save RAM.")
        return adata

    logger.info("  Applying ultra-low-RAM in-place matrix mutation...")
    
    n_kept = keep_mask.sum()
    
    # 1. Mutate the main matrix in-place
    if sp.issparse(adata.X) and adata.X.format == "csr":
        indptr = adata.X.indptr
        indices = adata.X.indices
        data = adata.X.data
        
        new_indptr = np.zeros(n_kept + 1, dtype=indptr.dtype)
        
        padded = np.concatenate(([False], keep_mask, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        write_ptr = 0
        new_row = 0
        for start, end in zip(starts, ends):
            n_rows_block = end - start
            data_start = indptr[start]
            data_end = indptr[end]
            nnz_block = data_end - data_start
            
            if nnz_block > 0:
                indices[write_ptr : write_ptr + nnz_block] = indices[data_start:data_end]
                data[write_ptr : write_ptr + nnz_block] = data[data_start:data_end]
                
            new_indptr[new_row + 1 : new_row + 1 + n_rows_block] = indptr[start + 1 : end + 1] - data_start + write_ptr
            write_ptr += nnz_block
            new_row += n_rows_block
            
        # Create a fresh CSR matrix directly from the truncated buffers
        new_X = sp.csr_matrix(
            (data[:write_ptr], indices[:write_ptr], new_indptr), 
            shape=(n_kept, adata.X.shape[1])
        )
    else:
        logger.warning("  Matrix is not CSR. Falling back to default slicing.")
        new_X = adata.X[keep_mask]
    
    # 2. Subset metadata cleanly
    new_obs = adata.obs.iloc[keep_mask].copy()
    new_var = adata.var.copy()
    
    # 3. Clean up heavy secondary structures safely
    new_obsm = {k: v[keep_mask] for k, v in adata.obsm.items()} if hasattr(adata, 'obsm') else {}
    new_varm = {k: v.copy() for k, v in adata.varm.items()} if hasattr(adata, 'varm') else {}
    new_uns = adata.uns.copy() if hasattr(adata, 'uns') else {}
        
    # 4. Obliterate the old wrapper and free memory
    del adata
    force_gc(logger)
    
    # 5. Rebuild pure AnnData wrapper to satisfy dimension validation
    adata_new = ad.AnnData(
        X=new_X, obs=new_obs, var=new_var,
        obsm=new_obsm, varm=new_varm, uns=new_uns
    )
    return adata_new


def filter_escapers(adata, cfg: dict, logger):
    """
    For each perturbation, compare target gene expression in perturbed cells
    against the control distribution. 
    """
    log_phase_header(logger, 2, "Escaper Filtering")
    p2 = cfg["phase2_escaper_filtering"]
    ds = cfg["dataset"]

    pert_col = ds["perturbation_col"]
    ctrl_label = ds["control_label"]
    percentile = p2["escaper_percentile"]

    pert_values = adata.obs[pert_col].values
    ctrl_mask = pert_values == ctrl_label
    ctrl_indices = np.where(ctrl_mask)[0]
    n_ctrl = len(ctrl_indices)
    logger.info(f"  Control cells ('{ctrl_label}'): {n_ctrl:,}")

    perturbations = [p for p in np.unique(pert_values) if p != ctrl_label]
    logger.info(f"  Unique perturbation targets: {len(perturbations):,}")

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    # Pre-cache controls in CSC format (tiny memory footprint, extreme speed)
    logger.info("  Pre-caching control matrix...")
    ctrl_X_csc = adata.X[ctrl_indices, :].tocsc()

    cells_to_keep = list(ctrl_indices)
    escaper_records = []
    n_skipped = 0
    n_filtered = 0

    log_memory(logger, "Before surgical escaper classification")
    
    for target in perturbations:
        pert_indices = np.where(pert_values == target)[0]
        n_total = len(pert_indices)

        if target not in gene_to_idx:
            escaper_records.append({
                "perturbation": target, "n_total": n_total, 
                "n_escaped": 0, "n_kept": n_total, "status": "bypassed"
            })
            cells_to_keep.extend(pert_indices)
            n_skipped += 1
            continue

        gene_idx = gene_to_idx[target]
        
        # Fast extract control column
        ctrl_expr = ctrl_X_csc[:, gene_idx].toarray().flatten()
        
        # Fast extract perturbed cells (Row slice first, then Col slice)
        pert_expr = adata.X[pert_indices, :][:, gene_idx].toarray().flatten()

        threshold = np.percentile(ctrl_expr, percentile)
        is_knockdown = pert_expr <= threshold
        
        kept = pert_indices[is_knockdown]
        n_escaped = n_total - len(kept)

        cells_to_keep.extend(kept)
        escaper_records.append({
            "perturbation": target, "n_total": n_total, 
            "n_escaped": n_escaped, "n_kept": len(kept), "status": "filtered"
        })
        n_filtered += 1
        
        del ctrl_expr, pert_expr, is_knockdown

    del ctrl_X_csc
    gc.collect()

    escaper_stats = pd.DataFrame(escaper_records)
    total_escaped = escaper_stats["n_escaped"].sum()
    logger.info(f"  Escaper filter applied to {n_filtered:,} perturbations "
                f"({n_skipped:,} bypassed — target not in features)")
    logger.info(f"  Total escapers removed: {total_escaped:,}")

    keep_mask = np.zeros(adata.n_obs, dtype=bool)
    keep_mask[cells_to_keep] = True

    adata_new = safe_in_memory_subset(adata, keep_mask, logger)
    snapshot(adata_new, "Post escaper filter", logger)
    
    return adata_new, escaper_stats


def filter_undersized_perturbations(adata, cfg: dict, logger):
    """Remove perturbation groups smaller than the minimum threshold."""
    p2 = cfg["phase2_escaper_filtering"]
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    min_cells = p2["min_cells_per_perturbation"]

    sizes = adata.obs[pert_col].value_counts()
    undersized = sizes[(sizes < min_cells) & (sizes.index != ctrl_label)]
    
    n_drop_perts = len(undersized)
    n_drop_cells = undersized.sum()

    if n_drop_perts > 0:
        drop_labels = set(undersized.index)
        keep_mask = ~adata.obs[pert_col].isin(drop_labels).values
        logger.info(f"  Perturbation triage (min {min_cells} cells): "
                    f"dropped {n_drop_perts:,} perturbations ({n_drop_cells:,} cells)")

        adata = safe_in_memory_subset(adata, keep_mask, logger)
    else:
        logger.info(f"  Perturbation triage: all meet minimum ({min_cells} cells)")

    final_sizes = adata.obs[pert_col].value_counts()
    snapshot(adata, "Post perturbation triage", logger)
    return adata, final_sizes


def run_phase2(adata, cfg: dict, logger):
    """Full Phase 2: escaper filtering → perturbation triage."""
    adata, escaper_stats = filter_escapers(adata, cfg, logger)
    adata, pert_sizes = filter_undersized_perturbations(adata, cfg, logger)
    return adata, escaper_stats, pert_sizes
