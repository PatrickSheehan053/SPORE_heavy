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


def _get_cell_cycle_genes():
    """Tirosh et al. 2016 S-phase and G2M-phase gene lists for ghosting."""
    s_genes = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
    g2m_genes = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]
    return s_genes + g2m_genes


def select_hvg(train_adata, cfg: dict, logger):
    log_phase_header(logger, 5, "Dimensionality Reduction (HVG)")
    p5 = cfg["phase5_hvg"]

    if not p5["enabled"]:
        logger.info("  Phase 5 SKIPPED (disabled in config)")
        return list(train_adata.var_names), list(train_adata.var_names), []

    n_top = p5["n_top_genes"]
    method = p5["method"]

    logger.info(f"  HVG method: {method} | Target: {n_top:,} genes")
    
    # ── Memory-Safe HVG Subsampling ──────────────────────────────────────────
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

    core_features = sorted([g for g in train_adata.var_names if g in hvg_genes])
    logger.info(f"  Core feature set: {len(core_features):,} genes")

    # ── Cell Cycle Ghosting ──────────────────────────────────────────────────
    cc_genes = _get_cell_cycle_genes()
    var_upper_map = {v.upper(): v for v in train_adata.var_names}
    cc_present = [var_upper_map[g] for g in cc_genes if g in var_upper_map]
    
    ghost_genes = sorted(list(set(cc_present) - set(core_features)))
    extended_features = core_features + ghost_genes
    logger.info(f"  Ghosting {len(ghost_genes)} cell cycle genes to preserve Phase 7 regression...")

    return extended_features, core_features, rescued


def apply_hvg_mask(adata, extended_genes: list, core_genes: list, label: str, logger):
    extended_set = set(extended_genes)
    gene_mask = np.array([g in extended_set for g in adata.var_names])
    n_valid = gene_mask.sum()
    logger.info(f"  [{label}] Applying HVG + Ghost mask: {n_valid:,}/{adata.n_vars:,} genes")

    adata_new = safe_in_memory_gene_subset(adata, keep_mask=gene_mask, logger=logger)
    
    # Save the true target list in the metadata so Phase 7 knows what to delete later
    adata_new.uns["spore_core_features"] = core_genes
    return adata_new


def run_phase5(splits: dict, cfg: dict, logger):
    extended_genes, core_genes, rescued = select_hvg(splits["train"], cfg, logger)

    log_memory(logger, "before HVG masking")
    for key in ["train", "val", "test"]:
        splits[key] = apply_hvg_mask(splits[key], extended_genes, core_genes, key.title(), logger)

    return core_genes, rescued
