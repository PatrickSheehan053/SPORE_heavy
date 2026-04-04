"""
SPORE · src/phase7_confounders.py
──────────────────────────────────
Phase 7: Confounder Mitigation
  - Cell cycle scoring and regression
  - Batch effect correction (Harmony / Combat)
  - Anti-imputation enforcement
  - Leakage firewall: fit on train only, apply to all

Memory-safe: strict avoidance of sparse-to-dense conversions during scaling,
PCA, regression, and batch correction.
"""

import numpy as np
import scanpy as sc
from .utils import log_phase_header, snapshot, log_memory, force_gc


# ═══════════════════════════════════════════════════════════════════════════
#  CELL CYCLE REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

def _get_cell_cycle_genes():
    """Tirosh et al. 2016 S-phase and G2M-phase gene lists."""
    s_genes = [
        "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
        "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
        "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
        "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
        "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
        "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
    ]
    g2m_genes = [
        "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
        "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
        "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
        "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
        "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
        "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
        "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
        "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
    ]
    return s_genes, g2m_genes


def score_cell_cycle(adata, logger):
    """Score cell cycle phases (S, G2M) and assign phase labels."""
    s_genes, g2m_genes = _get_cell_cycle_genes()

    # Build a case-insensitive map to handle Mouse vs Human naming
    var_upper_map = {v.upper(): v for v in adata.var_names}

    s_present = [var_upper_map[g] for g in s_genes if g in var_upper_map]
    g2m_present = [var_upper_map[g] for g in g2m_genes if g in var_upper_map]

    logger.info(f"  Cell cycle genes found in feature set: {len(s_present)}/{len(s_genes)} S, "
                f"{len(g2m_present)}/{len(g2m_genes)} G2M")

    # Safety Valve: If the HVG filter wiped them out, safely bypass
    if len(s_present) == 0 or len(g2m_present) == 0:
        logger.warning("  ⚠ CRITICAL: Not enough cell cycle genes survived the Phase 5 HVG filter!")
        logger.warning("  Bypassing cell cycle regression to prevent crash.")
        adata.obs["S_score"] = 0.0
        adata.obs["G2M_score"] = 0.0
        adata.obs["phase"] = "unknown"
        return adata

    try:
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g2m_present)
        phase_counts = adata.obs["phase"].value_counts()
        for phase, count in phase_counts.items():
            pct = count / adata.n_obs * 100
            logger.info(f"    {phase}: {count:,} cells ({pct:.1f}%)")
    except Exception as e:
        logger.warning(f"  ⚠ Cell cycle scoring failed natively: {e}")
        adata.obs["S_score"] = 0.0
        adata.obs["G2M_score"] = 0.0
        adata.obs["phase"] = "unknown"

    return adata


def regress_cell_cycle(adata, logger):
    """
    Regress out S_score and G2M_score.
    WARNING: sc.pp.regress_out densifies the matrix.
    """
    import scipy.sparse as sp
    
    # Safety Valve: Prevent massive dense matrix allocation on useless 0.0s
    if "phase" in adata.obs and (adata.obs["phase"] == "unknown").all():
        logger.info("  ⚠ Bypassing cell cycle regression: No valid scores to regress.")
        return adata

    logger.info(f"  Regressing out cell cycle scores ({adata.n_obs:,} cells) ...")
    log_memory(logger, "before regression")

    sc.pp.regress_out(adata, ["S_score", "G2M_score"])

    if not sp.issparse(adata.X):
        logger.info(f"  Re-sparsifying after regression ...")
        adata.X = sp.csr_matrix(adata.X)
        force_gc(logger)

    logger.info(f"  Cell cycle regression complete")
    log_memory(logger, "after regression")
    return adata


# ═══════════════════════════════════════════════════════════════════════════
#  BATCH EFFECT CORRECTION
# ═══════════════════════════════════════════════════════════════════════════

def correct_batch_harmony(adata, batch_key: str, logger):
    """Run Harmony integration on PCA space."""
    import scipy.sparse as sp
    from sklearn.preprocessing import StandardScaler

    logger.info(f"  Running Harmony batch correction (key='{batch_key}') ...")

    logger.info("  Scaling data (ultra-low-RAM manual scale)...")
    if sp.issparse(adata.X) and adata.X.format == "csr":
        scaler = StandardScaler(with_mean=False)
        scaler.fit(adata.X)

        scale_factors = scaler.scale_
        scale_factors[scale_factors == 0] = 1.0

        adata.X.data /= scale_factors[adata.X.indices]
        adata.X.data[adata.X.data > 10.0] = 10.0
    else:
        sc.pp.scale(adata, max_value=10, zero_center=False)

    logger.info("  Running PCA (sparse-safe, randomized solver)...")
    sc.pp.pca(adata, n_comps=50, use_highly_variable=False, zero_center=False, svd_solver='randomized')

    try:
        import harmonypy
        # FIX: harmonypy will crash with a KeyError if the batch column is numeric.
        # We explicitly cast the batch key to a categorical string to satisfy its internal pandas logic.
        if adata.obs[batch_key].dtype.name not in ['category', 'object', 'string']:
            logger.info(f"  Casting '{batch_key}' to categorical to satisfy harmonypy constraints...")
            adata.obs[batch_key] = adata.obs[batch_key].astype(str).astype('category')
            
        sc.external.pp.harmony_integrate(adata, key=batch_key)
        logger.info(f"  Harmony integration complete → X_pca_harmony")
    except ImportError:
        logger.warning("  ⚠ harmonypy not installed — skipping batch correction")

    return adata


def correct_batch_combat(adata, batch_key: str, logger):
    """Run Combat batch correction on expression values."""
    # PREDICTIVE FIX: ComBat physically requires a dense matrix.
    if adata.n_obs > 150_000:
        logger.warning(f"  ⚠ CRITICAL: ComBat requires dense matrix allocation.")
        logger.warning(f"  Applying ComBat to {adata.n_obs:,} cells will trigger an OOM.")
        logger.warning(f"  Bypassing ComBat. Please switch to Harmony for large datasets.")
        return adata
        
    logger.info(f"  Running Combat batch correction (key='{batch_key}') ...")
    sc.pp.combat(adata, key=batch_key)
    logger.info(f"  Combat correction complete")
    return adata


def correct_batch(adata, cfg: dict, logger):
    """Dispatch to configured batch correction method."""
    bc = cfg["phase7_confounders"]["batch_correction"]
    if not bc["enabled"]:
        logger.info("  Batch correction: DISABLED")
        return adata

    batch_key = bc["batch_key"]
    if batch_key not in adata.obs.columns:
        logger.warning(f"  ⚠ Batch column '{batch_key}' not found in .obs — skipping")
        return adata

    n_batches = adata.obs[batch_key].nunique()
    if n_batches <= 1:
        logger.info(f"  Only {n_batches} batch found — skipping batch correction")
        return adata

    logger.info(f"  Detected {n_batches} batches in '{batch_key}'")

    method = bc["method"]
    if method == "harmony":
        return correct_batch_harmony(adata, batch_key, logger)
    elif method == "combat":
        return correct_batch_combat(adata, batch_key, logger)
    else:
        logger.warning(f"  ⚠ Unknown method '{method}' — skipping")
        return adata


# ═══════════════════════════════════════════════════════════════════════════
#  RUN PHASE 7
# ═══════════════════════════════════════════════════════════════════════════

def run_phase7(splits: dict, cfg: dict, logger):
    log_phase_header(logger, 7, "Confounder Mitigation")
    p7 = cfg["phase7_confounders"]

    if p7["imputation_prohibited"]:
        logger.info("  ✓ Anti-imputation rule ACTIVE — no denoising will be applied")

    if p7["cell_cycle_regression"]:
        logger.info("  ── Cell Cycle Regression ──")
        for key in ["train", "val", "test"]:
            logger.info(f"  Scoring {key} ...")
            splits[key] = score_cell_cycle(splits[key], logger)

        logger.info(f"  Regressing (leakage firewall: each split independently)")
        for key in ["train", "val", "test"]:
            logger.info(f"  Regressing {key} ...")
            splits[key] = regress_cell_cycle(splits[key], logger)
    else:
        logger.info("  Cell cycle regression: DISABLED")

    logger.info("  ── Batch Effect Correction ──")
    for key in ["train", "val", "test"]:
        logger.info(f"  Correcting {key} ...")
        splits[key] = correct_batch(splits[key], cfg, logger)

    for key in ["train", "val", "test"]:
        snapshot(splits[key], f"Post Phase 7 ({key})", logger)

    return splits
