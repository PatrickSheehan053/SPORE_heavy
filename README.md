# 🍄 SPORE

### Systematic Preprocessing and Optimization for Robust Evaluation

SPORE is a memory-safe preprocessing pipeline for massive-scale single-cell RNA sequencing perturbation datasets. It takes a raw `.h5ad` file containing genome-wide CRISPR screen data (e.g., Replogle et al. 2022, ~2M cells, 62 GB) and produces clean, normalized, leakage-free train/validation/test splits ready for downstream Gene Regulatory Network (GRN) inference and Graph Neural Network evaluation.

SPORE was built to solve a practical problem: standard single-cell preprocessing tools (Scanpy, AnnData) assume datasets fit comfortably in RAM. When they don't (when a single boolean subset triggers a 100 GB memory spike that kills your HPC session) you need infrastructure that works at the level of C-buffers and sparse matrix internals, not convenience wrappers. SPORE is that infrastructure.

---

## What It Does

SPORE processes perturbation scRNA-seq data through eight sequential phases, each designed to remove a specific class of noise while preserving the causal signals that downstream GRNs depend on.

**Phase 0 — Data Ingestion & Sparsification.** Raw dense `.h5ad` matrices are converted to CSR sparse format in memory-bounded chunks (50k cells at a time) using backed-mode reading. The sparse result is cached to disk so subsequent runs skip the conversion entirely. This reduces a 62 GB dense matrix to ~5 GB in sparse form.

**Phase 1 — Cell-Level Triage.** Dead cells (high mitochondrial RNA fraction), empty droplets (too few detected genes), and doublets (too many detected genes) are flagged and removed. Ribosomal protein genes (RPL/RPS) are stripped from the feature space to prevent them from drowning out perturbation-specific regulatory signals. All filtering is done via boolean mask accumulation with a single in-place C-buffer mutation at the end, no intermediate copies.

**Phase 2 — Escaper Filtering.** CRISPRi efficiency is variable. Cells that received a guide RNA but failed to knock down the target gene carry wild-type transcriptomes that would corrupt causal inference. SPORE compares each perturbed cell's target gene expression against the 10th percentile of the unperturbed control distribution and removes cells above that threshold. Perturbation groups that fall below the minimum population size (default: 50 cells) are dropped entirely to prevent downstream batch instability.

**Phase 3 — Gene-Level Triage.** Noisy, rarely-expressed genes are removed using a scale-adaptive strategy: for datasets above 50k cells, genes with mean expression below 0.25 UMI are flagged. A perturbation target rescue override ensures that any gene experimentally targeted by a CRISPR guide is force-retained in the feature set regardless of its expression level, losing a perturbation's root node would silently destroy the causal graph downstream.

**Phase 4 — Data Splits.** The dataset is partitioned into train, validation, and test sets with strict leakage prevention. In zero-shot mode (default), perturbation labels are stratified by mean expression shift into quartile bins, then split so that each partition contains a representative mix of mild, moderate, and severe knockdowns. In cell-wise mode, every perturbation appears in all splits but with different cells. A test mode generates five replicate splits across different random seeds for cross-validated benchmarking. The split uses a destructive 3-way strategy: small test and validation matrices are extracted first, then the original object is mutated in-place to become the training set, avoiding the 200% RAM spike that would occur if all three copies existed simultaneously.

**Phase 5 — Dimensionality Reduction.** The top 5,000 Highly Variable Genes are selected using the Seurat v3 algorithm (variance-stabilizing LOESS regression on raw UMI counts), computed exclusively on a 100k-cell subsample of the training split to prevent both data leakage and memory exhaustion. Perturbation targets that fail the variance cutoff are rescued into the feature set. Cell cycle genes required for Phase 7 regression are preserved as temporary "ghost" features that will be pruned after confounder mitigation. The identical gene mask is applied to all splits.

**Phase 6 — Normalization & Scaling.** Library size normalization scales each cell's expression vector to the median total count across the dataset. Counts are then log-transformed (`log1p`) to stabilize variance and compress the dynamic range. The integer-to-float conversion is performed in-place on the sparse matrix's C-buffer to avoid allocating a duplicate matrix. Raw counts are not duplicated into `adata.layers`, they are already saved to disk from previous phases.

**Phase 7 — Confounder Mitigation.** Cell cycle effects are scored using the Tirosh 2016 reference gene lists and regressed out of the expression matrix. If cell cycle genes were eliminated by upstream filters, the regression step is gracefully bypassed rather than triggering a catastrophic dense matrix allocation on dummy values. Batch effects across sequencing runs are corrected via Harmony integration, operating in PCA space with a randomized SVD solver and sparse-safe scaling (`zero_center=False`). An anti-imputation rule is enforced: denoising tools like MAGIC or scVI are strictly prohibited because they manufacture artificial correlations that would cause downstream GRN algorithms to hallucinate spurious causal edges.

**Phase 8 — Meta-Cell Aggregation (Optional).** Transcriptionally similar cells within each perturbation group are clustered via MiniBatchKMeans (on Harmony PCA embeddings) and aggregated into dense meta-cells. This compresses 1.5M single cells into ~150k meta-cells, combating dropout sparsity while preserving perturbation-level structure. Systema geometric centroids (control centroid and per-perturbation offsets) are computed on the training split for downstream evaluation calibration.

---

## Architecture

SPORE is structured as a Jupyter notebook (`spore.ipynb`) backed by modular Python source files in `src/`. The notebook orchestrates the pipeline and generates diagnostic visualizations at each phase; the source modules handle the computation.

```
SPORE/
├── spore.ipynb                 # Pipeline orchestrator
├── spore_config.yaml           # All thresholds, paths, and runtime settings
├── raw_h5ads/                  # Input .h5ad files
├── src/
│   ├── phase0_sparse_convert.py
│   ├── phase1_cell_triage.py
│   ├── phase2_escaper_filtering.py
│   ├── phase3_gene_triage.py
│   ├── phase4_splits.py
│   ├── phase5_hvg.py
│   ├── phase6_normalization.py
│   ├── phase7_confounders.py
│   ├── phase8_metacell.py
│   ├── plotting.py
│   └── utils.py
├── processed/                  # Final GRN-ready .h5ad outputs
├── splits/                     # Intermediate split files + indices
├── figures/                    # Saved diagnostic plots
└── logs/                       # Timestamped execution logs
```

All configuration (file paths, filter thresholds, split ratios, parallelization settings) lives in `spore_config.yaml`. Changing datasets requires updating the YAML; no source code modifications are needed.

---

## Memory Safety

SPORE was engineered for datasets that exceed available RAM. The Replogle 2022 K562 dataset (1,989,578 cells × 8,248 genes, 62 GB dense) runs end-to-end on a server with 80 GB of available memory, peaking at 88.5 GB during the split phase. Key techniques:

- **Chunked sparsification** converts dense matrices in 50k-cell blocks rather than loading the full matrix into memory.
- **In-place C-buffer mutation** filters rows and columns by directly shifting the underlying `data`, `indices`, and `indptr` arrays of CSR matrices, avoiding SciPy's subsetting engine which allocates a full duplicate.
- **Destructive splitting** extracts small partitions first, then mutates the original object into the largest partition, ensuring the dataset never exists in duplicate.
- **Plot downsampling** caps scatter and violin plots at 200k points, computing summary statistics on the full dataset but rendering only a subsample.
- **Aggressive garbage collection** with `del` and `gc.collect()` at every phase boundary, with continuous RAM monitoring via `psutil`.

---

## Configuration

The default `spore_config.yaml` is tuned for the Replogle 2022 K562 GWPS CRISPRi dataset. Key parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `mt_threshold` | 0.20 | Max mitochondrial RNA fraction |
| `min_cells_per_perturbation` | 50 | Drop perturbations below this cell count |
| `mean_umi_threshold` | 0.25 | Min mean UMI for gene retention (large datasets) |
| `train_test_ratio` | 0.90 | 90/10 split for 2000+ perturbation datasets |
| `n_top_genes` | 5000 | HVG target count |
| `stratify_by` | mean_shift | Fast stratification proxy for large datasets |
| `batch_correction.method` | harmony | Batch correction algorithm |
| `n_jobs` | 8 | Parallel workers for Phase 2 and Phase 8 |

---

## Requirements

- Python 3.9+
- scanpy, anndata, scipy, numpy (<2.0), pandas
- scikit-misc (for Seurat v3 LOESS regression)
- harmonypy (for batch correction)
- scikit-learn (for meta-cell clustering)
- joblib, psutil, pyyaml, matplotlib, seaborn

---

## Part of MYCELIUM

SPORE is the preprocessing layer of the [MYCELIUM](https://github.com/patricksheehan/MYCELIUM) pipeline for building and evaluating Gene Regulatory Networks from single-cell perturbation data. Its outputs feed directly into GuanLab's PSGRN for GRN construction, followed by FUNGI for graph pruning and SPECTRA for GNN-based perturbation response prediction.
