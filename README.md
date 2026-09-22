# RegAtlas: A Learning-to-Rank Multi-Omics Framework for Post-GWAS Locus-to-Gene Mapping in Psychiatric Disorders

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LightGBM LambdaRank](https://img.shields.io/badge/Model-LightGBM%20LambdaRank-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.xxxxxx-blue.svg)](https://zenodo.org/)

**RegAtlas** is a machine learning framework that reformulates post-GWAS **Locus-to-Gene (L2G)** mapping as a **Learning-to-Rank (LTR)** query problem. By integrating **spatial genomic distance**, **GTEx v10 brain cortex cis-eQTLs**, and **ENCODE-rE2G predicted enhancer-to-gene regulatory links (Gschwind et al. 2023)**, RegAtlas learns intra-locus relative separation among competing candidate genes, effectively overcoming proximity bias and linkage disequilibrium (LD) confounding.

---

## Key Innovations

| Feature | Prior Post-GWAS Approaches | RegAtlas Framework |
| :--- | :--- | :--- |
| **Formulation** | Pointwise binary classification or heuristic distance/eQTL thresholds | **Pairwise LambdaRank (NDCG@5)** optimizing within-locus relative gene ranking |
| **Functional Epigenomics** | Linear proximity or bulk Hi-C topological domains | **ENCODE-rE2G (Gschwind et al. 2023)** high-resolution brain predicted enhancer-to-promoter regulatory links |
| **Validation Rigor** | Random sample-level splits (prone to LD leakage) | **Chromosome-held-out cross-validation** ($K=5$) and $200\times$ within-locus permutation null |
| **Distal Discovery** | Often restricted to nearest-TSS gene | **65.8% non-nearest distal overrides** prioritized in schizophrenia GWAS |
| **Consensus Concordance** | Low or unbenchmarked against expert truth sets | **Significant enrichment against official PGC3 fine-mapped genes** (10/37 loci, OR = 8.84, $P = 1.43 \times 10^{-6}$ vs. nearest-TSS 4/37 loci, OR = 2.89, $P = 0.061$) |

---

## Pipeline Overview

```
 ┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
 │   Gold-Standard Loci      │      │    Tri-Modal Features     │      │   LightGBM LambdaRank     │
 │  1,075 Open Targets Loci  │ ───► │  • Spatial Distance (5)   │ ───► │  • Objective: LambdaRank  │
 │  35,358 Candidate Pairs   │      │  • GTEx v10 Brain eQTL(10)│      │  • GroupKFold by Chrom    │
 └───────────────────────────┘      │  • ENCODE-rE2G Links (6)  │      │  • Chromosome-Held-Out    │
                                    │  • Synergy Feature (1)    │      └─────────────┬─────────────┘
                                    └───────────────────────────┘                    │
                                                                                     ▼
 ┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
 │   Biological Validation   │      │   Distal Overrides (66%)  │      │   Schizophrenia GWAS      │
 │  • PGC3 Replication (OR 8.8x)│ ◄─── │  • Non-nearest rescue     │ ◄─── │  • 111 PGC3 Loci (Frozen) │
 │  • Synaptic Targets       │      │  • High-margin rankings   │      │  • 3,635 Candidate Pairs  │
 └───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

---

## Directory Structure

```
RegAtlas/
├── data/
│   ├── raw/                  # Downloaded raw public data (GTEx, Open Targets, ENCODE, PGC3)
│   ├── interim/              # Intermediate extracted features and coordinate overlaps
│   └── processed/            # Final model-ready parquet matrices (~12 MB total)
│       ├── training_matrix_dataset_a.parquet       # 1,075 training loci
│       ├── scz_application_matrix_dataset_b.parquet# 111 SCZ application loci
│       └── scz_prioritized_gene_rankings.parquet   # RegAtlas output rankings
├── scripts/
│   ├── 00_audit_current_input.py                   # Data integrity and leak-prevention audit
│   ├── 01_inspect_gold_standards.py                # Inspect Open Targets L2G universe
│   ├── 02_build_gold_standard_universe.py          # Construct +/- 500kb candidate gene space
│   ├── 03_extract_features_and_audit_coverage.py   # Extract 22 multi-omics features
│   ├── 04_train_and_evaluate_ranker.py             # LambdaRank training, CV, and ablations
│   ├── 05_build_scz_application_matrix.py          # Construct 111 PGC3 SCZ locus matrix
│   ├── 06_apply_frozen_regatlas_to_scz.py          # Apply frozen model to schizophrenia
│   ├── 07_pathway_and_concordance_analysis.py      # PGC3 concordance & g:Profiler GO enrichment
│   ├── 08_generate_publication_figures.py          # Generate Figures 1-5 (PDF/PNG 300 DPI)
│   ├── 09_compile_supplementary_tables.py          # Compile Supplementary Tables S1-S3 & Data S1-S2
│   ├── 10_generate_manuscript_draft.py             # Compile reproducible manuscript draft
│   └── 11_generate_supplementary_figures.py        # Generate Supplementary Figures S1-S4
├── results/
│   ├── downstream_biology/   # PGC3 concordance and g:Profiler GO enrichment outputs
│   ├── figures/              # Main publication figures (Figures 1-5)
│   ├── figures/supplementary/# Supplementary figures (Figures S1-S4)
│   ├── models/               # Frozen LightGBM booster model and feature importances
│   └── tables/               # Supplementary Tables (S1-S3) & Data (S1-S2)
├── environment.yml           # Conda environment definition
├── requirements.txt          # Python package requirements
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## Installation

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/Mansoor-Research/RegAtlas.git
cd RegAtlas

# Create and activate environment
conda env create -f environment.yml
conda activate regatlas
```

### Option 2: Pip Install

```bash
# Clone repository
git clone https://github.com/Mansoor-Research/RegAtlas.git
cd RegAtlas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Data Access & Downloads

The repository includes model-ready processed feature matrices in `data/processed/` (~12 MB). If you wish to reproduce the full feature extraction from scratch, the raw source files can be acquired from public sources:

| Dataset | Description | Source / Accession |
| :--- | :--- | :--- |
| **Open Targets L2G** | Gold standard causal gene labels (`otg_gs_230511.json`) | [Open Targets Genetics](https://ftp.ebi.ac.uk/pub/databases/opentargets/genetics/) |
| **GTEx v10 Brain** | Single-tissue cis-eQTLs (Cortex & Frontal Cortex BA9) | [GTEx Portal](https://gtexportal.org/home/downloads/adult-gtex/qtl) |
| **ENCODE-rE2G** | Predicted enhancer-to-gene regulatory links (DLPFC & Brain) | [ENCODE Portal / Gschwind et al. 2023](https://www.encodeproject.org/) |
| **PGC3 SCZ GWAS** | Schizophrenia GWAS summary statistics (Trubetskoy 2022) | [PGC Data Portal](https://pgc.unc.edu/for-researchers/download-results/) |
| **PGC3 SCZ Fine-Mapped Targets** | Official 120 fine-mapped genes (`Supplementary Table 12.xlsx`) | [Nature 2022 ESM Zip](https://doi.org/10.1038/s41586-022-04434-5) |

*Note: All raw datasets are freely and publicly available from their official consortia repositories. Model-ready processed matrices (~12 MB) are provided directly in `data/processed/` for instant replication.*

---

## Running the End-to-End Pipeline

To execute the complete pipeline from scratch or reproduce the figures and tables from processed matrices:

```bash
# Step 1: Build the Gold Standard Universe (+/- 500 kb candidate window)
python scripts/02_build_gold_standard_universe.py

# Step 2: Extract 22 Multi-Omics Features (GTEx v10 + ENCODE-rE2G)
python scripts/03_extract_features_and_audit_coverage.py

# Step 3: Train LambdaRank Model under Chromosome-Held-Out Cross-Validation (GroupKFold)
python scripts/04_train_and_evaluate_ranker.py

# Step 4: Construct Schizophrenia Application Matrix (111 PGC3 Loci)
python scripts/05_build_scz_application_matrix.py

# Step 5: Apply Frozen RegAtlas Model to Prioritize SCZ Candidate Genes
python scripts/06_apply_frozen_regatlas_to_scz.py

# Step 6: Perform PGC3 Landmark Concordance and Synaptic Pathway Analysis
python scripts/07_pathway_and_concordance_analysis.py

# Step 7: Generate Publication-Ready Figures (300 DPI PNG + Vector PDF)
python scripts/08_generate_publication_figures.py
python scripts/11_generate_supplementary_figures.py

# Step 8: Compile Supplementary Excel Workbooks (Tables S1-S3)
python scripts/09_compile_supplementary_tables.py
```

---

## Key Results Summary

### Cross-Validated Benchmark Performance (Dataset A, 1,075 Loci)

| Configuration | Modalities Included | Top-1 Accuracy | Recall@5 | MRR | NDCG@5 | Statistical Significance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Permutation Null** | Random within-locus assignment | 4.58% | N/A | 0.158 | 0.133 | Empirical null baseline |
| **Nearest-TSS** | Linear distance heuristic | 17.30% | 49.67% | 0.332 | 0.343 | Baseline proximity |
| **Distance Baseline** | Distance to gene body boundary | 30.51% | 55.81% | 0.431 | 0.437 | $P < 10^{-15}$ vs Nearest |
| **GTEx eQTL Only** | Brain Cortex + BA9 cis-eQTLs | 9.95% | 39.91% | 0.245 | 0.245 | $P < 0.001$ vs Null |
| **ENCODE-rE2G Only** | DLPFC + cortex enhancer links | 17.49% | 55.35% | 0.347 | 0.370 | Baseline single-modality |
| **RegAtlas (Full)** | **Distance + Brain eQTL + rE2G** | **32.37%** | **64.09%** | **0.470** | **0.490** | **empirical $P \le 0.005$ (>45.6 SD vs Null)** |

### Schizophrenia Validation Highlights (Dataset B, 111 Loci)
- **PGC3 Landmark Concordance:** Prioritized official fine-mapped schizophrenia risk genes from PGC3 (Trubetskoy et al., Nature 2022) at 10 of 37 testable loci (27.0%, Fisher's exact $P = 1.43 \times 10^{-6}$, OR = 8.84 [exact 95% CI: 3.82–18.64]), compared to 4 of 37 loci for the nearest-TSS heuristic (10.8%, Fisher's exact $P = 0.061$, OR = 2.89 [exact 95% CI: 0.74–8.13]). In a paired locus-by-locus comparison, RegAtlas alone prioritized the official gene at 7 loci vs 1 for nearest-TSS (paired exact binomial test $P = 0.035$). Replicated genes include *CUL9*, *DPYD*, *ENSG00000262319*, *IMMP2L*, *KLF6*, *MAD1L1*, *OPCML*, *PCGF3*, *RERE*, and *TMTC1*.
- **65.8% Distal Overrides:** Overrode the proximal nearest-TSS gene in 73 of 111 loci in favor of distal genes supported by convergent predicted enhancer-to-gene regulatory links and brain eQTLs.
- **Pathway Convergence & Robustness:** Prioritized candidates nominate biologically grounded targets across synaptic signaling, ion-channel, and receptor genes (*CACNA2D2*, *FYN*, *SHANK3*, *GRIA1*, *RIMS2*). Functional over-representation analysis against the protein-coding candidate background yielded nominal enrichment for synaptic transmission and receptor binding pathways, though no terms survived genome-wide multiple testing correction after excluding training-overlap genes (minimum FDR = 0.22). This transparent result highlights the critical necessity of empirical ground truth benchmarks (such as PGC3 fine-mapped targets) over pathway over-representation heuristics.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
