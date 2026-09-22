# RegAtlas

**RegAtlas** is a learning-to-rank framework for post-GWAS locus-to-gene prioritization. For each GWAS locus it ranks all candidate genes within ±500 kb of the lead variant, using genomic distance, GTEx v10 brain cis-eQTLs and ENCODE-rE2G predicted enhancer–gene links in a LightGBM LambdaRank model.

This repository contains the code and processed data needed to reproduce the analyses in:

> Jan SM, *et al.* RegAtlas: A Learning-to-Rank Multi-Omics Framework for Post-GWAS Locus-to-Gene Mapping in Psychiatric Disorders. *Manuscript under review.*

## Overview

| Stage | Data | Description |
|---|---|---|
| Training (Dataset A) | 1,075 Open Targets gold-standard loci; 35,358 candidate genes | One gold-standard gene per locus; all other genes in the window are competitors |
| Features | 22 per locus–gene pair | Distance (5), GTEx v10 Brain Cortex and Frontal Cortex BA9 cis-eQTLs (10), ENCODE-rE2G DLPFC and whole-brain links (6), eQTL × rE2G indicator (1) |
| Model | LightGBM LambdaRank | Genes grouped by locus; NDCG objective |
| Evaluation | Chromosome-held-out 5-fold CV | Early stopping on an inner validation split of the training loci; 200 within-locus label permutations; 7-way feature-group ablation |
| Application (Dataset B) | 111 PGC3 schizophrenia loci; 3,635 candidate genes | Frozen model applied without retraining |

## Key results

Cross-validation on Dataset A (1,075 loci):

| Model | Top-1 | Recall@5 | MRR |
|---|---|---|---|
| Nearest TSS | 17.3% | 49.7% | 0.332 |
| Distance features only | 30.5% | 55.8% | 0.431 |
| Distance + ENCODE-rE2G | 33.7% | 65.3% | 0.484 |
| **RegAtlas (all 22 features)** | **32.4%** | **64.1%** | **0.470** |
| Within-locus permutation null | 4.6% | – | 0.158 |

- Enhancer–gene links improve ranking beyond distance and remain informative within distance-matched strata. Brain cis-eQTL features did not improve cross-validated performance (see the manuscript for discussion).
- Schizophrenia: the top-ranked gene was an official PGC3 prioritized gene ([Trubetskoy *et al.* 2022](https://doi.org/10.1038/s41586-022-04434-5), Supplementary Table 12) at 10 of 37 testable loci (gene-level OR = 8.84, P = 1.4 × 10⁻⁶), compared with 4 of 37 for the nearest-TSS heuristic (P = 0.061).
- Gene Ontology analysis of the prioritized genes showed no enrichment at FDR < 0.05.

## Repository structure

```
RegAtlas/
├── scripts/            Analysis pipeline, numbered in run order
├── data/
│   ├── raw/            Public source files (not tracked; see data/raw/README.md)
│   └── processed/      Model-ready feature matrices and rankings
├── results/
│   ├── model_evaluation/     Cross-validation outputs
│   ├── downstream_biology/   PGC3 concordance and GO enrichment tables
│   ├── figures/              Figures 2–5 and Supplementary Figures S1–S4
│   └── tables/               Supplementary Tables and Supplementary Data
├── requirements.txt
├── environment.yml
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
git clone https://github.com/Mansoor-Research/RegAtlas.git
cd RegAtlas
conda env create -f environment.yml
conda activate regatlas
```

Alternatively, with Python 3.10: `pip install -r requirements.txt`.

## Reproducing the analyses

The processed matrices in `data/processed/` are sufficient to run steps 3–10. Steps 1, 2 and 5 require the raw inputs described in [`data/raw/README.md`](data/raw/README.md).

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/01_build_candidate_universe.py` | Dataset A candidate universe |
| 2 | `scripts/02_extract_features.py` | Dataset A feature matrix |
| 3 | `scripts/03_train_and_evaluate.py` | Cross-validation, baselines, ablation, permutation null |
| 4 | `scripts/04_distance_matched_analysis.py` | Distance-matched rE2G analysis |
| 5 | `scripts/05_build_scz_matrix.py` | Dataset B (schizophrenia) feature matrix |
| 6 | `scripts/06_prioritize_scz_genes.py` | Frozen-model rankings for Dataset B |
| 7 | `scripts/07_pgc3_concordance_and_go.py` | PGC3 concordance and GO enrichment (requires internet access to g:Profiler) |
| 8 | `scripts/08_main_figures.py` | Figures 2–5 |
| 9 | `scripts/09_supplementary_tables.py` | Supplementary Tables and Data |
| 10 | `scripts/10_supplementary_figures.py` | Supplementary Figures S1–S4 |

Run the scripts from the repository root, for example:

```bash
python scripts/03_train_and_evaluate.py
```

## Data sources

| Resource | Version | Source |
|---|---|---|
| Open Targets Genetics gold standard | May 2023 (`otg_gs_230511.json`) | [Mountjoy *et al.* 2021](https://doi.org/10.1038/s41588-021-00945-5) |
| GTEx cis-eQTLs (Brain Cortex, Frontal Cortex BA9) | v10 | [GTEx Portal](https://gtexportal.org/home/downloads/adult-gtex/qtl) |
| ENCODE-rE2G enhancer–gene predictions (DLPFC, whole brain) | ENCFF280TEO, ENCFF371VKL, ENCFF307BFL | [ENCODE Portal](https://www.encodeproject.org/); [Gschwind *et al.* 2023](https://doi.org/10.1101/2023.11.09.563812) |
| Ensembl gene annotation | GRCh38 | [Ensembl](https://www.ensembl.org/) |
| PGC3 schizophrenia GWAS | Trubetskoy *et al.* 2022 | [PGC downloads](https://pgc.unc.edu/for-researchers/download-results/) |
| g:Profiler | e114_eg62_p19 | [g:Profiler](https://biit.cs.ut.ee/gprofiler) |

## Citation

If you use RegAtlas, please cite the article above (citation details will be updated on publication). See also [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
