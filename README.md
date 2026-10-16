# RegAtlas: reproducible GWAS gene-shortlisting analysis

This repository contains the code and processed inputs for the revised RegAtlas gene-shortlisting analysis. It includes the complete-pool model fits, matched reduced-feature controls, fixed chromosome folds, held-out predictions, trained models and scripts to verify the reported results. The analysis begins with a frozen processed feature matrix; it does not reconstruct every feature from raw third-party downloads.

## What the analysis shows

The cross-trait benchmark has 1,075 loci and 35,358 candidate gene–locus pairs. In the complete mixed-biotype candidate pool, RegAtlas (LightGBM LambdaRank) reaches 64.17% Top-5 source-nomination recovery versus 53.78% for gene-body distance. A same-feature LightGBM classifier reaches 64.19%. Reduced-feature models show that annotation availability accounts for much of the improvement over learned proximity, and a protein-coding-only candidate-pool sensitivity changes the comparison. The positive labels are curated source nominations, not established causal genes. Gene-level molecular summaries do not demonstrate variant-to-enhancer-to-gene mechanisms.

## Files

| Location | Role |
| --- | --- |
| `data/processed/training_matrix_dataset_a.parquet` | Frozen 35,358-row, 41-column candidate matrix; 22 configured features enter the full models. |
| `data/splits/locus_split_assignments.csv` | Fixed chromosome-held-out outer folds and locus-held-out inner validation roles. |
| `config/final_protocol.json`, `config/internal_controls_protocol.json` | Features, objectives, seeds, stopping and evaluation settings. |
| `scripts/run_final_models.py`, `scripts/run_internal_controls.py` | Validating preflight by default; opt-in fitting. |
| `scripts/review_final_training.py`, `scripts/review_internal_controls.py` | Reload saved models and recalculate held-out results without fitting. |
| `results/final_fit_01/`, `results/internal_controls_01/` | Saved full and control model artifacts, predictions, metrics and run records. |
| `internal_controls_design.md` | Frozen post-development control plan and interpretation limits. |
| `provenance/` | Input hashes, package-file hashes and release verification. |
| `reference/historical_predictions.parquet` | Historical-score diagnostic required by the unchanged full-fit runner; never a model input or manuscript endpoint. |

Manuscript drafts, figure production scripts, screenshots, prior rejected analyses, raw third-party downloads and working notes are intentionally outside this code package.

## Environment

The recorded execution used Python 3.12.10 on Windows and the exact library versions in `requirements.txt`. From this directory:

```shell
python -m pip install -r requirements.txt
```

The scripts enforce pinned package versions and SHA-256 checks. Other systems may require a LightGBM native runtime and may not produce bit-identical refits.

## Verify the supplied results

```shell
python scripts/run_final_models.py
python scripts/run_internal_controls.py
python scripts/review_final_training.py
python scripts/review_internal_controls.py
```

The first two commands are preflight only. The two review commands reload 30 saved models across the full and reduced-feature analyses, compare saved test predictions, and recalculate summary metrics; the full-fit review also recalculates paired bootstrap intervals. It writes review files in `results/final_fit_01/`. Neither command trains a model.

## Optional refit

```shell
python scripts/run_final_models.py --fit --run-id reproduced_fit_01
```

This creates a new directory and never overwrites the supplied fit. The internal-control runner is locked to the included `internal_controls_01` run ID and refuses to overwrite it. To refit those controls, use a separate copy of this package, remove only that copy's control results, then run `python scripts/run_internal_controls.py --fit`. Do not delete or alter the supplied results in this release.

## Scope and provenance

Training begins with the supplied processed feature matrix, not a raw-data reconstruction. Source annotations include Ensembl GRCh38.p14, GTEx v10 Cortex and frontal cortex BA9 significant cis-eQTL pairs, and ENCODE-rE2G brain/DLPFC files ENCFF307BFL, ENCFF280TEO and ENCFF371VKL. The curated nomination source is `otg_gs_230511.json`. Candidate genes overlap a sentinel-centered ±500-kb window. The 22 full-model features comprise five proximity, ten brain cis-eQTL, six regulatory element–gene summary and one joint-availability feature. These are gene-level summaries; they are not locus-specific colocalization or variant-overlap tests.

Top-5 source-nomination recovery is the primary endpoint of the retrospective documented evaluation. Exact-score ties receive fractional expected credit and metrics weight loci equally. The 2,000-resample paired bootstrap intervals condition on the saved fits. The internal controls are post-development sensitivity analyses. The results do not establish a ranker advantage over the matched classifier, external-tool superiority, enhancer causality or independent disease validation.

## Data and license

Code is released under the repository's MIT license. The processed feature matrix and historical reference scores are supplied for reproduction of this analysis; rights to underlying third-party raw datasets are not asserted. Source releases and accessions are identified in the manuscript. The repository URL is `https://github.com/Mansoor-Research/RegAtlas`.
