# RegAtlas

**Learning-to-rank for GWAS gene shortlisting with brain regulatory annotations**

RegAtlas ranks candidate genes within GWAS loci using LightGBM LambdaRank. This repository provides the processed benchmark, fixed evaluation partitions, training and verification scripts, trained models, and numerical results for a controlled cross-trait evaluation.

## Benchmark and results

The benchmark contains **1,075 loci**, **35,358 locus–gene candidate pairs**, and **22 features**: five proximity features, ten brain cis-eQTL summaries, six ENCODE-rE2G summaries, and one joint-availability indicator. Each locus has one curated source nomination.

| Method | Top-5 nomination recovery |
| --- | ---: |
| RegAtlas, LightGBM LambdaRank | 64.17% |
| Matched LightGBM binary classifier | 64.19% |
| Gene-body distance | 53.78% |
| Nearest TSS | 49.67% |

RegAtlas improves recovery over gene-body distance by **10.38 percentage points** (95% paired bootstrap interval, 7.70–13.23). The ranker–classifier difference is −0.02 points (−1.76 to 1.77). Differences are calculated before rounding the displayed percentages. Reduced-feature and coding-only controls assess the roles of annotation availability and candidate definition.

## Installation

The reference execution used Python 3.12.10 on Windows. Install the pinned dependencies in a virtual environment:

```shell
git clone https://github.com/Mansoor-Research/RegAtlas.git
cd RegAtlas
python -m venv .venv
```

Activate `.venv` using the command for your operating system, then install the dependencies:

```shell
python -m pip install -r requirements.txt
```

The scripts validate dependency versions and SHA-256 hashes. LightGBM may require a native runtime on some systems. Refits on different systems are not guaranteed to be bit-identical.

## Verify the supplied analysis

Run these commands from the repository root:

```shell
python scripts/run_final_models.py
python scripts/run_internal_controls.py
python scripts/review_final_training.py
python scripts/review_internal_controls.py
```

The first two commands validate inputs and configuration. The verification commands reload all 30 saved models, compare held-out predictions, and recalculate reported metrics and paired intervals. They do not train models. The full-model verification report is written to `results/verification/`, which Git ignores.

## Refit models

Explicit fitting creates new output directories and refuses to overwrite existing results:

```shell
python scripts/run_final_models.py --fit --run-id reproduced_full
python scripts/run_internal_controls.py --fit --run-id reproduced_controls
```

The control refit uses the supplied full-feature predictions as its reference. Feature sets, objectives, seeds and stopping parameters are defined in `config/`.

## Repository structure

```text
config/        Model configurations and evaluation settings
data/          Processed candidate matrix, fixed folds and data documentation
provenance/    Input checksums, release checksums and source-history records
results/       Trained models, held-out predictions and numerical results
scripts/       Training, evaluation and saved-model verification
```

See [data documentation](data/README.md) for the matrix schema and source accessions, and [control design](internal_controls_design.md) for the sensitivity analyses. The repository contains the analysis package required to reproduce the reported numerical results.

## Interpretation and reproducibility

The labels are curated source nominations. Recovery measures agreement with those nominations, rather than experimentally established causal genes. The retrospective analysis starts from the supplied processed feature matrix; it does not rebuild every feature from raw downloads. Regulatory features are gene-level summaries, and the reduced-feature controls show that annotation availability captures much of the gain over learned proximity.

Five outer folds hold out chromosomes. Inner validation holds out loci. Exact-score ties receive fractional expected credit, and metrics weight loci equally. The 2,000-resample paired bootstrap intervals condition on the fitted models. Similar observed recovery does not establish equivalence between learning objectives.

Saved model files, predictions and execution metadata retain their original checksums. [Source history](provenance/source_history.json) records public-package changes against the generating code, including removal of an optional historical-score diagnostic. Training settings, features and fitted models are unchanged.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). Code is distributed under the [MIT license](LICENSE). The processed data are supplied for reproduction of this analysis; underlying third-party resources retain their respective terms.
