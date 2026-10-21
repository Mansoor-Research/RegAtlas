# RegAtlas

**Candidate gene shortlisting at GWAS loci using brain regulatory annotations**

RegAtlas takes a list of GWAS sentinel loci and ranks candidate genes within each locus. It combines genomic proximity with gene-level GTEx v10 cortex/BA9 cis-eQTL and ENCODE-rE2G brain/DLPFC summaries in a LightGBM LambdaRank model. The package includes the gene annotation resource, processed molecular summaries, saved models, fixed benchmark folds and evaluation results.

## Shortlist genes at your loci

Use Python 3.12 and install the pinned dependencies:

```shell
git clone https://github.com/Mansoor-Research/RegAtlas.git
cd RegAtlas
python -m venv .venv
```

Activate the environment using your operating system's command, then run:

```shell
python -m pip install -r requirements.txt
python scripts/shortlist.py --loci examples/loci.csv --genome-build GRCh38 --output results/example_application
```

No model fitting or raw GTEx downloads are needed for this prediction command. It uses the supplied versioned gene resource and models. Choose a new output directory for each application; existing directories are not overwritten.

### Input

Provide a CSV with these columns:

```csv
locus_id,chrom,sentinel_pos
example_1,1,10000000
example_2,7,20000000
example_3,16,10000000
```

These are synthetic example coordinates, not disease findings. Positions must be 1-based GRCh38 coordinates. Chromosomes 1-22 and X are supported, with or without a `chr` prefix. Each locus ID must be unique. Harmonize other genome builds before running. Use appropriately defined GWAS loci; this command does not perform LD clumping, fine-mapping or association testing on a full GWAS summary-statistics file.

Candidate genes are selected when their annotated bodies overlap the sentinel-centred ±500-kb window. Retained biotypes match the benchmark. A broad published GWAS region can extend beyond this fixed window; the output does not claim to cover every candidate across that broader region.

### Output

| File | Contents |
|---|---|
| `rankings.csv` / `rankings.parquet` | All candidate scores, gene identifiers/names, minimum and maximum tied ranks, model fold and benchmark-window overlap |
| `shortlist.csv` | Candidates within the requested rank cutoff, including all boundary ties |
| `candidate_features.parquet` | Candidate metadata and ordered model predictors |
| `candidate_coverage.csv` | Number of eligible genes at every supplied locus, including empty windows |
| `benchmark_overlap.csv` | Overlap with sentinel-centred benchmark windows and exact-sentinel matches |
| `run_manifest.json` | Model hashes, feature order, genome build and run dimensions |

The default cutoff is five. Use `--top-k 3` for another cutoff. All genes tied across a cutoff are retained, so a shortlist can contain more than the requested number. `ranker_rank_min` and `ranker_rank_max` show the occupied rank interval. Scores are relative ranking values, not causal probabilities, disease-risk estimates or expression-effect directions. Compare ranks within a locus; do not interpret scores from different fold models as a global gene ranking.

Each chromosome is scored using the saved model that held that chromosome out of both fitting and early-stopping validation. The overlap report describes the full benchmark; a reported overlap does not mean the selected model trained on that region. Input molecular annotations remain gene-level summaries, not locus-specific colocalization or credible-variant regulatory overlap.

## Benchmark and disease application

The cross-trait benchmark contains 1,075 loci, 35,358 candidate pairs and 22 predictors. Version 2.0.0 harmonizes the target-gene column name in all three ENCODE-rE2G sources. It retains the original candidates, labels, folds, features, hyperparameters and seeds, and refits the full models and reduced-feature controls. Earlier results remain in Git history; do not mix model/resource versions.

| Method | Top-5 nomination recovery |
|---|---:|
| RegAtlas LambdaRank | 71.39% |
| Matched binary classifier | 72.16% |
| Gene-body distance | 53.78% |
| Nearest TSS | 49.67% |

The paired RegAtlas gain over gene-body distance is 17.60 percentage points (95% paired bootstrap interval 15.15 to 20.16). The ranker-classifier Top-5 difference is -0.77 points (-2.29 to 0.78); this interval does not establish equivalence. These metrics describe recovery of curated source nominations, not experimentally proven causal genes.

The schizophrenia application uses 287 published PGC regions and 6,439 candidate pairs. It illustrates use of the framework and assesses published genetic support, without assigning the general benchmark recovery rate to schizophrenia. Disease evidence tables retain source provenance and distinguish same-GWAS prioritization from complementary rare-variant evidence. The application does not establish a causal regulatory mechanism or treatment benefit.

To reproduce the rankings from the supplied harmonized sentinel coordinates:

```shell
python scripts/shortlist.py --loci examples/schizophrenia_2022_loci.csv --genome-build GRCh38 --output results/schizophrenia_application
```

The coordinates are derived from [PGC 2022 Supplementary Table 3](https://doi.org/10.1038/s41586-022-04434-5), uniquely mapped from GRCh37 to GRCh38. `examples/schizophrenia_2022_locus_provenance.csv` records the original coordinates, mapping status and source-region bounds. These are sentinel-centred windows, not a replacement for the publication's broader locus definitions. The tie-inclusive RegAtlas shortlist contains 2,223 pairs across 2,169 genes. Boundary ties expand 116 loci beyond five genes, with a maximum of 51. Published PGC and rare-variant annotations are supplied with the manuscript's supplementary tables, where mapping eligibility and evidence dependence are explicit.

## Reproduce the benchmark

```shell
python scripts/run_final_models.py
python scripts/run_internal_controls.py
python scripts/review_final_training.py
python scripts/review_internal_controls.py
```

The first two commands validate the fixed inputs. The review commands reload the supplied models and recalculate metrics; they do not train models. For explicit refitting into new directories:

```shell
python scripts/run_final_models.py --fit --run-id reproduced_full
python scripts/run_internal_controls.py --fit --run-id reproduced_controls
```

Control refitting uses the supplied full-model predictions as its reference. Versions and deterministic settings are recorded, but bit-identical results across all hardware/software combinations are not guaranteed.

## Resources and scientific scope

Five proximity features are calculated for each supplied locus. Ten GTEx summaries, six rE2G summaries and one joint-availability indicator are retrieved from the bundled gene resource. Missing catalog representation is encoded consistently with training and does not mean absence of biological regulation. See `data/README.md` for source accessions and feature details, and `provenance/correction_record.json` for the source-schema correction.

Underlying third-party resources retain their providers' terms. The schizophrenia example supplies published sentinel coordinates and derived mappings, not individual-level data or a full GWAS summary-statistics dataset.

Citation metadata is in `CITATION.cff`. Code is distributed under the existing MIT license.
