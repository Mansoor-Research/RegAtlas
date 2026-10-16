# Benchmark data

## Included files

| File | Shape | Purpose |
| --- | --- | --- |
| `processed/training_matrix_dataset_a.parquet` | 35,358 rows × 41 columns | Candidate metadata, source labels and processed features |
| `splits/locus_split_assignments.csv` | 1,075 loci | Fixed chromosome folds and inner training, validation and test roles |

The candidate key is `(locus_id, gene_id)`. Each locus has one positive source nomination (`label = 1`); other candidates have `label = 0`. The matrix includes protein-coding and noncoding candidate genes whose gene bodies overlap the sentinel-centred ±500-kb window. The 22 full-model feature names are listed, in training order, in `config/final_protocol.json`. Other columns carry candidate identifiers, annotations and evaluation metadata.

## Source resources

| Resource | Release or identifier | Source |
| --- | --- | --- |
| Open Targets Genetics curated nominations | `otg_gs_230511.json` | [Gold-standard catalogue](https://github.com/opentargets/genetics-gold-standards) |
| Ensembl human gene annotation | GRCh38.p14 | [Ensembl human annotation](https://www.ensembl.org/Homo_sapiens/Info/Annotation) |
| GTEx significant cis-eQTL pairs | v10, Brain Cortex and Brain Frontal Cortex BA9 | [GTEx downloads](https://gtexportal.org/home/downloads/adult-gtex) |
| ENCODE-rE2G regulatory links | Brain/DLPFC annotations | [ENCFF307BFL](https://www.encodeproject.org/files/ENCFF307BFL/), [ENCFF280TEO](https://www.encodeproject.org/files/ENCFF280TEO/), [ENCFF371VKL](https://www.encodeproject.org/files/ENCFF371VKL/) |

The supplied matrix is the fixed processed input for the reported fits. Molecular summaries describe annotations available for a gene; they are not a direct test of whether a particular GWAS variant regulates that gene. Raw third-party source files are not redistributed in this package.

## Integrity

`provenance/input_manifest.json` records the matrix and split checksums and dependency versions. Training and verification scripts validate these records before reading the analysis inputs. No local absolute paths are required.
