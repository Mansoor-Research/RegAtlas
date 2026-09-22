# Raw Data Directory

This directory is for optional raw source downloads if you wish to re-extract features or re-run external benchmarks from scratch (`scripts/02_build_gold_standard_universe.py` and `scripts/03_extract_features_and_audit_coverage.py`).

The model-ready processed feature matrices are already provided in `../processed/` (~12 MB), allowing you to immediately run model training and all downstream evaluations without downloading raw data.

### Public Raw Sources:
- **Open Targets Genetics L2G:** `otg_gs_230511.json` from https://ftp.ebi.ac.uk/pub/databases/opentargets/genetics/
- **GTEx Analysis v10 Brain cis-eQTLs:** Cortex & BA9 from https://gtexportal.org/home/downloads/adult-gtex/qtl
- **ENCODE-rE2G Predicted Enhancer Links:** DLPFC & Whole Brain from https://www.encodeproject.org/ (Gschwind et al. 2023)
- **PGC3 SCZ GWAS Summary Stats:** Trubetskoy et al. 2022 from https://pgc.unc.edu/for-researchers/download-results/
- **PGC3 SCZ Supplementary Table 12 (`Supplementary Table 12.xlsx`):** Required by `scripts/07_pathway_and_concordance_analysis.py` for evaluating concordance against the official 120 fine-mapped schizophrenia risk genes. Download the Supplementary Tables zip (`41586_2022_4434_MOESM11_ESM.zip`) from Nature (*Trubetskoy et al. 2022*, Nature 604, 502–508; DOI: [10.1038/s41586-022-04434-5](https://doi.org/10.1038/s41586-022-04434-5)), extract `Supplementary Table 12.xlsx`, and place it in `data/raw/pgc_scz/Supplementary Table 12.xlsx`.
