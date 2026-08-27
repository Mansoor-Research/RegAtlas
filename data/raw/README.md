# Raw Data Directory

This directory is for optional raw source downloads if you wish to run feature extraction from scratch (`01_download_public_sources.py` and `03_extract_features_and_audit_coverage.py`).

The model-ready processed feature matrices are already provided in `../processed/` (~12 MB), allowing you to immediately run model training and all downstream evaluations without downloading raw data.

### Public Raw Sources:
- **Open Targets Genetics L2G:** `otg_gs_230511.json` from https://ftp.ebi.ac.uk/pub/databases/opentargets/genetics/
- **GTEx Analysis v10 Brain cis-eQTLs:** Cortex & BA9 from https://gtexportal.org/home/downloads/adult-gtex/qtl
- **ENCODE-rE2G Enhancer Links:** DLPFC & Whole Brain from https://www.encodeproject.org/ (Nasser et al. 2024)
- **PGC3 SCZ GWAS Summary Stats:** Trubetskoy et al. 2022 from https://pgc.unc.edu/for-researchers/download-results/
