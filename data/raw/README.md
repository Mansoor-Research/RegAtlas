# Raw input data

Raw inputs are not tracked in this repository because of their size and the terms of their original sources. Download them to the paths below to rerun steps 1, 2 and 5. Steps 3–10 run from the processed files in `data/processed/`.

| Path | Description | Source |
|---|---|---|
| `data/raw/opentargets/otg_gs_230511.json` | Open Targets Genetics gold-standard loci (May 2023) | https://ftp.ebi.ac.uk/pub/databases/opentargets/genetics/ |
| `data/raw/ensembl/Homo_sapiens.GRCh38.gtf.gz` | Ensembl GRCh38 gene annotation | https://www.ensembl.org/ |
| `data/raw/gtex/GTEx_Analysis_v10_eQTL_updated/Brain_Cortex.v10.eQTLs.signif_pairs.parquet` | GTEx v10 significant cis-eQTL pairs, Brain Cortex | https://gtexportal.org/home/downloads/adult-gtex/qtl |
| `data/raw/gtex/GTEx_Analysis_v10_eQTL_updated/Brain_Frontal_Cortex_BA9.v10.eQTLs.signif_pairs.parquet` | GTEx v10 significant cis-eQTL pairs, Frontal Cortex BA9 | same as above |
| `data/raw/encode_re2g/DLPFC_ENCFF280TEO.bed.gz`, `DLPFC_ENCFF371VKL.bed.gz`, `Brain_ENCFF307BFL.bed.gz` | ENCODE-rE2G predicted enhancer–gene links | https://www.encodeproject.org/ (search by accession) |
| `data/raw/pgc_scz/Supplementary Table 12.xlsx` | PGC3 prioritized genes (included in this repository) | Trubetskoy *et al.* 2022, *Nature* 604:502–508, Supplementary Tables (`41586_2022_4434_MOESM11_ESM.zip`) |
| `data/current/rna_seq_eQTLs_gwas_schizophrenia.csv` | PGC3 schizophrenia GWAS summary statistics (columns `CHROM, ID, POS, A1, A2, BETA, SE, PVAL`) merged with gene-level expression columns used in step 6 | PGC3 summary statistics: https://pgc.unc.edu/for-researchers/download-results/ |
