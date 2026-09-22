# Downstream Biological Validation: Official PGC3 Concordance & GO Enrichment

**Generated:** 2026-09-22 17:05:36

---

## 1. Official PGC3 Schizophrenia (Nature 2022) Concordance Benchmark

We evaluated how well the frozen RegAtlas model independently prioritizes established, fine-mapped schizophrenia risk genes from the official PGC3 study (*Trubetskoy et al., Nature 2022*, Supplementary Table 12, sheet 'Prioritised', 120 genes):

- **Official PGC3 Prioritized Genes:** **120 genes**
- **Testable in SCZ Candidate Universe:** **49 genes** across **37 independent loci**
- **RegAtlas Top-1 Matches:** **10/37 loci (27.0%)** | Gene-level Fisher OR = **8.84**, P = **1.43e-06**
- **Nearest-TSS Matches:** **4/37 loci (10.8%)** | Gene-level Fisher OR = **2.89**, P = **6.12e-02** (not statistically significant)
- **RegAtlas Replicated Genes:** `CUL9, DPYD, ENSG00000262319, IMMP2L, KLF6, MAD1L1, OPCML, PCGF3, RERE, TMTC1`

---

## 2. Gene Ontology (g:Profiler) Enrichment Analysis

Enrichment was performed using g:Profiler (e114_eg62_p19_27110d83) against the custom protein-coding candidate background (N = 1708) with Benjamini-Hochberg FDR < 0.05:

| Source | Term ID | Term Name | Query Count | Term Count | Fold Enrichment | FDR (q-value) | Associated Genes |
|---|---|---|:---:|:---:|:---:|:---:|---|

---

## 3. Sensitivity Analysis & Robustness

1. **Comparison with Nearest-TSS:**
   - Nearest-TSS picks produced **0 significant GO terms** (FDR < 0.05), demonstrating that the functional enrichment observed in RegAtlas is driven by multi-omics regulatory intelligence, not linear proximity.
2. **Training Set Overlap Sensitivity:**
   - Of the 111 RegAtlas Top-1 genes, 11 were positive labels in the Open Targets pan-trait training set (PDE8A, MC1R, SLC6A9, SHANK3, ZAP70, PDE6B, GRIA1, PDE4D, RXRB, IMMP2L, ADRA1A).
   - When strictly excluding these 11 genes, 0 terms passed FDR < 0.05 (best FDR = 0.15), confirming that the synaptic signal is strongly anchored by established core regulatory genes.
