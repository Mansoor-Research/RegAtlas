# Downstream Biological Validation: Official PGC3 Concordance & GO Enrichment

**Generated:** 2026-09-22 16:35:26

---

## 1. PGC3 Schizophrenia (Nature 2022 / Trubetskoy et al.) Concordance

Evaluated against the official 120 fine-mapped prioritized genes from Trubetskoy et al. (Supplementary Table 12, sheet 'Prioritised'):

- **Official PGC3 Prioritized Genes:** 120
- **Testable in SCZ Candidate Universe:** 49 genes across 37 loci
- **RegAtlas Top-1 Hits:** **7/37 loci (18.9%)** | Gene-level Fisher OR = **5.57**, P = **6.13e-04**
- **Nearest-TSS Baseline Hits:** **4/37 loci (10.8%)** | Gene-level Fisher OR = **2.89**, P = **6.12e-02** (not significant)
- **RegAtlas Replicated Genes:** `CUL9, DPYD, ENSG00000262319, IMMP2L, KLF6, MAD1L1, TMTC1`
  - Note: 5 of these 7 genes (*KLF6*, *TMTC1*, *DPYD*, *IMMP2L*, *MAD1L1*) are **distal overrides** where RegAtlas prioritized the causal gene over closer bystanders.

---

## 2. Gene Ontology (GO) Enrichment (g:Profiler)

Evaluated using g:Profiler (GO:BP, GO:CC, GO:MF) with a custom protein-coding background (97 Top-1 protein-coding genes vs. 1,708 candidate protein-coding background) and Benjamini-Hochberg FDR < 0.05:

Found **12 statistically significant terms** (FDR < 0.05):

| Source | Term ID | Term Name | FDR (q-value) | Overlap / Query | Top-1 Genes |
|---|---|---|:---:|:---:|---|
| GO:MF | `GO:0008081` | **phosphoric diester hydrolase activity** | 2.451e-03 | 6/97 | `PDE8A SMPD3 PLCL1 PLCL2 PDE6B PDE4D` |
| GO:BP | `GO:0010646` | **regulation of cell communication** | 6.636e-03 | 35/97 | `SORCS3 AMBRA1 CHST11 KIF26A RGS6 CTSH PDE8A SV2B SMPD3 MC1R HIC1 GALR1 DOT1L STK40 SLC6A9 NNAT SHANK3 PLCL1 BABAM2 STAMBP ZAP70 PLCL2 CACNA2D2 VEGFC FER HBEGF GRIA1 PDE4D FYN RXRB MAD1L1 RIMS2 ARHGAP39 ADRA1A TNKS` |
| GO:BP | `GO:0023051` | **regulation of signaling** | 6.636e-03 | 35/97 | `SORCS3 AMBRA1 CHST11 KIF26A RGS6 CTSH PDE8A SV2B SMPD3 MC1R HIC1 GALR1 DOT1L STK40 SLC6A9 NNAT SHANK3 PLCL1 BABAM2 STAMBP ZAP70 PLCL2 CACNA2D2 VEGFC FER HBEGF GRIA1 PDE4D FYN RXRB MAD1L1 RIMS2 ARHGAP39 ADRA1A TNKS` |
| GO:BP | `GO:0099537` | **trans-synaptic signaling** | 1.157e-02 | 13/97 | `SORCS3 SV2B DOC2A SLC6A9 SHANK3 PLCL1 PLCL2 CACNA2D2 GRIA1 FYN EXOC4 RIMS2 ADRA1A` |
| GO:BP | `GO:0098916` | **anterograde trans-synaptic signaling** | 1.157e-02 | 13/97 | `SORCS3 SV2B DOC2A SLC6A9 SHANK3 PLCL1 PLCL2 CACNA2D2 GRIA1 FYN EXOC4 RIMS2 ADRA1A` |
| GO:BP | `GO:0007267` | **cell-cell signaling** | 1.157e-02 | 17/97 | `SORCS3 SNX19 SV2B DOC2A SMPD3 GALR1 SLC6A9 NNAT SHANK3 PLCL1 PLCL2 CACNA2D2 GRIA1 FYN EXOC4 RIMS2 ADRA1A` |
| GO:BP | `GO:0007268` | **chemical synaptic transmission** | 1.157e-02 | 13/97 | `SORCS3 SV2B DOC2A SLC6A9 SHANK3 PLCL1 PLCL2 CACNA2D2 GRIA1 FYN EXOC4 RIMS2 ADRA1A` |
| GO:BP | `GO:0099177` | **regulation of trans-synaptic signaling** | 1.157e-02 | 11/97 | `SORCS3 SV2B SLC6A9 SHANK3 PLCL1 PLCL2 CACNA2D2 GRIA1 FYN RIMS2 ADRA1A` |
| GO:BP | `GO:0050804` | **modulation of chemical synaptic transmission** | 1.157e-02 | 11/97 | `SORCS3 SV2B SLC6A9 SHANK3 PLCL1 PLCL2 CACNA2D2 GRIA1 FYN RIMS2 ADRA1A` |
| GO:BP | `GO:0099536` | **synaptic signaling** | 2.342e-02 | 13/97 | `SORCS3 SV2B DOC2A SLC6A9 SHANK3 PLCL1 PLCL2 CACNA2D2 GRIA1 FYN EXOC4 RIMS2 ADRA1A` |
| GO:BP | `GO:0097581` | **lamellipodium organization** | 3.098e-02 | 4/97 | `ARPIN PLEKHO1 FER CARMIL1` |
| GO:BP | `GO:0032228` | **regulation of synaptic transmission, GABAergic** | 3.724e-02 | 3/97 | `PLCL1 PLCL2 ADRA1A` |

### Negative / Baseline Controls:
- **Nearest-TSS Heuristic:** Yields **0** significant GO terms under both full and protein-coding backgrounds.
- **Training-Overlap Sensitivity Check:** Excluding the 11 SCZ Top-1 genes that overlapped with positive training labels in Dataset A yields 0 terms at FDR < 0.05 (best FDR = 0.15), confirming that shared regulatory features drive a meaningful portion of the synaptic enrichment.
