#!/usr/bin/env python
"""
Compile Complete Supplementary Tables and Supplementary Data Workbooks for RegAtlas.

Outputs:
- Supplementary_Tables.xlsx:
    Sheet 1: README (Data Dictionary)
    Sheet 2: Table S1 - SCZ Rankings (3,635 rows)
    Sheet 3: Table S2 - Ablation (10 rows)
    Sheet 4: Table S3 - Pathway Enrichment (12 significant GO terms from g:Profiler)
- Supplementary_Data.xlsx:
    Sheet 1: README (Data Dictionary)
    Sheet 2: Data S1 - Training Universe (35,358 rows)
    Sheet 3: Data S2 - SCZ Application (3,635 rows)
"""

import logging
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
GO_CSV = RESULTS_DIR / "downstream_biology" / "go_enrichment" / "regatlas_pc.csv"


def write_readme_sheet(wb, readme_lines):
    ws = wb.active
    ws.title = 'README'
    ws.column_dimensions['A'].width = 120
    header_font = Font(bold=True, size=11)
    for i, line in enumerate(readme_lines, start=1):
        cell = ws.cell(row=i, column=1, value=line)
        if (line.startswith('===') or line.startswith('---') or
                line.startswith('Sheet') or line.startswith('Column') or
                line.startswith('Supplementary')):
            cell.font = header_font


def write_data_sheet(wb, sheet_name, df):
    ws = wb.create_sheet(sheet_name)
    header_font = Font(bold=True, size=10)
    header_fill = PatternFill(start_color='D9E1F2', end_color='D9E1F2', fill_type='solid')
    thin_border = Border(bottom=Side(style='thin'))

    for c, col_name in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=c, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border
        cell.alignment = Alignment(horizontal='center')

    for r, row in enumerate(df.itertuples(index=False), start=2):
        for c, val in enumerate(row, start=1):
            ws.cell(row=r, column=c, value=val)

    for c, col_name in enumerate(df.columns, start=1):
        max_len = max(len(str(col_name)), df[col_name].astype(str).str.len().max())
        col_letter = ws.cell(row=1, column=c).column_letter
        ws.column_dimensions[col_letter].width = min(max_len + 3, 40)


def build_supplementary_tables():
    log.info("Compiling Supplementary_Tables.xlsx...")

    # 1. SCZ Rankings (Table S1)
    df_scz = pd.read_parquet(DATA_PROCESSED / "scz_prioritized_gene_rankings.parquet")
    s1_cols = [
        'locus_id', 'chrom', 'sentinel_rsid', 'sentinel_pos', 'gwas_pval', 'gwas_beta',
        'regatlas_rank', 'regatlas_score', 'gene_id', 'gene_name', 'gene_biotype',
        'abs_tss_distance', 'tss_distance_rank', 'is_nearest_tss',
        'has_any_brain_eqtl', 'brain_eqtl_max_slope', 'brain_eqtl_max_neg_log10_pval',
        'has_re2g_link', 're2g_max_score', 're2g_total_elements', 'has_both_eqtl_and_re2g'
    ]
    df_s1 = df_scz[s1_cols].copy()

    # 2. Ablation & Baselines (Table S2)
    df_s2 = pd.DataFrame([
        {'Model_Configuration': 'A: Distance Only', 'Num_Features': 5, 'Top1_Accuracy_Pct': 30.51, 'Recall_at_3_Pct': 46.98, 'Recall_at_5_Pct': 55.81, 'MRR': 0.431, 'NDCG_at_5': 0.437},
        {'Model_Configuration': 'B: GTEx eQTL Only', 'Num_Features': 10, 'Top1_Accuracy_Pct': 9.95, 'Recall_at_3_Pct': 24.93, 'Recall_at_5_Pct': 39.91, 'MRR': 0.245, 'NDCG_at_5': 0.245},
        {'Model_Configuration': 'C: rE2G Only', 'Num_Features': 6, 'Top1_Accuracy_Pct': 17.49, 'Recall_at_3_Pct': 40.65, 'Recall_at_5_Pct': 55.35, 'MRR': 0.347, 'NDCG_at_5': 0.370},
        {'Model_Configuration': 'D: Distance + GTEx', 'Num_Features': 15, 'Top1_Accuracy_Pct': 29.86, 'Recall_at_3_Pct': 46.79, 'Recall_at_5_Pct': 57.21, 'MRR': 0.429, 'NDCG_at_5': 0.439},
        {'Model_Configuration': 'E: Distance + rE2G', 'Num_Features': 11, 'Top1_Accuracy_Pct': 33.67, 'Recall_at_3_Pct': 55.91, 'Recall_at_5_Pct': 65.30, 'MRR': 0.484, 'NDCG_at_5': 0.504},
        {'Model_Configuration': 'F: GTEx + rE2G', 'Num_Features': 17, 'Top1_Accuracy_Pct': 17.40, 'Recall_at_3_Pct': 41.30, 'Recall_at_5_Pct': 53.02, 'MRR': 0.340, 'NDCG_at_5': 0.357},
        {'Model_Configuration': 'G: Full Model (RegAtlas)', 'Num_Features': 22, 'Top1_Accuracy_Pct': 32.37, 'Recall_at_3_Pct': 53.12, 'Recall_at_5_Pct': 64.09, 'MRR': 0.470, 'NDCG_at_5': 0.490},
        {'Model_Configuration': 'Nearest TSS Heuristic', 'Num_Features': 1, 'Top1_Accuracy_Pct': 17.30, 'Recall_at_3_Pct': 38.51, 'Recall_at_5_Pct': 49.67, 'MRR': 0.332, 'NDCG_at_5': 0.343},
        {'Model_Configuration': 'Gene Body Boundary Heuristic', 'Num_Features': 1, 'Top1_Accuracy_Pct': 30.79, 'Recall_at_3_Pct': 45.02, 'Recall_at_5_Pct': 54.05, 'MRR': 0.425, 'NDCG_at_5': 0.427},
        {'Model_Configuration': 'Permutation Null (200x mean)', 'Num_Features': 22, 'Top1_Accuracy_Pct': 4.58, 'Recall_at_3_Pct': None, 'Recall_at_5_Pct': None, 'MRR': 0.158, 'NDCG_at_5': 0.133}
    ])

    # 3. Pathway Enrichment (Table S3 from g:Profiler output)
    if GO_CSV.exists():
        df_go = pd.read_csv(GO_CSV)
        sig_go = df_go[df_go['significant']].copy()
        if sig_go.empty:
            sig_go = df_go.head(15).copy()
        sig_go['Fold_Enrichment'] = (
            (sig_go['intersection_size'] / sig_go['query_size']) /
            (sig_go['term_size'] / sig_go['background_size'])
        ).round(2)
        s3_rows = []
        for _, r in sig_go.iterrows():
            s3_rows.append({
                'Ontology_Source': r['source'],
                'Term_ID': r['term_id'],
                'Term_Name': r['term_name'],
                'Top1_Count': r['intersection_size'],
                'Query_Size': r['query_size'],
                'Term_Background_Count': r['term_size'],
                'Background_Size': r['background_size'],
                'Fold_Enrichment': r['Fold_Enrichment'],
                'FDR_q_value': f"{r['fdr']:.3e}",
                'Enriched_Genes': str(r['gene_symbols']).replace(" ", ", ")
            })
        df_s3 = pd.DataFrame(s3_rows)
    else:
        log.warning(f"g:Profiler output not found at {GO_CSV}, using fallback.")
        df_s3 = pd.DataFrame()

    wb = Workbook()
    readme_tables = [
        "Supplementary Tables",
        "Manuscript: RegAtlas: A Learning-to-Rank Multi-Omics Framework for Post-GWAS Locus-to-Gene Mapping in Psychiatric Disorders",
        "",
        "This workbook contains three supplementary tables referenced in the manuscript.",
        "",
        "========================================",
        "Sheet: Table S1 - SCZ Rankings",
        "========================================",
        "Complete list of candidate locus-gene pairs across 111 schizophrenia GWAS loci from PGC3 (3,635 pairs),",
        "including within-locus candidate rankings, RegAtlas prioritization scores, nearest-TSS status,",
        "GTEx brain cis-eQTL annotations, and ENCODE-rE2G predicted enhancer-to-gene regulatory links.",
        "",
        "Column descriptions:",
        "  locus_id                        Unique locus identifier",
        "  chrom                           Chromosome containing sentinel variant",
        "  sentinel_rsid                   dbSNP rsID of sentinel GWAS variant",
        "  sentinel_pos                    Genomic coordinate (GRCh38) of sentinel variant",
        "  gwas_pval                       GWAS P-value of sentinel variant",
        "  gwas_beta                       GWAS effect size (beta) of sentinel variant",
        "  regatlas_rank                   Predicted rank within locus (1 = top prioritized gene)",
        "  regatlas_score                  Raw LambdaRank prediction score",
        "  gene_id                         Ensembl Gene ID",
        "  gene_name                       HGNC Gene Symbol",
        "  gene_biotype                    Gene biotype (e.g., protein_coding, lncRNA)",
        "  abs_tss_distance                Absolute linear distance (bp) to TSS",
        "  tss_distance_rank               Rank by TSS proximity within locus",
        "  is_nearest_tss                  Binary: 1 if nearest gene to sentinel",
        "  has_any_brain_eqtl              Binary: 1 if significant brain cis-eQTL exists",
        "  brain_eqtl_max_slope            Max absolute eQTL slope across brain cortex and BA9",
        "  brain_eqtl_max_neg_log10_pval   Max -log10(P) of eQTL association",
        "  has_re2g_link                   Binary: 1 if ENCODE-rE2G enhancer link predicted",
        "  re2g_max_score                  Max rE2G prediction score",
        "  re2g_total_elements             Total enhancer elements linking to gene",
        "  has_both_eqtl_and_re2g          Binary: 1 if both eQTL and rE2G evidence present",
        "",
        "========================================",
        "Sheet: Table S2 - Ablation",
        "========================================",
        "Full 7-way feature-group ablation benchmark and baseline comparison across 1,075 training loci",
        "under chromosome-held-out cross-validation (GroupKFold by chromosome).",
        "",
        "Column descriptions:",
        "  Model_Configuration             Feature modality or baseline method evaluated",
        "  Num_Features                    Number of features in model configuration",
        "  Top1_Accuracy_Pct               % of loci where true causal gene is ranked #1",
        "  Recall_at_3_Pct                 % of loci where true causal gene is in top 3",
        "  Recall_at_5_Pct                 % of loci where true causal gene is in top 5",
        "  MRR                             Mean Reciprocal Rank across held-out loci",
        "  NDCG_at_5                       Normalized Discounted Cumulative Gain at rank 5",
        "",
        "========================================",
        "Sheet: Table S3 - Pathway Enrichment",
        "========================================",
        "Top Gene Ontology (GO) terms evaluated among protein-coding RegAtlas Top-1",
        "prioritized schizophrenia genes (N=94) against the protein-coding candidate background",
        "(N=1,708) via g:Profiler (no terms passed Benjamini-Hochberg FDR < 0.05; all FDR >= 0.22).",
        "",
        "Column descriptions:",
        "  Ontology_Source                 Gene Ontology domain (GO:BP = Biological Process, GO:MF = Molecular Function, GO:CC = Cellular Component)",
        "  Term_ID                         Native Gene Ontology accession ID",
        "  Term_Name                       Gene Ontology term description",
        "  Top1_Count                      Number of Top-1 prioritized genes annotated to this term",
        "  Query_Size                      Total number of evaluated protein-coding Top-1 genes (94)",
        "  Term_Background_Count           Number of background candidate genes annotated to this term",
        "  Background_Size                 Total protein-coding candidate background size (1,708)",
        "  Fold_Enrichment                 Fold enrichment relative to background: (Top1_Count/Query_Size) / (Term_Background_Count/Background_Size)",
        "  FDR_q_value                     Benjamini-Hochberg false discovery rate adjusted P-value",
        "  Enriched_Genes                  RegAtlas Top-1 prioritized genes belonging to this functional term",
    ]

    write_readme_sheet(wb, readme_tables)
    write_data_sheet(wb, 'Table S1 - SCZ Rankings', df_s1)
    write_data_sheet(wb, 'Table S2 - Ablation', df_s2)
    write_data_sheet(wb, 'Table S3 - Pathway Enrichment', df_s3)

    out_xlsx = RESULTS_DIR / "tables" / "Supplementary_Tables.xlsx"
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_xlsx)
    log.info(f"Saved {out_xlsx}")


def build_supplementary_data():
    log.info("Compiling Supplementary_Data.xlsx...")
    df_train = pd.read_parquet(DATA_PROCESSED / "training_matrix_dataset_a.parquet")
    df_scz = pd.read_parquet(DATA_PROCESSED / "scz_prioritized_gene_rankings.parquet")

    wb = Workbook()
    readme_data = [
        "Supplementary Data",
        "Manuscript: RegAtlas: A Learning-to-Rank Multi-Omics Framework for Post-GWAS Locus-to-Gene Mapping in Psychiatric Disorders",
        "",
        "This workbook contains the complete processed, model-ready feature matrices and ranking outputs.",
        "",
        "========================================",
        "Sheet: Data S1 - Training Universe",
        "========================================",
        "Processed multi-omics feature matrix and gold-standard supervision labels for 1,075 Open Targets GWAS loci",
        "(35,358 candidate gene pairs), representing Dataset A.",
        "",
        "Column descriptions:",
        "  locus_id                        Unique locus identifier",
        "  chrom                           Chromosome containing sentinel variant",
        "  sentinel_pos                    Genomic position (GRCh38) of sentinel variant",
        "  confidence                      Confidence tier from Open Targets L2G gold standard",
        "  label_set                       Evidence source (ot_platform: clinical/Mendelian, otg_original: fine-mapped coding)",
        "  gene_id                         Ensembl Gene ID",
        "  gene_name                       HGNC Gene Symbol",
        "  gene_biotype                    Ensembl biotype (e.g., protein_coding, lncRNA)",
        "  label                           Binary truth label (1 = true causal target, 0 = competitor gene)",
        "  abs_tss_distance                Absolute linear distance (bp) to gene TSS",
        "  tss_distance_rank               Within-locus rank by TSS distance",
        "  is_nearest_tss                  Binary: 1 if nearest gene to sentinel",
        "  has_any_brain_eqtl              Binary: 1 if significant brain cis-eQTL exists",
        "  brain_eqtl_max_slope            Max absolute eQTL slope across brain tissues",
        "  brain_eqtl_max_neg_log10_pval   Max -log10(P) of eQTL association",
        "  has_re2g_link                   Binary: 1 if ENCODE-rE2G enhancer link predicted",
        "  re2g_max_score                  Max rE2G prediction score",
        "  re2g_total_elements             Total enhancer elements linking to gene",
        "  has_both_eqtl_and_re2g          Binary: 1 if both eQTL and rE2G evidence present",
        "",
        "========================================",
        "Sheet: Data S2 - SCZ Application",
        "========================================",
        "Complete schizophrenia application multi-omics feature matrix and frozen RegAtlas ranking outputs",
        "across 111 PGC3 schizophrenia GWAS loci (3,635 candidate gene pairs), representing Dataset B.",
        "",
        "Column descriptions:",
        "  regatlas_rank                   Final predicted rank of candidate gene within locus (1 = top prioritized gene)",
        "  regatlas_score                  Raw LambdaRank model score",
        "  (All feature and identifier columns follow identical definitions to Data S1)",
    ]

    write_readme_sheet(wb, readme_data)
    write_data_sheet(wb, 'Data S1 - Training Universe', df_train)
    write_data_sheet(wb, 'Data S2 - SCZ Application', df_scz)

    out_xlsx = RESULTS_DIR / "tables" / "Supplementary_Data.xlsx"
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_xlsx)
    log.info(f"Saved {out_xlsx}")


def main():
    build_supplementary_tables()
    build_supplementary_data()
    print("\n" + "="*70)
    print("SUPPLEMENTARY WORKBOOKS COMPILED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
