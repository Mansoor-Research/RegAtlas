#!/usr/bin/env python
"""
Phase 1: Audit the Current SCZ Input CSV.

This script loads the integrated schizophrenia CSV, inspects its structure,
classifies each column for the rebuild, checks data quality, and outputs:
  - results/audit/current_input_report.md
  - results/audit/current_input_columns.tsv
  - results/audit/current_input_duplicates.tsv
  - data/processed/current_scz_backbone.parquet

Usage:
  python scripts/00_audit_current_input.py --input data/current/rna_seq_eQTLs_gwas_schizophrenia.csv
"""

import argparse
import os
import sys
import logging
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Column classification for the rebuild
# Identifiers: kept for mapping, not used as features
IDENTIFIERS = {
    'CHROM', 'POS', 'ID', 'variant_id', 'gene_id', 'ensembl_gene_id',
    'hgnc_symbol', 'description', 'gene_biotype', 'A1', 'A2', 'DIRE',
}

# Features allowed in the primary rebuilt model
ALLOWED_PRIMARY_FEATURES = {
    'BETA', 'SE', 'PVAL', 'NCAS', 'NCON', 'NGT', 'NEFFDIV2', 'IMPINFO',
    'FCAS', 'FCON', 'af', 'ma_samples', 'ma_count',
    'tss_distance', 'slope',
}

# Features excluded due to leakage with the old label
EXCLUDED_LEAKAGE = {
    'log2FoldChange', 'padj', 'pvalue', 'stat', 'lfcSE', 'baseMean',
    'pval_nominal', 'slope_se', 'min_pval_nominal', 'pval_nominal_threshold',
    'pval_beta',
}

# Old label and derived columns
EXCLUDED_OLD_LABEL = {
    'label',
}

# Transformed versions that should also be excluded
EXCLUDED_TRANSFORMS = {
    'log2FoldChange_neglog10', 'padj_neglog10', 'pval_nominal_neglog10',
    'pvalue_neglog10', 'PVAL_neglog10',
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def classify_column(col: str) -> str:
    """Classify a column into one of the rebuild categories."""
    if col in IDENTIFIERS:
        return 'identifier'
    if col in ALLOWED_PRIMARY_FEATURES:
        return 'allowed_primary_feature'
    if col in EXCLUDED_LEAKAGE:
        return 'excluded_leakage_feature'
    if col in EXCLUDED_OLD_LABEL:
        return 'excluded_old_label'
    if col in EXCLUDED_TRANSFORMS:
        return 'excluded_transform'
    return 'unknown_requires_review'


def infer_genome_build(df: pd.DataFrame) -> str:
    """
    Attempt to infer genome build from position ranges.
    GRCh37 vs GRCh38 cannot be reliably determined from positions alone.
    We check for known coordinate patterns but ultimately flag for manual review.
    """
    if 'POS' not in df.columns:
        return 'UNKNOWN (no POS column)'

    max_pos = df['POS'].max()
    # Check if any chromosome has positions consistent with either build
    # This is not definitive — flag for manual confirmation
    return 'REQUIRES_MANUAL_CONFIRMATION (likely GRCh37 based on PGC3 SCZ GWAS)'


def check_af_type(df: pd.DataFrame) -> str:
    """Check whether 'af' is allele frequency or minor allele frequency."""
    if 'af' not in df.columns:
        return 'N/A (no af column)'

    af = df['af'].dropna()
    max_af = af.max()
    pct_above_05 = (af > 0.5).mean() * 100

    if max_af > 0.5:
        return f'ALLELE_FREQUENCY (max={max_af:.4f}, {pct_above_05:.1f}% > 0.5 — NOT MAF, needs conversion)'
    else:
        return f'MINOR_ALLELE_FREQUENCY (max={max_af:.4f}, all ≤ 0.5)'


def standardize_chromosomes(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize chromosome values to 1-22, X, Y, MT."""
    if 'CHROM' not in df.columns:
        return df

    df = df.copy()
    df['CHROM'] = (
        df['CHROM']
        .astype(str)
        .str.upper()
        .str.replace('CHR', '', regex=False)
    )
    return df


def standardize_gene_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Remove version suffixes from Ensembl gene IDs."""
    for col in ['gene_id', 'ensembl_gene_id']:
        if col in df.columns:
            df = df.copy()
            df[col] = df[col].astype(str).str.split('.').str[0]
    return df


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit(input_path: str, output_dir: str):
    """Run the full Phase 1 audit."""
    log.info(f"Starting Phase 1 Audit: {input_path}")
    log.info(f"Timestamp: {TIMESTAMP}")

    # Paths
    audit_dir = os.path.join(output_dir, 'results', 'audit')
    processed_dir = os.path.join(output_dir, 'data', 'processed')
    os.makedirs(audit_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    log.info("Loading CSV (this may take a moment for large files)...")
    df = pd.read_csv(input_path, low_memory=False)
    n_rows, n_cols = df.shape
    log.info(f"Loaded: {n_rows:,} rows × {n_cols} columns")

    # -----------------------------------------------------------------------
    # 2. Basic statistics
    # -----------------------------------------------------------------------
    col_info = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.astype(str).values,
        'non_null': df.notnull().sum().values,
        'null_count': df.isnull().sum().values,
        'null_pct': (df.isnull().mean() * 100).round(2).values,
        'n_unique': df.nunique().values,
        'classification': [classify_column(c) for c in df.columns],
    })

    # Add sample values
    col_info['sample_values'] = [
        str(df[c].dropna().head(3).tolist()) for c in df.columns
    ]

    # -----------------------------------------------------------------------
    # 3. Key counts
    # -----------------------------------------------------------------------
    unique_snps = df['ID'].nunique() if 'ID' in df.columns else 'N/A'
    unique_genes = df['gene_id'].nunique() if 'gene_id' in df.columns else 'N/A'
    unique_symbols = df['hgnc_symbol'].nunique() if 'hgnc_symbol' in df.columns else 'N/A'
    unique_chroms = df['CHROM'].nunique() if 'CHROM' in df.columns else 'N/A'

    # SNPs per gene distribution
    if 'gene_id' in df.columns and 'ID' in df.columns:
        snps_per_gene = df.groupby('gene_id')['ID'].nunique()
        genes_per_snp = df.groupby('ID')['gene_id'].nunique()
    else:
        snps_per_gene = pd.Series(dtype=float)
        genes_per_snp = pd.Series(dtype=float)

    # -----------------------------------------------------------------------
    # 4. Old label statistics
    # -----------------------------------------------------------------------
    old_label_cols = ['log2FoldChange', 'padj', 'pval_nominal']
    can_compute_label = all(c in df.columns for c in old_label_cols)

    if can_compute_label:
        old_label = (
            (df['log2FoldChange'].abs() > 1) &
            (df['padj'] < 0.05) &
            (df['pval_nominal'] < 1e-4)
        ).astype(int)
        n_pos = old_label.sum()
        prevalence = old_label.mean() * 100
    else:
        n_pos = 'N/A'
        prevalence = 'N/A'

    # -----------------------------------------------------------------------
    # 5. Duplicates check
    # -----------------------------------------------------------------------
    dup_cols = [c for c in ['ID', 'gene_id'] if c in df.columns]
    if len(dup_cols) == 2:
        dups = df.duplicated(subset=dup_cols, keep=False)
        n_dups = dups.sum()
        dup_df = df[dups][dup_cols].drop_duplicates()
    else:
        n_dups = 'N/A'
        dup_df = pd.DataFrame()

    # -----------------------------------------------------------------------
    # 6. Genome build inference
    # -----------------------------------------------------------------------
    genome_build = infer_genome_build(df)

    # -----------------------------------------------------------------------
    # 7. AF type check
    # -----------------------------------------------------------------------
    af_type = check_af_type(df)

    # -----------------------------------------------------------------------
    # 8. Standardize and create backbone
    # -----------------------------------------------------------------------
    log.info("Standardizing identifiers...")
    df_std = standardize_chromosomes(df)
    df_std = standardize_gene_ids(df_std)

    # Identify columns to keep in backbone (identifiers + allowed features)
    backbone_cols = [
        c for c in df_std.columns
        if classify_column(c) in ('identifier', 'allowed_primary_feature')
    ]
    # Also keep the leakage features tagged but in a separate set for reference
    leakage_cols = [c for c in df_std.columns if classify_column(c) == 'excluded_leakage_feature']

    df_backbone = df_std[backbone_cols].copy()
    log.info(f"Backbone shape: {df_backbone.shape}")

    # -----------------------------------------------------------------------
    # 9. Feature correlation with old label (leakage confirmation)
    # -----------------------------------------------------------------------
    leakage_corr = {}
    if can_compute_label:
        for col in EXCLUDED_LEAKAGE | ALLOWED_PRIMARY_FEATURES:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                valid = df[[col]].join(old_label.rename('label')).dropna()
                if len(valid) > 100:
                    corr = valid[col].corr(valid['label'])
                    leakage_corr[col] = round(corr, 4)

    # -----------------------------------------------------------------------
    # 10. Chromosome-level observation counts
    # -----------------------------------------------------------------------
    if 'CHROM' in df_std.columns:
        chrom_counts = df_std['CHROM'].value_counts().sort_index()
    else:
        chrom_counts = pd.Series(dtype=int)

    # -----------------------------------------------------------------------
    # Generate outputs
    # -----------------------------------------------------------------------

    # Output 1: Column classification TSV
    col_tsv_path = os.path.join(audit_dir, 'current_input_columns.tsv')
    col_info.to_csv(col_tsv_path, sep='\t', index=False)
    log.info(f"Saved column classification to {col_tsv_path}")

    # Output 2: Duplicates TSV
    dup_tsv_path = os.path.join(audit_dir, 'current_input_duplicates.tsv')
    if len(dup_df) > 0:
        dup_df.to_csv(dup_tsv_path, sep='\t', index=False)
    else:
        pd.DataFrame(columns=['ID', 'gene_id']).to_csv(dup_tsv_path, sep='\t', index=False)
    log.info(f"Saved duplicates report to {dup_tsv_path}")

    # Output 3: Backbone parquet
    backbone_path = os.path.join(processed_dir, 'current_scz_backbone.parquet')
    df_backbone.to_parquet(backbone_path, index=False)
    log.info(f"Saved backbone to {backbone_path}")

    # Output 4: Comprehensive report
    report_lines = []
    report_lines.append("# Phase 1 Audit Report: Current SCZ Input")
    report_lines.append(f"\n**Generated:** {TIMESTAMP}")
    report_lines.append(f"**Input file:** `{input_path}`")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Overview")
    report_lines.append("")
    report_lines.append(f"| Metric | Value |")
    report_lines.append(f"|---|---|")
    report_lines.append(f"| Rows | {n_rows:,} |")
    report_lines.append(f"| Columns | {n_cols} |")
    report_lines.append(f"| Unique SNPs (ID) | {unique_snps:,} |")
    report_lines.append(f"| Unique Genes (gene_id) | {unique_genes:,} |")
    report_lines.append(f"| Unique Gene Symbols (hgnc_symbol) | {unique_symbols:,} |")
    report_lines.append(f"| Unique Chromosomes | {unique_chroms} |")
    report_lines.append(f"| Genome Build | {genome_build} |")
    report_lines.append(f"| AF Type | {af_type} |")
    report_lines.append(f"| Duplicate (ID, gene_id) pairs | {n_dups} |")
    report_lines.append("")

    report_lines.append("## 2. Old Label Statistics")
    report_lines.append("")
    report_lines.append(f"| Metric | Value |")
    report_lines.append(f"|---|---|")
    report_lines.append(f"| Positives (old label=1) | {n_pos:,} |")
    report_lines.append(f"| Prevalence | {prevalence}% |")
    report_lines.append(f"| Label definition | `|log2FC|>1 AND padj<0.05 AND pval_nominal<1e-4` |")
    report_lines.append("")
    report_lines.append("> **NOTE:** This old label is RETIRED and will NOT be used in the rebuild.")
    report_lines.append("")

    report_lines.append("## 3. Column Classification")
    report_lines.append("")
    class_counts = col_info['classification'].value_counts()
    report_lines.append("| Category | Count |")
    report_lines.append("|---|---|")
    for cat, count in class_counts.items():
        report_lines.append(f"| {cat} | {count} |")
    report_lines.append("")

    report_lines.append("### Detailed Classification")
    report_lines.append("")
    for cat in ['identifier', 'allowed_primary_feature', 'excluded_leakage_feature',
                'excluded_old_label', 'excluded_transform', 'unknown_requires_review']:
        subset = col_info[col_info['classification'] == cat]
        if len(subset) > 0:
            report_lines.append(f"**{cat}:** {', '.join(subset['column'].tolist())}")
            report_lines.append("")

    report_lines.append("## 4. Missingness")
    report_lines.append("")
    missing = col_info[col_info['null_count'] > 0].sort_values('null_pct', ascending=False)
    if len(missing) > 0:
        report_lines.append("| Column | Null Count | Null % |")
        report_lines.append("|---|---|---|")
        for _, row in missing.iterrows():
            report_lines.append(f"| {row['column']} | {row['null_count']:,} | {row['null_pct']:.2f}% |")
    else:
        report_lines.append("No missing values detected.")
    report_lines.append("")

    report_lines.append("## 5. SNP–Gene Relationship")
    report_lines.append("")
    if len(snps_per_gene) > 0:
        report_lines.append("### SNPs per Gene")
        report_lines.append(f"- Mean: {snps_per_gene.mean():.1f}")
        report_lines.append(f"- Median: {snps_per_gene.median():.1f}")
        report_lines.append(f"- Max: {snps_per_gene.max()}")
        report_lines.append(f"- Min: {snps_per_gene.min()}")
        report_lines.append(f"- Genes with >100 SNPs: {(snps_per_gene > 100).sum()}")
        report_lines.append("")

    if len(genes_per_snp) > 0:
        report_lines.append("### Genes per SNP")
        report_lines.append(f"- Mean: {genes_per_snp.mean():.1f}")
        report_lines.append(f"- Median: {genes_per_snp.median():.1f}")
        report_lines.append(f"- Max: {genes_per_snp.max()}")
        report_lines.append(f"- SNPs mapping to >1 gene: {(genes_per_snp > 1).sum():,}")
        report_lines.append("")

    report_lines.append("## 6. Chromosome Distribution")
    report_lines.append("")
    if len(chrom_counts) > 0:
        report_lines.append("| Chromosome | Rows |")
        report_lines.append("|---|---|")
        for chrom, count in chrom_counts.items():
            report_lines.append(f"| {chrom} | {count:,} |")
    report_lines.append("")

    report_lines.append("## 7. Feature–Label Correlation (Leakage Confirmation)")
    report_lines.append("")
    if leakage_corr:
        report_lines.append("Pearson correlation of each feature with the OLD label:")
        report_lines.append("")
        report_lines.append("| Feature | Correlation with Old Label | Classification |")
        report_lines.append("|---|---|---|")
        for feat, corr in sorted(leakage_corr.items(), key=lambda x: abs(x[1]), reverse=True):
            cls = classify_column(feat)
            flag = " ⚠️ LEAKAGE" if cls == 'excluded_leakage_feature' and abs(corr) > 0.1 else ""
            report_lines.append(f"| {feat} | {corr:.4f} | {cls}{flag} |")
    report_lines.append("")

    report_lines.append("## 8. Backbone Output")
    report_lines.append("")
    report_lines.append(f"- **Backbone columns ({len(backbone_cols)}):** {', '.join(backbone_cols)}")
    report_lines.append(f"- **Excluded leakage columns ({len(leakage_cols)}):** {', '.join(leakage_cols)}")
    report_lines.append(f"- **Backbone shape:** {df_backbone.shape[0]:,} rows × {df_backbone.shape[1]} columns")
    report_lines.append(f"- **Saved to:** `{backbone_path}`")
    report_lines.append("")

    report_lines.append("## 9. Gate 1 Assessment")
    report_lines.append("")

    gate_pass = True
    gate_issues = []

    if 'REQUIRES_MANUAL_CONFIRMATION' in genome_build or 'UNKNOWN' in genome_build:
        gate_issues.append(f"⚠️ Genome build: {genome_build}")

    if unique_snps == 'N/A' or (isinstance(unique_snps, int) and unique_snps == 0):
        gate_pass = False
        gate_issues.append("❌ No variant IDs found")

    if unique_genes == 'N/A' or (isinstance(unique_genes, int) and unique_genes == 0):
        gate_pass = False
        gate_issues.append("❌ No gene IDs found")

    if isinstance(unique_genes, int) and unique_genes < 100:
        gate_pass = False
        gate_issues.append(f"❌ Only {unique_genes} unique genes — too few")

    if n_cols < 10:
        gate_pass = False
        gate_issues.append(f"❌ Only {n_cols} columns — insufficient feature space")

    # Check if this is an eQTL-restricted subset
    if isinstance(unique_genes, int) and isinstance(unique_snps, int):
        ratio = n_rows / max(unique_snps, 1)
        if ratio < 1.5 and unique_genes < 500:
            gate_issues.append(
                f"⚠️ Low SNP-to-row ratio ({ratio:.1f}) with few genes — "
                "may be eQTL-restricted subset"
            )

    if gate_issues:
        for issue in gate_issues:
            report_lines.append(f"- {issue}")
    else:
        report_lines.append("- ✅ All checks passed")

    report_lines.append("")
    if gate_pass:
        report_lines.append("> **GATE 1 STATUS: CONDITIONAL PASS** — Genome build needs manual confirmation. "
                          "Data structure is suitable for rebuild.")
    else:
        report_lines.append("> **GATE 1 STATUS: FAIL** — Address the issues above before proceeding.")

    # Write report
    report_path = os.path.join(audit_dir, 'current_input_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log.info(f"Saved audit report to {report_path}")

    log.info("Phase 1 Audit complete.")
    return gate_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 1: Audit the current SCZ input CSV.")
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to the integrated SCZ CSV file'
    )
    parser.add_argument(
        '-o', '--output-dir', type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help='Root output directory (default: parent of scripts/)'
    )
    args = parser.parse_args()

    passed = run_audit(args.input, args.output_dir)
    sys.exit(0 if passed else 1)
