#!/usr/bin/env python
"""
Phase 6 / Task 1 & 2: Build Clean Schizophrenia (SCZ) Application Matrix (Dataset B).

Pipeline:
1. Identifies genome-wide significant SCZ GWAS lead variants (P < 5e-8) from the original SCZ data.
2. Clumps variants into independent genomic loci (minimum 500 kb separation).
3. Defines a +/- 500 kb candidate window around each sentinel variant.
4. Maps all candidate genes from the Ensembl GRCh38 reference.
5. Attaches identical regulatory features (Distance, GTEx v10 Brain eQTLs, ENCODE-rE2G Enhancers).
6. Strictly excludes all circular RNA-seq features (log2FoldChange, padj, lfcSE, baseMean).
7. Exports clean Dataset B for frozen model inference.
"""

import gzip
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CURRENT = PROJECT_ROOT / "data" / "current"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "downstream_biology"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE_BP = 500_000


def load_gtex_and_re2g_lookups():
    """Load GTEx v10 Brain eQTLs and ENCODE-rE2G predictions for candidate gene feature mapping."""
    log.info("Loading GTEx Brain eQTL and ENCODE-rE2G lookups...")
    
    # 1. GTEx Cortex
    cortex_path = DATA_RAW / "gtex" / "GTEx_Analysis_v10_eQTL_updated" / "Brain_Cortex.v10.eQTLs.signif_pairs.parquet"
    df_cortex = pd.read_parquet(cortex_path, columns=['gene_id', 'pval_nominal', 'slope'])
    df_cortex['gene_id_clean'] = df_cortex['gene_id'].str.split('.').str[0]
    df_cortex['abs_slope'] = df_cortex['slope'].abs()
    cortex_agg = df_cortex.groupby('gene_id_clean').agg(
        cortex_min_pval=('pval_nominal', 'min'),
        cortex_max_slope=('abs_slope', 'max')
    ).to_dict(orient='index')
    del df_cortex

    # 2. GTEx BA9
    ba9_path = DATA_RAW / "gtex" / "GTEx_Analysis_v10_eQTL_updated" / "Brain_Frontal_Cortex_BA9.v10.eQTLs.signif_pairs.parquet"
    df_ba9 = pd.read_parquet(ba9_path, columns=['gene_id', 'pval_nominal', 'slope'])
    df_ba9['gene_id_clean'] = df_ba9['gene_id'].str.split('.').str[0]
    df_ba9['abs_slope'] = df_ba9['slope'].abs()
    ba9_agg = df_ba9.groupby('gene_id_clean').agg(
        ba9_min_pval=('pval_nominal', 'min'),
        ba9_max_slope=('abs_slope', 'max')
    ).to_dict(orient='index')
    del df_ba9

    # 3. ENCODE-rE2G
    re2g_stats = defaultdict(lambda: {
        'dlpfc_max_score': 0.0,
        'dlpfc_elements': 0,
        'brain_max_score': 0.0,
        'brain_elements': 0,
        'is_promoter_overlap': 0
    })
    
    re2g_dir = DATA_RAW / "encode_re2g"
    files = {
        'dlpfc1': re2g_dir / "DLPFC_ENCFF371VKL.bed.gz",
        'dlpfc2': re2g_dir / "DLPFC_ENCFF280TEO.bed.gz",
        'brain': re2g_dir / "Brain_ENCFF307BFL.bed.gz"
    }
    
    for tag, fpath in files.items():
        if not fpath.exists():
            continue
        with gzip.open(fpath, 'rt', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            gene_col = header.index('TargetGeneEnsemblID') if 'TargetGeneEnsemblID' in header else 6
            score_col = header.index('Score') if 'Score' in header else len(header) - 1
            prom_col = header.index('isSelfPromoter') if 'isSelfPromoter' in header else -1
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) <= max(gene_col, score_col):
                    continue
                gene_id = parts[gene_col].split('.')[0]
                try:
                    score = float(parts[score_col])
                except ValueError:
                    score = 0.0
                is_prom = 1 if (prom_col != -1 and parts[prom_col].upper() == 'TRUE') else 0
                
                if 'dlpfc' in tag:
                    re2g_stats[gene_id]['dlpfc_max_score'] = max(re2g_stats[gene_id]['dlpfc_max_score'], score)
                    re2g_stats[gene_id]['dlpfc_elements'] += 1
                else:
                    re2g_stats[gene_id]['brain_max_score'] = max(re2g_stats[gene_id]['brain_max_score'], score)
                    re2g_stats[gene_id]['brain_elements'] += 1
                if is_prom:
                    re2g_stats[gene_id]['is_promoter_overlap'] = 1
                    
    log.info("Lookups successfully loaded.")
    return cortex_agg, ba9_agg, dict(re2g_stats)


def identify_scz_loci():
    """Extract and clump genome-wide significant SCZ GWAS lead variants."""
    csv_path = DATA_CURRENT / "rna_seq_eQTLs_gwas_schizophrenia.csv"
    log.info(f"Reading SCZ GWAS variants from {csv_path}...")
    
    # Load GWAS columns
    df_raw = pd.read_csv(
        csv_path,
        usecols=['CHROM', 'ID', 'POS', 'A1', 'A2', 'BETA', 'SE', 'PVAL']
    ).drop_duplicates(subset=['ID'])
    
    log.info(f"Loaded {len(df_raw):,} unique variants.")
    
    # Filter for genome-wide significance (P < 5e-8)
    df_sig = df_raw[df_raw['PVAL'] < 5e-8].copy()
    log.info(f"Found {len(df_sig):,} genome-wide significant variants (P < 5e-8).")
    
    # Sort by chromosome and position
    df_sig['CHROM'] = df_sig['CHROM'].astype(str)
    df_sig['POS'] = df_sig['POS'].astype(int)
    df_sig = df_sig.sort_values(by=['CHROM', 'PVAL']).reset_index(drop=True)
    
    # Clump into independent loci (lead SNP within 500 kb window)
    clumped_loci = []
    
    for chrom, group in df_sig.groupby('CHROM'):
        occupied_windows = []
        for _, row in group.iterrows():
            pos = row['POS']
            # Check overlap with existing loci on this chromosome
            overlap = False
            for w_start, w_end in occupied_windows:
                if not (pos + WINDOW_SIZE_BP < w_start or pos - WINDOW_SIZE_BP > w_end):
                    overlap = True
                    break
            if not overlap:
                occupied_windows.append((pos - WINDOW_SIZE_BP, pos + WINDOW_SIZE_BP))
                clumped_loci.append({
                    'locus_id': f"SCZ_LOC_{chrom}_{pos}_{row['ID']}",
                    'chrom': chrom,
                    'pos': pos,
                    'lead_snp_id': row['ID'],
                    'a1': row['A1'],
                    'a2': row['A2'],
                    'gwas_beta': row['BETA'],
                    'gwas_se': row['SE'],
                    'gwas_pval': row['PVAL']
                })
                
    df_loci = pd.DataFrame(clumped_loci)
    log.info(f"Identified {len(df_loci):,} independent genome-wide significant SCZ GWAS loci.")
    return df_loci


def build_scz_application_matrix(df_loci, cortex_agg, ba9_agg, re2g_stats):
    """Build candidate gene universe and attach multi-omics features for SCZ loci."""
    log.info("Constructing SCZ candidate universe (+/- 500 kb) and extracting features...")
    
    # Load Ensembl genes catalog
    genes_path = DATA_PROCESSED / "ensembl_genes_grch38.parquet"
    df_genes = pd.read_parquet(genes_path)
    
    genes_by_chrom = defaultdict(list)
    for _, row in df_genes.iterrows():
        genes_by_chrom[str(row['chrom'])].append(row)
    for c in genes_by_chrom:
        genes_by_chrom[c] = pd.DataFrame(genes_by_chrom[c])
        
    candidate_rows = []
    
    for _, loc in df_loci.iterrows():
        chrom = str(loc['chrom'])
        pos = loc['pos']
        locus_id = loc['locus_id']
        lead_snp = loc['lead_snp_id']
        
        if chrom not in genes_by_chrom:
            continue
            
        c_df = genes_by_chrom[chrom]
        win_start = pos - WINDOW_SIZE_BP
        win_end = pos + WINDOW_SIZE_BP
        
        # Overlapping genes
        candidates = c_df[
            (c_df['end'] >= win_start) & (c_df['start'] <= win_end)
        ].copy()
        
        if candidates.empty:
            continue
            
        candidates['locus_id'] = locus_id
        candidates['sentinel_rsid'] = lead_snp
        candidates['sentinel_pos'] = pos
        candidates['gwas_pval'] = loc['gwas_pval']
        candidates['gwas_beta'] = loc['gwas_beta']
        
        # 1. Distance features
        candidates['tss_distance'] = candidates['tss'] - pos
        candidates['abs_tss_distance'] = candidates['tss_distance'].abs()
        candidates['log10_tss_distance'] = np.log10(candidates['abs_tss_distance'] + 1.0)
        
        candidates['gene_body_distance'] = 0.0
        up = candidates['start'] > pos
        down = candidates['end'] < pos
        candidates.loc[up, 'gene_body_distance'] = (candidates.loc[up, 'start'] - pos).astype(float)
        candidates.loc[down, 'gene_body_distance'] = (pos - candidates.loc[down, 'end']).astype(float)
        candidates['log10_gene_body_distance'] = np.log10(candidates['gene_body_distance'] + 1.0)
        
        candidates['tss_distance_rank'] = candidates['abs_tss_distance'].rank(method='min').astype(int)
        candidates['is_nearest_tss'] = (candidates['tss_distance_rank'] == 1).astype(int)
        
        # Clean ID
        candidates['gene_id_clean'] = candidates['gene_id'].str.split('.').str[0]
        
        # 2. GTEx Brain eQTL features
        cortex_pval = []
        cortex_slope = []
        has_cortex = []
        ba9_pval = []
        ba9_slope = []
        has_ba9 = []
        
        for gid in candidates['gene_id_clean']:
            if gid in cortex_agg:
                cortex_pval.append(cortex_agg[gid]['cortex_min_pval'])
                cortex_slope.append(cortex_agg[gid]['cortex_max_slope'])
                has_cortex.append(1)
            else:
                cortex_pval.append(1.0)
                cortex_slope.append(0.0)
                has_cortex.append(0)
                
            if gid in ba9_agg:
                ba9_pval.append(ba9_agg[gid]['ba9_min_pval'])
                ba9_slope.append(ba9_agg[gid]['ba9_max_slope'])
                has_ba9.append(1)
            else:
                ba9_pval.append(1.0)
                ba9_slope.append(0.0)
                has_ba9.append(0)
                
        candidates['cortex_min_pval'] = cortex_pval
        candidates['cortex_neg_log10_pval'] = -np.log10(np.clip(candidates['cortex_min_pval'], 1e-300, 1.0))
        candidates['cortex_max_abs_slope'] = cortex_slope
        candidates['has_cortex_eqtl'] = has_cortex
        
        candidates['ba9_min_pval'] = ba9_pval
        candidates['ba9_neg_log10_pval'] = -np.log10(np.clip(candidates['ba9_min_pval'], 1e-300, 1.0))
        candidates['ba9_max_abs_slope'] = ba9_slope
        candidates['has_ba9_eqtl'] = has_ba9
        
        candidates['brain_eqtl_tissue_count'] = candidates['has_cortex_eqtl'] + candidates['has_ba9_eqtl']
        candidates['has_any_brain_eqtl'] = (candidates['brain_eqtl_tissue_count'] > 0).astype(int)
        candidates['brain_eqtl_max_slope'] = np.maximum(candidates['cortex_max_abs_slope'], candidates['ba9_max_abs_slope'])
        candidates['brain_eqtl_max_neg_log10_pval'] = np.maximum(candidates['cortex_neg_log10_pval'], candidates['ba9_neg_log10_pval'])
        
        # 3. ENCODE-rE2G features
        re2g_dlpfc = []
        re2g_brain = []
        re2g_elem_count = []
        re2g_promoter = []
        has_re2g_list = []
        
        for gid in candidates['gene_id_clean']:
            if gid in re2g_stats:
                st = re2g_stats[gid]
                re2g_dlpfc.append(st['dlpfc_max_score'])
                re2g_brain.append(st['brain_max_score'])
                re2g_elem_count.append(st['dlpfc_elements'] + st['brain_elements'])
                re2g_promoter.append(st['is_promoter_overlap'])
                has_re2g_list.append(1 if (st['dlpfc_max_score'] > 0 or st['brain_max_score'] > 0) else 0)
            else:
                re2g_dlpfc.append(0.0)
                re2g_brain.append(0.0)
                re2g_elem_count.append(0)
                re2g_promoter.append(0)
                has_re2g_list.append(0)
                
        candidates['re2g_dlpfc_score'] = re2g_dlpfc
        candidates['re2g_brain_score'] = re2g_brain
        candidates['re2g_max_score'] = np.maximum(candidates['re2g_dlpfc_score'], candidates['re2g_brain_score'])
        candidates['re2g_total_elements'] = re2g_elem_count
        candidates['re2g_is_promoter'] = re2g_promoter
        candidates['has_re2g_link'] = has_re2g_list
        
        candidates['has_both_eqtl_and_re2g'] = ((candidates['has_any_brain_eqtl'] == 1) & (candidates['has_re2g_link'] == 1)).astype(int)
        
        candidate_rows.append(candidates)
        
    df_scz_matrix = pd.concat(candidate_rows, ignore_index=True)
    
    out_path = DATA_PROCESSED / "scz_application_matrix_dataset_b.parquet"
    df_scz_matrix.to_parquet(out_path, index=False)
    log.info(f"Saved SCZ Application Matrix (Dataset B) with {len(df_scz_matrix):,} candidate pairs across {df_scz_matrix['locus_id'].nunique():,} loci to {out_path}.")
    
    return df_scz_matrix


def main():
    log.info("Starting Phase 6: Build SCZ Application Matrix (Dataset B)...")
    
    # 1. Load Lookups
    cortex_agg, ba9_agg, re2g_stats = load_gtex_and_re2g_lookups()
    
    # 2. Extract and Clump SCZ Loci
    df_loci = identify_scz_loci()
    
    # 3. Build Application Matrix
    df_scz_matrix = build_scz_application_matrix(df_loci, cortex_agg, ba9_agg, re2g_stats)
    
    print("\n" + "="*80)
    print("PHASE 6: SCZ APPLICATION MATRIX (DATASET B) BUILT")
    print("="*80)
    print(f"Independent SCZ Loci (P < 5e-8): {df_scz_matrix['locus_id'].nunique():,}")
    print(f"Total Candidate Locus-Gene Pairs:  {len(df_scz_matrix):,}")
    print(f"Avg Candidates per Locus:         {len(df_scz_matrix)/df_scz_matrix['locus_id'].nunique():.1f}")
    print(f"Output File:                      {DATA_PROCESSED / 'scz_application_matrix_dataset_b.parquet'}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
