"""Rank genes at user-supplied GRCh38 GWAS sentinel loci with RegAtlas."""
from pathlib import Path
import argparse
import hashlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def prepare(loci, genes):
    required = {'locus_id', 'chrom', 'sentinel_pos'}
    if not required.issubset(loci.columns):
        raise ValueError('Input CSV requires locus_id, chrom and sentinel_pos.')
    if loci[list(required)].isna().any().any() or loci.locus_id.duplicated().any():
        raise ValueError('Locus keys must be complete and unique.')
    loci = loci.copy()
    loci['chrom'] = loci.chrom.astype(str).str.removeprefix('chr').replace({'23': 'X'})
    positions = pd.to_numeric(loci.sentinel_pos, errors='raise')
    if not ((positions > 0) & (positions == np.floor(positions))).all():
        raise ValueError('Sentinel positions must be positive 1-based integers.')
    loci['sentinel_pos'] = positions.astype('int64')
    candidates, coverage = [], []
    for locus in loci.itertuples(index=False):
        selected = genes.loc[genes.chrom.eq(locus.chrom) & genes.start.le(locus.sentinel_pos + 500000)
                             & genes.end.ge(max(1, locus.sentinel_pos - 500000))].copy()
        selected['locus_id'] = locus.locus_id
        selected['sentinel_pos'] = locus.sentinel_pos
        coverage.append({'locus_id': locus.locus_id, 'candidate_count': len(selected)})
        candidates.append(selected)
    matrix = pd.concat(candidates, ignore_index=True)
    if matrix.empty:
        raise ValueError('No eligible genes overlap the supplied windows.')
    matrix['tss_distance'] = matrix.tss - matrix.sentinel_pos
    matrix['abs_tss_distance'] = matrix.tss_distance.abs()
    matrix['gene_body_distance'] = np.maximum.reduce([matrix.start - matrix.sentinel_pos,
                                                     np.zeros(len(matrix)), matrix.sentinel_pos - matrix.end])
    matrix['log10_tss_distance'] = np.log10(matrix.abs_tss_distance + 1)
    matrix['log10_gene_body_distance'] = np.log10(matrix.gene_body_distance + 1)
    matrix['tss_distance_rank'] = matrix.groupby('locus_id').abs_tss_distance.rank(method='min').astype(int)
    matrix['is_nearest_tss'] = matrix.tss_distance_rank.eq(1).astype(int)
    return loci, matrix.sort_values(['locus_id', 'gene_id'], kind='stable').reset_index(drop=True), pd.DataFrame(coverage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--loci', type=Path, required=True, help='CSV with locus_id, chrom, sentinel_pos.')
    p.add_argument('--genome-build', choices=['GRCh38'], required=True, help='Coordinates must already be harmonized.')
    p.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are not overwritten.')
    p.add_argument('--resource-root', type=Path, default=ROOT)
    p.add_argument('--top-k', type=int, default=5)
    args = p.parse_args()
    if args.top_k < 1:
        raise ValueError('--top-k must be positive.')
    if args.output.exists():
        raise FileExistsError('Output directory already exists; use a new name.')
    resource = args.resource_root.resolve()
    resource_manifest = resource / 'provenance/resource_manifest.json'
    if resource_manifest.exists():
        manifest = json.loads(resource_manifest.read_text())
        for record in manifest['files']:
            if hashlib.sha256((resource / record['path']).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Resource checksum mismatch: ' + record['path'])
    cfg = json.loads((resource / 'config/final_protocol.json').read_text())
    genes = pd.read_parquet(resource / 'data/resources/gene_annotations.parquet')
    genes['chrom'] = genes.chrom.astype(str)
    input_loci = pd.read_csv(args.loci, dtype={'locus_id': str, 'chrom': str})
    if input_loci.empty:
        raise ValueError('The locus input is empty.')
    loci, matrix, coverage = prepare(input_loci, genes)
    splits = pd.read_csv(resource / 'data/splits/locus_split_assignments.csv', dtype={'chrom': str})
    folds = splits.groupby('chrom').outer_test_fold.agg(lambda s: int(s.iloc[0]) if s.nunique() == 1 else None).to_dict()
    if not set(loci.chrom).issubset(folds):
        raise ValueError('Supported chromosomes are 1-22 and X; all require a saved chromosome-held-out model.')
    if not np.isfinite(matrix[cfg['features']].to_numpy(float)).all():
        raise ValueError('Nonfinite predictors; check the resource package.')
    matrix['model_fold'] = matrix.chrom.map(folds).astype(int)
    scores = matrix[['locus_id', 'gene_id', 'gene_name', 'chrom', 'gene_biotype', 'sentinel_pos',
                     'gene_body_distance', 'abs_tss_distance', 'model_fold']].copy()
    model_records = []
    for kind in ['ranker', 'classifier']:
        scores[kind + '_score'] = np.nan
        for fold in sorted(matrix.model_fold.unique()):
            path = resource / f'results/final_fit_01/models/{kind}_fold_{fold}.txt'
            model = lgb.Booster(model_file=str(path))
            if model.feature_name() != cfg['features']:
                raise ValueError('Model feature schema does not match the resource package.')
            mask = matrix.model_fold.eq(fold)
            allowed = set(splits.loc[splits[f'fold_{fold}_role'].eq('test'), 'chrom'])
            if not set(matrix.loc[mask, 'chrom']).issubset(allowed):
                raise ValueError('Chromosome exclusion failed.')
            scores.loc[mask, kind + '_score'] = model.predict(matrix.loc[mask, cfg['features']])
            model_records.append(dict(model=kind, fold=int(fold), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        scores[kind + '_rank_min'] = scores.groupby('locus_id')[kind + '_score'].rank(method='min', ascending=False).astype(int)
        scores[kind + '_rank_max'] = scores.groupby('locus_id')[kind + '_score'].rank(method='max', ascending=False).astype(int)
    if not np.isfinite(scores[['ranker_score', 'classifier_score']].to_numpy()).all():
        raise ValueError('Incomplete scores.')
    registry = pd.read_parquet(resource / 'data/processed/training_matrix_dataset_a.parquet',
                               columns=['locus_id', 'chrom', 'sentinel_pos']).drop_duplicates()
    overlap_rows = []
    for row in loci.itertuples(index=False):
        old = registry.loc[registry.chrom.astype(str).eq(row.chrom)]
        overlaps = old.loc[(old.sentinel_pos - row.sentinel_pos).abs().le(1000000)]
        overlap_rows.append(dict(locus_id=row.locus_id, benchmark_window_overlap_count=len(overlaps),
                                  exact_benchmark_sentinel=bool(old.sentinel_pos.eq(row.sentinel_pos).any()),
                                  benchmark_locus_ids=';'.join(overlaps.locus_id),
                                  selected_model_chromosome_excluded=True))
    overlap = pd.DataFrame(overlap_rows)
    scores = scores.merge(overlap, on='locus_id', validate='many_to_one')
    scores['ranker_top5_including_ties'] = scores.ranker_rank_min.le(5)
    scores['classifier_top5_including_ties'] = scores.classifier_rank_min.le(5)
    scores['ranker_shortlisted'] = scores.ranker_rank_min.le(args.top_k)
    scores = scores.sort_values(['locus_id', 'ranker_rank_min', 'gene_id'], kind='stable')
    args.output.mkdir(parents=True)
    matrix.to_parquet(args.output / 'candidate_features.parquet', index=False)
    scores.to_parquet(args.output / 'rankings.parquet', index=False)
    scores.to_csv(args.output / 'rankings.csv', index=False)
    scores.loc[scores.ranker_shortlisted].to_csv(args.output / 'shortlist.csv', index=False)
    overlap.to_csv(args.output / 'benchmark_overlap.csv', index=False)
    coverage.to_csv(args.output / 'candidate_coverage.csv', index=False)
    report = dict(version='2.0.0', genome_build=args.genome_build, input_loci=len(loci),
                  scored_loci=int(scores.locus_id.nunique()), candidate_pairs=len(scores), top_k=args.top_k,
                  predictor_columns=cfg['features'], model_fits=0, scores_are_causal_probabilities=False,
                  chromosome_held_out_prediction=True, ties_included=True, models=model_records)
    (args.output / 'run_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['version', 'input_loci', 'scored_loci', 'candidate_pairs', 'top_k']}, indent=2))


if __name__ == '__main__':
    main()
