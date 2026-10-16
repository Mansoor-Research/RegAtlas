"""Train and evaluate matched LightGBM models on fixed processed inputs.

Default command only validates. --fit explicitly starts ten model fits.
"""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, platform, random, sys, traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def verify_recorded_source(relative, recorded_sha256):
    """Validate release sources against the unchanged generating-run metadata."""
    if sha(ROOT / relative) == recorded_sha256:
        return
    history = json.loads((ROOT / 'provenance/source_history.json').read_text())
    entry = history['files'].get(relative)
    if entry is None or entry['generating_sha256'] != recorded_sha256:
        raise ValueError('Generating source hash mismatch: ' + relative)
    if sha(ROOT / relative) != entry['release_sha256']:
        raise ValueError('Release source hash mismatch: ' + relative)

def load_inputs():
    manifest = json.loads((ROOT/'provenance/input_manifest.json').read_text())
    for rec in manifest['files']:
        if sha(ROOT/rec['path']) != rec['sha256']: raise ValueError('Input changed: '+rec['path'])
    package = ROOT/'provenance/package_lock.json'
    if package.exists():
        for rec in json.loads(package.read_text())['files']:
            if sha(ROOT/rec['path']) != rec['sha256']: raise ValueError('Active file changed: '+rec['path'])
    cfg = json.loads((ROOT/'config/final_protocol.json').read_text())
    # A changed dependency cannot quietly change the frozen execution environment.
    for name, version in manifest['environment'].items():
        if importlib.metadata.version(name) != version: raise ValueError('Dependency mismatch: '+name)
    df = pd.read_parquet(ROOT/'data/processed/training_matrix_dataset_a.parquet')
    splits = pd.read_csv(ROOT/'data/splits/locus_split_assignments.csv', dtype={'locus_id':str,'chrom':str})
    df['locus_id'] = df.locus_id.astype(str)
    df['chrom'] = df.chrom.astype(str)
    if len(df) != cfg['expected_pairs'] or df.locus_id.nunique() != cfg['expected_loci']: raise ValueError('Cohort mismatch')
    if df.duplicated(['locus_id','gene_id']).any(): raise ValueError('Duplicate candidate key')
    if df[['locus_id','gene_id','chrom','label']].isna().any().any(): raise ValueError('Missing keys/labels')
    if not df.label.isin([0,1]).all() or not df.groupby('locus_id').label.sum().eq(1).all(): raise ValueError('Require exactly one nomination per locus')
    if not df.groupby('locus_id').chrom.nunique().eq(1).all(): raise ValueError('Locus spans chromosomes')
    if not df.groupby('gene_id').chrom.nunique().eq(1).all(): raise ValueError('Gene spans chromosomes')
    if splits.locus_id.duplicated().any() or set(splits.locus_id)!=set(df.locus_id): raise ValueError('Split key mismatch')
    if not np.isfinite(df[cfg['features']].to_numpy(dtype=float)).all(): raise ValueError('Nonfinite model features; do not silently impute')
    sm = splits.set_index('locus_id')
    if not df.chrom.eq(df.locus_id.map(sm.chrom)).all(): raise ValueError('Split chromosome mismatch')
    if not splits.outer_test_fold.isin(range(5)).all(): raise ValueError('Invalid outer fold')
    for fold in range(5):
        roles = sm[f'fold_{fold}_role']
        if not roles.isin(['fit','val','test']).all() or set(roles)!= {'fit','val','test'}: raise ValueError('Invalid roles')
        if not roles.eq('test').equals(sm.outer_test_fold.eq(fold)): raise ValueError('Outer/role mismatch')
        if int(roles.eq('test').sum()) != cfg['expected_test_loci'][fold]: raise ValueError('Fold count mismatch')
        test_chroms = set(sm.loc[roles.eq('test'),'chrom'])
        if test_chroms & set(sm.loc[~roles.eq('test'),'chrom']): raise ValueError('Outer chromosome leakage')
    # Stable row ordering is shared by both objectives and all group arrays.
    df = df.sort_values(['locus_id','gene_id'],kind='stable').reset_index(drop=True)
    df['outer_fold'] = df.locus_id.map(sm.outer_test_fold).astype(int)
    return cfg, df, sm

def neutral_metrics(df, column, ascending=False):
    rows=[]
    for locus, group in df.groupby('locus_id',sort=True):
        scores=group[column].to_numpy(dtype=float)
        if not np.isfinite(scores).all(): raise ValueError('Nonfinite score: '+column)
        positive=scores[group.label.to_numpy()==1]
        if len(positive)!=1: raise ValueError('Metric requires one nomination')
        p=positive[0]
        better=int(np.sum(scores<p if ascending else scores>p))
        ties=int(np.sum(scores==p))
        ranks=np.arange(better+1,better+ties+1,dtype=float)
        rows.append(dict(locus_id=locus,method=column,top1=float(np.mean(ranks<=1)),
            top3=float(np.mean(ranks<=3)),top5=float(np.mean(ranks<=5)),
            mrr=float(np.mean(1/ranks)),ndcg5=float(np.mean(np.where(ranks<=5,1/np.log2(ranks+1),0))),
            expected_rank=float(np.mean(ranks)),tied_candidates=ties))
    return pd.DataFrame(rows)

def self_test():
    x=pd.DataFrame(dict(locus_id=['x']*7,gene_id=list('abcdefg'),label=[1,0,0,0,0,0,0],s=[0.]*7))
    m=neutral_metrics(x,'s').iloc[0]
    assert np.isclose(m.top1,1/7) and np.isclose(m.top5,5/7) and m.expected_rank==4
    x['s']=[7,6,5,4,3,2,1]
    assert neutral_metrics(x,'s').iloc[0].top1==1
    assert neutral_metrics(x,'s',True).iloc[0].top5==0

def fit_models(cfg, df, splits, out):
    modeldir=out/'models'; modeldir.mkdir()
    predictions=df[['locus_id','gene_id','chrom','label','outer_fold']].copy()
    records=[]
    for kind in cfg['models']:
        predictions[kind]=np.nan
        for fold in range(5):
            seed=cfg['seed']+fold
            random.seed(seed); np.random.seed(seed)
            roles=df.locus_id.map(splits[f'fold_{fold}_role'])
            fit=df.loc[roles.eq('fit')]; val=df.loc[roles.eq('val')]; test=df.loc[roles.eq('test')]
            params=dict(cfg['common_parameters'],**cfg[f'{kind}_parameters'])
            params.update(seed=seed,data_random_seed=seed,feature_fraction_seed=seed,bagging_seed=seed,drop_seed=seed)
            def dataset(frame, reference=None):
                groups=frame.groupby('locus_id',sort=False).size().to_numpy() if kind=='ranker' else None
                return lgb.Dataset(frame[cfg['features']],label=frame.label.to_numpy(),group=groups,reference=reference)
            train=dataset(fit); valid=dataset(val,train); history={}
            model=lgb.train(params,train,num_boost_round=cfg['num_boost_round'],valid_sets=[valid],valid_names=['validation'],
                callbacks=[lgb.early_stopping(cfg['early_stopping_rounds'],verbose=False),lgb.record_evaluation(history)])
            path=modeldir/f'{kind}_fold_{fold}.txt'
            model.save_model(str(path),num_iteration=model.best_iteration)
            score=model.predict(test[cfg['features']],num_iteration=model.best_iteration)
            loaded=lgb.Booster(model_file=str(path))
            check=loaded.predict(test[cfg['features']])
            if not np.array_equal(score,check): raise ValueError('Saved model prediction mismatch')
            if not np.isfinite(score).all(): raise ValueError('Nonfinite held-out predictions')
            predictions.loc[test.index,kind]=score
            record=dict(model=kind,fold=fold,seed=seed,parameters=params,best_iteration=model.best_iteration,
                fit_loci=fit.locus_id.nunique(),validation_loci=val.locus_id.nunique(),test_loci=test.locus_id.nunique(),
                model_sha256=sha(path),reload_prediction_exact_match=True,evaluation_history=history)
            dump(out/f'{kind}_fold_{fold}.json',record);records.append(record)
            print(f'{kind} fold {fold} saved; best iteration {model.best_iteration}',flush=True)
    if predictions[cfg['models']].isna().any().any(): raise ValueError('Incomplete OOF predictions')
    predictions.to_parquet(out/'oof_predictions.parquet',index=False)
    dump(out/'fit_records.json',records)
    return predictions

def evaluate(cfg, df, predictions, out):
    merged=df.merge(predictions[['locus_id','gene_id']+cfg['models']],on=['locus_id','gene_id'],validate='one_to_one')
    parts=[neutral_metrics(merged,k) for k in cfg['models']]
    parts += [neutral_metrics(merged,'log10_gene_body_distance',True),neutral_metrics(merged,'abs_tss_distance',True)]
    metrics=pd.concat(parts,ignore_index=True)
    metrics['outer_fold']=metrics.locus_id.map(df.drop_duplicates('locus_id').set_index('locus_id').outer_fold)
    metrics.to_csv(out/'per_locus_metrics.csv',index=False)
    summary=metrics.groupby('method')[['top1','top3','top5','mrr','ndcg5','expected_rank']].mean()
    summary.to_csv(out/'metrics.csv')
    metrics.groupby(['method','outer_fold'])[['top1','top3','top5','mrr','ndcg5','expected_rank']].mean().to_csv(out/'fold_metrics.csv')
    comparisons={}
    for left,right in [('ranker','classifier'),('ranker','log10_gene_body_distance'),('classifier','log10_gene_body_distance')]:
        a=metrics.loc[metrics.method.eq(left)].set_index('locus_id').sort_index()
        b=metrics.loc[metrics.method.eq(right)].set_index('locus_id').reindex(a.index)
        comparisons[left+' minus '+right]={}
        for metric in ['top5','top1','mrr']:
            delta=(a[metric]-b[metric]).to_numpy()
            rng=np.random.RandomState(cfg['bootstrap_seed'])
            boot=np.array([delta[rng.randint(0,len(delta),len(delta))].mean() for _ in range(cfg['bootstrap_resamples'])])
            lo,hi=np.quantile(boot,[.025,.975])
            comparisons[left+' minus '+right][metric]=dict(delta=float(delta.mean()),ci95=[float(lo),float(hi)])
    dump(out/'summary.json',dict(metrics=summary.to_dict(orient='index'),paired_intervals=comparisons,
        primary_endpoint='Top-5 nomination recovery for this final retrospective protocol',
        limitations=['Nominations are not experimental causal labels.','Bootstrap conditions on fixed fits; not full training uncertainty.',
            'Inner validation is locus-held-out, not chromosome-held-out.','No external L2G comparison or independent biological validation.',
            'Full-feature versus heuristic differences do not isolate molecular contributions.']))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',action='store_true',help='Fit exactly five ranker and five classifier models.')
    parser.add_argument('--run-id',default=None,help='New results folder name; existing folders are never overwritten.')
    args=parser.parse_args()
    self_test();cfg,df,splits=load_inputs()
    if not args.fit:
        print(json.dumps(dict(validation='passed',loci=df.locus_id.nunique(),pairs=len(df),features=len(cfg['features']),models_fit=0)))
        return
    if not args.run_id or Path(args.run_id).name!=args.run_id or args.run_id in ['.','..']:
        parser.error('--fit requires a simple new --run-id folder name')
    out=ROOT/'results'/args.run_id
    out.mkdir(exist_ok=False)
    status=dict(status='running',started=datetime.now(timezone.utc).isoformat())
    dump(out/'run_status.json',status)
    try:
        dump(out/'protocol.json',cfg)
        dump(out/'input_manifest.json',json.loads((ROOT/'provenance/input_manifest.json').read_text()))
        if (ROOT/'provenance/package_lock.json').exists():
            dump(out/'package_lock.json',json.loads((ROOT/'provenance/package_lock.json').read_text()))
        dump(out/'environment.json',dict(python=sys.version,platform=platform.platform(),
            dependencies={p:importlib.metadata.version(p) for p in ['lightgbm','numpy','pandas','scikit-learn','pyarrow']},
            script_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'config/final_protocol.json')))
        predictions=fit_models(cfg,df,splits,out)
        evaluate(cfg,df,predictions,out)
        status.update(status='completed_needs_review',models_fit=10,finished=datetime.now(timezone.utc).isoformat())
    except Exception:
        (out/'failure.log').write_text(traceback.format_exc())
        status.update(status='failed',finished=datetime.now(timezone.utc).isoformat())
        raise
    finally: dump(out/'run_status.json',status)

if __name__=='__main__':main()
