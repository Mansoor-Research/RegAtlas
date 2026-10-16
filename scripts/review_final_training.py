"""Independent saved-model and metric checks. No training or tuning."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from run_final_models import load_inputs, verify_recorded_source

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/final_fit_01'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cfg,df,splits=load_inputs()
    status=json.loads((OUT/'run_status.json').read_text())
    assert status['status']=='completed_needs_review' and status['models_fit']==10
    environment=json.loads((OUT/'environment.json').read_text())
    verify_recorded_source('scripts/run_final_models.py', environment['script_sha256'])
    assert environment['protocol_sha256']==sha(ROOT/'config/final_protocol.json')
    assert json.loads((OUT/'protocol.json').read_text())==cfg
    predictions=pd.read_parquet(OUT/'oof_predictions.parquet')
    assert len(predictions)==35358 and predictions.locus_id.nunique()==1075
    assert not predictions.duplicated(['locus_id','gene_id']).any()
    assert np.isfinite(predictions[['ranker','classifier']]).all().all()
    joined=df.merge(predictions,on=['locus_id','gene_id','chrom','label','outer_fold'],validate='one_to_one')
    assert len(joined)==len(df)
    records=json.loads((OUT/'fit_records.json').read_text())
    assert len(records)==10 and len(list((OUT/'models').glob('*.txt')))==10
    reload_checks=[]
    for kind in ['ranker','classifier']:
        for fold in range(5):
            rec=[r for r in records if r['model']==kind and r['fold']==fold]
            assert len(rec)==1
            rec=rec[0]
            path=OUT/'models'/f'{kind}_fold_{fold}.txt'
            assert sha(path)==rec['model_sha256']
            test=joined.loc[joined.outer_fold.eq(fold)]
            assert test.locus_id.nunique()==cfg['expected_test_loci'][fold]
            model=lgb.Booster(model_file=str(path))
            assert np.array_equal(model.predict(test[cfg['features']],num_threads=1),test[kind].to_numpy())
            assert model.feature_name()==cfg['features']
            assert model.current_iteration()==rec['best_iteration']
            reload_checks.append(dict(model=kind,fold=fold,exact_saved_prediction_match=True))
    # Recompute tie expectations independently by sorting score blocks and enumerating
    # every permissible position occupied by the nominated gene.
    methods={'ranker':False,'classifier':False,'log10_gene_body_distance':True,'abs_tss_distance':True}
    metrics={}
    for method,ascending in methods.items():
        rows=[]
        for locus,group in joined.groupby('locus_id',sort=True):
            ordered=group.sort_values(method,ascending=ascending)
            positive=float(ordered.loc[ordered.label.eq(1),method].iloc[0])
            positions=np.flatnonzero(ordered[method].to_numpy()==positive)+1
            rows.append(dict(locus_id=locus,top1=np.mean(positions<=1),top3=np.mean(positions<=3),top5=np.mean(positions<=5),
                mrr=np.mean(1/positions),ndcg5=np.mean(np.where(positions<=5,1/np.log2(positions+1),0)),expected_rank=np.mean(positions)))
        metrics[method]=pd.DataFrame(rows).set_index('locus_id')
    summary=json.loads((OUT/'summary.json').read_text())
    for method,frame in metrics.items():
        for metric,val in frame.mean().items():assert np.isclose(val,summary['metrics'][method][metric],atol=1e-14,rtol=0)
    for name,comparison in summary['paired_intervals'].items():
        left,right=name.split(' minus ')
        for metric,reported in comparison.items():
            delta=(metrics[left][metric]-metrics[right][metric]).to_numpy()
            rng=np.random.RandomState(cfg['bootstrap_seed'])
            means=[delta[rng.randint(0,len(delta),len(delta))].mean() for _ in range(cfg['bootstrap_resamples'])]
            assert np.isclose(delta.mean(),reported['delta'],atol=1e-14,rtol=0)
            assert np.allclose(np.quantile(means,[.025,.975]),reported['ci95'],atol=1e-14,rtol=0)
    result = dict(
        status='passed', models_refit=0, models_reloaded=10, pairs=35358, loci=1075,
        source_and_protocol_hashes='passed', independent_metrics='passed',
        paired_bootstrap_recalculation='passed', saved_prediction_checks=reload_checks,
        verification_script_sha256=sha(Path(__file__)),
        scope='Saved-model predictions, candidate keys, fold counts, metrics and paired intervals.'
    )
    verification = ROOT / 'results/verification'
    verification.mkdir(exist_ok=True)
    (verification / 'full_models.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (verification / 'full_models.md').write_text(
        '# Full-model verification\n\n'
        'All ten saved models passed hash and feature checks. Reloaded predictions exactly '
        'matched the supplied held-out scores. Candidate keys and fold counts matched the '
        'fixed benchmark. Independent metric calculations and paired bootstrap intervals '
        'matched the recorded results. No models were fitted.\n', encoding='utf-8'
    )
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
