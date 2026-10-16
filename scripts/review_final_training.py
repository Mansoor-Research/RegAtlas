"""Independent saved-model and metric checks. No training or tuning."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from run_final_models import load_inputs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/final_fit_01'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cfg,df,splits=load_inputs()
    status=json.loads((OUT/'run_status.json').read_text())
    assert status['status']=='completed_needs_review' and status['models_fit']==10
    environment=json.loads((OUT/'environment.json').read_text())
    assert environment['script_sha256']==sha(ROOT/'scripts/run_final_models.py')
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
    result=dict(status='passed',models_refit=0,models_reloaded=10,pairs=35358,loci=1075,
        source_and_protocol_hashes='passed',independent_metrics='passed',paired_bootstrap_recalculation='passed',
        saved_prediction_checks=reload_checks,review_script_sha256=sha(Path(__file__)),
        review_scope='Validates recorded fresh execution and outputs; not a second training run or certification of historical selection.')
    (OUT/'coordinator_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'coordinator_review.md').write_text('''# Final training coordinator review

The saved fresh execution passes review. All ten model files were independently reloaded, their hashes checked, and their predictions exactly matched to the stored held-out scores. Cohort keys and fixed fold counts match. Neutral-tie metrics and all reported paired bootstrap intervals were independently recalculated. No additional fits were run.

## What survives

Fresh RegAtlas Top-5 recovery is 64.17%, versus 53.78% for gene-body distance: +10.38 percentage points, paired 95% interval +7.70 to +13.23. The improvement over that proximity heuristic remains supported in this benchmark.

The classifier reaches 64.19% Top-5. Ranker minus classifier is -0.02 percentage points (95% interval -1.76 to +1.77). Fresh Top-1 is 31.77% versus 30.88%; the difference of +0.88 percentage points has interval -0.98 to +2.84. MRR difference +0.00489 also spans zero. No clear ranking advantage over this matched classifier is demonstrated on these endpoints. This is not an equivalence conclusion.

## Consequences for the paper

Use the fresh results as the main analysis. The historical positive Top-1/MRR ranker-over-classifier finding is not supported by this fresh run. Historical predictions did not match exactly, and the cause cannot be assigned conclusively from these outputs. Fresh documented execution addresses the current model-training provenance gap, while historical model selection remains uncertified.

The defensible claim is: under this fixed chromosome-held-out benchmark, RegAtlas improves nominated-gene shortlisting over gene-body proximity, with no demonstrated advantage over a same-feature classifier. It does not establish molecular attribution, external-tool superiority or independent biological validation. A completed reproducible analysis is not itself a guarantee of novelty or journal acceptance.

The reduced-feature and candidate-universe sensitivity analyses are supplied separately in `results/internal_controls_01` and checked by `scripts/review_internal_controls.py`.
''',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
