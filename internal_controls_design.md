# Reduced-feature and candidate-universe controls

These post-development sensitivity analyses assess the information captured by the full model. The control comparisons were fixed before the control models were fitted. They use the same processed candidate matrix, nomination labels, chromosome folds and inner validation assignments as the full-feature analysis.

## Model controls

| Feature set | Inputs | Objectives |
| --- | --- | --- |
| Distance | Five proximity features | LightGBM LambdaRank and binary classification |
| Distance and availability | Five proximity features plus five binary annotation-availability indicators | LightGBM LambdaRank and binary classification |

The availability indicators are `has_cortex_eqtl`, `has_ba9_eqtl`, `has_any_brain_eqtl`, `has_re2g_link` and `has_both_eqtl_and_re2g`. The full-versus-availability comparison covers the remaining quantitative summaries and promoter indicator together. It does not isolate a distal-enhancer mechanism.

All four control models use the same outer and inner assignments, fold seeds 42–46, common parameters and early-stopping rules as the full models. The complete feature lists and parameters are recorded in `config/internal_controls_protocol.json`. The supplied full-model predictions remain the reference for the control comparisons.

## Candidate-universe sensitivity

All methods are reevaluated on the same protein-coding candidate subset without retraining. The analysis reports the number of candidates per locus and the number of loci with five or fewer coding candidates, for which Top-5 recovery is automatic.

## Endpoints and uncertainty

Top-5 source-nomination recovery is the primary endpoint, with equal weights for loci and fractional expected credit for exact-score ties. Top-1 recovery and mean reciprocal rank are secondary endpoints. Prelisted Top-5 comparisons use 2,000 paired locus-bootstrap resamples with seed 42, conditional on the fitted models.

The comparisons are full versus distance-only, full versus distance-plus-availability, and distance-plus-availability versus distance-only within each objective. The coding-only analysis compares the full models with gene-body distance in the restricted universe. Exact contrasts are listed in the protocol configuration.

An interval spanning zero does not establish equivalence. Candidate filtering changes the evaluation task. Nomination recovery alone does not establish biological causality or experimental success.
