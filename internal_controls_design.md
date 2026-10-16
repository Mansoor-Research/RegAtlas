# Locked internal-control addendum

Authorized after the final full-feature fit, before the control results were examined. This addendum does not change `study_design.md`, `config/final_protocol.json`, the 22-feature matrix, the five outer/inner partitions, or `results/final_fit_01`.

## Question and scope

Determine whether the observed Top-5 gain over a single gene-body-distance rule survives stronger internal comparisons. These controls are a post-development sensitivity analysis, not a prospectively registered independent validation. Regardless of outcome, report all three control families and stop without results-driven feature search.

## Fixed controls

1. **Learned distance only:** five proximity features, with both LambdaRank and binary LightGBM.
2. **Learned distance plus source availability:** the same five features plus `has_cortex_eqtl`, `has_ba9_eqtl`, `has_any_brain_eqtl`, `has_re2g_link`, and `has_both_eqtl_and_re2g`. Both objectives are fitted. The full-versus-this-control contrast covers quantitative scores, counts, and the promoter indicator; it does not isolate an enhancer mechanism.
3. **Protein-coding gene-body heuristic:** rank protein-coding candidates by gene-body distance. Evaluate all existing and new methods on this same restricted candidate set, in addition to the unrestricted benchmark. Report the number of protein-coding candidates per locus and the number with five or fewer, where Top-5 recovery is automatic for any ordering.

Use identical candidate rows, labels, outer and inner roles, fold seeds 42–46, parameters, stopping rules and neutral tie metrics for the four new learned models. The fixed full-feature predictions in `results/final_fit_01` are the comparator; do not refit them. The primary endpoint remains equal-locus Top-5 source-nomination recovery. Top-1 and MRR are descriptive secondary endpoints. Use 2,000 paired locus bootstraps with seed 42 for prelisted Top-5 contrasts, conditional on fixed fits. Do not treat confidence intervals crossing zero as evidence of equivalence.

The control runner verifies SHA-256 hashes for every scientific file in the frozen input manifest and for the unchanged frozen configuration package. Historical R2 manuscript artifacts have been archived or revised for the new manuscript and are excluded from this new run's input-file check because they are not training inputs. The original full-fit runner and its saved outputs remain untouched.

## Prelisted comparisons and interpretation

- Full ranker versus distance-only ranker; full classifier versus distance-only classifier.
- Full ranker versus distance-plus-availability ranker; full classifier versus distance-plus-availability classifier.
- Distance-plus-availability versus distance-only within each objective.
- Within the protein-coding candidate subset, full ranker and full classifier versus protein-coding gene-body distance, with all methods scored on the same subset.

If full models exceed distance-only, that supports incremental predictive information from non-distance features in this benchmark, not causal enhancer attribution. If they do not, narrow the manuscript claim accordingly. Protein-coding filtering changes the candidate universe and must be reported as a sensitivity analysis, not silently substituted for the main task. No outcome by itself establishes publication readiness, external-tool superiority or experimental validity.
