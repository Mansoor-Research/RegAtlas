"""One locked post-development control batch on the final-fit candidate matrix."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_final_models import neutral_metrics, self_test, sha

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "config/internal_controls_protocol.json"
DESIGN = ROOT / "internal_controls_design.md"
MEASURES = ["top1", "top3", "top5", "mrr", "ndcg5", "expected_rank"]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_scientific_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Verify frozen scientific inputs; archived R2 manuscript files are not model inputs."""
    manifest = json.loads((ROOT / "provenance/input_manifest.json").read_text())
    for record in manifest["files"]:
        relative = record["path"]
        if relative.startswith("manuscript/"):
            continue
        if sha(ROOT / relative) != record["sha256"]:
            raise ValueError("Scientific input changed: " + relative)
    package = json.loads((ROOT / "provenance/package_lock.json").read_text())
    for record in package["files"]:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Frozen configuration changed: " + record["path"])
    for name, version in manifest["environment"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Dependency mismatch: " + name)
    base = json.loads((ROOT / "config/final_protocol.json").read_text())
    df = pd.read_parquet(ROOT / "data/processed/training_matrix_dataset_a.parquet")
    splits = pd.read_csv(
        ROOT / "data/splits/locus_split_assignments.csv",
        dtype={"locus_id": str, "chrom": str},
    )
    df["locus_id"] = df.locus_id.astype(str)
    df["chrom"] = df.chrom.astype(str)
    if len(df) != base["expected_pairs"] or df.locus_id.nunique() != base["expected_loci"]:
        raise ValueError("Frozen cohort mismatch")
    if df.duplicated(["locus_id", "gene_id"]).any():
        raise ValueError("Duplicate candidate key")
    if df[["locus_id", "gene_id", "chrom", "label"]].isna().any().any():
        raise ValueError("Missing keys or labels")
    if not df.label.isin([0, 1]).all() or not df.groupby("locus_id").label.sum().eq(1).all():
        raise ValueError("Require one nomination per locus")
    if not df.groupby("locus_id").chrom.nunique().eq(1).all():
        raise ValueError("Locus spans chromosomes")
    if not df.groupby("gene_id").chrom.nunique().eq(1).all():
        raise ValueError("Gene spans chromosomes")
    if not np.isfinite(df[base["features"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite model features")
    if splits.locus_id.duplicated().any() or set(splits.locus_id) != set(df.locus_id):
        raise ValueError("Split key mismatch")
    sm = splits.set_index("locus_id")
    if not df.chrom.eq(df.locus_id.map(sm.chrom)).all():
        raise ValueError("Split chromosome mismatch")
    if not splits.outer_test_fold.isin(range(5)).all():
        raise ValueError("Invalid outer fold")
    for fold in range(5):
        roles = sm[f"fold_{fold}_role"]
        if not roles.isin(["fit", "val", "test"]).all() or set(roles) != {"fit", "val", "test"}:
            raise ValueError("Invalid fold roles")
        if not roles.eq("test").equals(sm.outer_test_fold.eq(fold)):
            raise ValueError("Outer fold/role mismatch")
        if int(roles.eq("test").sum()) != base["expected_test_loci"][fold]:
            raise ValueError("Test locus count mismatch")
        if set(sm.loc[roles.eq("test"), "chrom"]) & set(sm.loc[~roles.eq("test"), "chrom"]):
            raise ValueError("Outer chromosome leakage")
    df = df.sort_values(["locus_id", "gene_id"], kind="stable").reset_index(drop=True)
    df["outer_fold"] = df.locus_id.map(sm.outer_test_fold).astype(int)
    return base, df, sm


def check_design(cfg: dict, base: dict, df: pd.DataFrame) -> None:
    features = base["features"]
    if cfg["source_full_fit"] != "results/final_fit_01":
        raise ValueError("Unexpected full-fit source")
    if cfg["run_id"] != "internal_controls_01":
        raise ValueError("Unexpected control run ID")
    if cfg["feature_sets"]["distance"] != features[:5]:
        raise ValueError("Distance features differ from frozen full model")
    expected_flags = [
        "has_cortex_eqtl",
        "has_ba9_eqtl",
        "has_any_brain_eqtl",
        "has_re2g_link",
        "has_both_eqtl_and_re2g",
    ]
    if cfg["feature_sets"]["distance_availability"] != features[:5] + expected_flags:
        raise ValueError("Availability features differ from locked addendum")
    if cfg["model_objectives"] != ["ranker", "classifier"]:
        raise ValueError("Unexpected model objectives")
    if cfg["seed"] != base["seed"] or cfg["bootstrap_seed"] != base["bootstrap_seed"]:
        raise ValueError("Control seeds differ from full fits")
    if cfg["bootstrap_resamples"] != base["bootstrap_resamples"]:
        raise ValueError("Control bootstrap differs from full fits")
    if cfg["primary_endpoint"] != "top5":
        raise ValueError("Unexpected primary endpoint")
    if not set(expected_flags).issubset(df.columns):
        raise ValueError("Missing availability flags")
    if df.loc[df.label.eq(1), "gene_biotype"].ne(cfg["coding_biotype"]).any():
        raise ValueError("Not every source nomination is protein coding")
    coding = df.loc[df.gene_biotype.eq(cfg["coding_biotype"])]
    if coding.locus_id.nunique() != base["expected_loci"]:
        raise ValueError("Coding restriction removes a nominated locus")
    if not coding.groupby("locus_id").label.sum().eq(1).all():
        raise ValueError("Coding restriction changed positive-label count")
    original = ROOT / cfg["source_full_fit"]
    if json.loads((original / "run_status.json").read_text())["status"] != "completed_needs_review":
        raise ValueError("Full-fit run is not complete")
    original_cfg = json.loads((original / "protocol.json").read_text())
    if original_cfg != base:
        raise ValueError("Full-fit protocol differs from frozen protocol")
    if not (original / "oof_predictions.parquet").is_file():
        raise ValueError("Missing final-fit out-of-fold predictions")


def fit_controls(cfg: dict, base: dict, df: pd.DataFrame, splits: pd.DataFrame, out: Path) -> pd.DataFrame:
    modeldir = out / "models"
    modeldir.mkdir()
    predictions = df[["locus_id", "gene_id", "chrom", "label", "outer_fold"]].copy()
    records = []
    for set_name, features in cfg["feature_sets"].items():
        for objective in cfg["model_objectives"]:
            column = f"{set_name}_{objective}"
            predictions[column] = np.nan
            for fold in range(5):
                seed = cfg["seed"] + fold
                random.seed(seed)
                np.random.seed(seed)
                roles = df.locus_id.map(splits[f"fold_{fold}_role"])
                fit = df.loc[roles.eq("fit")]
                validation = df.loc[roles.eq("val")]
                test = df.loc[roles.eq("test")]
                params = dict(base["common_parameters"], **base[f"{objective}_parameters"])
                params.update(
                    seed=seed,
                    data_random_seed=seed,
                    feature_fraction_seed=seed,
                    bagging_seed=seed,
                    drop_seed=seed,
                )

                def dataset(frame: pd.DataFrame, reference=None):
                    groups = (
                        frame.groupby("locus_id", sort=False).size().to_numpy()
                        if objective == "ranker"
                        else None
                    )
                    return lgb.Dataset(
                        frame[features],
                        label=frame.label.to_numpy(),
                        group=groups,
                        reference=reference,
                    )

                training = dataset(fit)
                valid = dataset(validation, training)
                history = {}
                model = lgb.train(
                    params,
                    training,
                    num_boost_round=base["num_boost_round"],
                    valid_sets=[valid],
                    valid_names=["validation"],
                    callbacks=[
                        lgb.early_stopping(base["early_stopping_rounds"], verbose=False),
                        lgb.record_evaluation(history),
                    ],
                )
                model_path = modeldir / f"{column}_fold_{fold}.txt"
                model.save_model(str(model_path), num_iteration=model.best_iteration)
                score = model.predict(test[features], num_iteration=model.best_iteration)
                reloaded = lgb.Booster(model_file=str(model_path))
                if reloaded.feature_name() != features:
                    raise ValueError(f"Model features changed: {column}, fold {fold}")
                if not np.array_equal(score, reloaded.predict(test[features])):
                    raise ValueError(f"Reload prediction mismatch: {column}, fold {fold}")
                if not np.isfinite(score).all():
                    raise ValueError(f"Nonfinite score: {column}, fold {fold}")
                predictions.loc[test.index, column] = score
                records.append(
                    {
                        "column": column,
                        "fold": fold,
                        "seed": seed,
                        "features": features,
                        "fit_loci": int(fit.locus_id.nunique()),
                        "validation_loci": int(validation.locus_id.nunique()),
                        "test_loci": int(test.locus_id.nunique()),
                        "best_iteration": int(model.best_iteration),
                        "model_sha256": sha(model_path),
                        "reload_prediction_exact_match": True,
                        "evaluation_history": history,
                    }
                )
                print(f"{column} fold {fold}: iteration {model.best_iteration}", flush=True)
    columns = [f"{name}_{objective}" for name in cfg["feature_sets"] for objective in cfg["model_objectives"]]
    if predictions[columns].isna().any().any():
        raise ValueError("Incomplete control predictions")
    predictions.to_parquet(out / "oof_control_predictions.parquet", index=False)
    write_json(out / "fit_records.json", records)
    return predictions


def method_metrics(frame: pd.DataFrame, columns: list[str], coding: bool) -> pd.DataFrame:
    parts = [neutral_metrics(frame, column) for column in columns]
    body = neutral_metrics(frame, "log10_gene_body_distance", ascending=True)
    if coding:
        body["method"] = "protein_coding_gene_body_distance"
    parts.append(body)
    parts.append(neutral_metrics(frame, "abs_tss_distance", ascending=True))
    metrics = pd.concat(parts, ignore_index=True)
    if metrics.groupby("method").locus_id.nunique().ne(1075).any():
        raise ValueError("Metric coverage is incomplete")
    if not metrics.top5.between(0, 1).all():
        raise ValueError("Invalid Top-5 credit")
    return metrics


def paired_top5(metrics: pd.DataFrame, left: str, right: str, seed: int, nboot: int) -> dict:
    a = metrics.loc[metrics.method.eq(left)].set_index("locus_id").sort_index()
    b = metrics.loc[metrics.method.eq(right)].set_index("locus_id").reindex(a.index)
    if len(a) != 1075 or b.top5.isna().any():
        raise ValueError("Paired contrast missing loci")
    delta = (a.top5 - b.top5).to_numpy()
    rng = np.random.RandomState(seed)
    samples = rng.randint(0, len(delta), size=(nboot, len(delta)))
    boot = delta[samples].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "left": left,
        "right": right,
        "delta": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
    }


def evaluate(cfg: dict, base: dict, df: pd.DataFrame, controls: pd.DataFrame, out: Path) -> None:
    original = pd.read_parquet(ROOT / cfg["source_full_fit"] / "oof_predictions.parquet")
    keys = ["locus_id", "gene_id"]
    if original.duplicated(keys).any() or controls.duplicated(keys).any():
        raise ValueError("Duplicated candidate key")
    merged = df.merge(original[keys + ["ranker", "classifier"]], on=keys, validate="one_to_one")
    merged = merged.merge(
        controls[keys + [c for c in controls.columns if c not in keys + ["chrom", "label", "outer_fold"]]],
        on=keys,
        validate="one_to_one",
    )
    if len(merged) != base["expected_pairs"]:
        raise ValueError("Candidate coverage changed")
    score_columns = ["ranker", "classifier"] + [
        f"{name}_{objective}" for name in cfg["feature_sets"] for objective in cfg["model_objectives"]
    ]
    all_metrics = method_metrics(merged, score_columns, coding=False)
    coding_frame = merged.loc[merged.gene_biotype.eq(cfg["coding_biotype"])].copy()
    coding_metrics = method_metrics(coding_frame, score_columns, coding=True)
    all_metrics.to_csv(out / "per_locus_all.csv", index=False)
    coding_metrics.to_csv(out / "per_locus_protein_coding.csv", index=False)
    all_summary = all_metrics.groupby("method")[MEASURES].mean().sort_index()
    coding_summary = coding_metrics.groupby("method")[MEASURES].mean().sort_index()
    all_summary.to_csv(out / "metrics_all.csv")
    coding_summary.to_csv(out / "metrics_protein_coding.csv")
    full_reference = json.loads((ROOT / cfg["source_full_fit"] / "summary.json").read_text())["metrics"]
    for name in ["ranker", "classifier", "log10_gene_body_distance", "abs_tss_distance"]:
        if not np.isclose(all_summary.loc[name, "top5"], full_reference[name]["top5"], atol=1e-12):
            raise ValueError(f"Full-fit metric mismatch: {name}")
    coding_counts = coding_frame.groupby("locus_id").size()
    counts = {
        "protein_coding_candidate_rows": int(len(coding_frame)),
        "other_biotype_candidate_rows": int(len(merged) - len(coding_frame)),
        "protein_coding_loci": int(len(coding_counts)),
        "coding_candidates_min": int(coding_counts.min()),
        "coding_candidates_median": float(coding_counts.median()),
        "coding_candidates_max": int(coding_counts.max()),
        "loci_with_five_or_fewer_protein_coding_candidates": int(coding_counts.le(5).sum()),
    }
    contrasts_all = [
        paired_top5(all_metrics, left, right, cfg["bootstrap_seed"], cfg["bootstrap_resamples"])
        for left, right in cfg["paired_top5_contrasts"]
    ]
    contrasts_coding = [
        paired_top5(coding_metrics, left, right, cfg["bootstrap_seed"], cfg["bootstrap_resamples"])
        for left, right in cfg["coding_subset_top5_contrasts"]
    ]
    write_json(
        out / "summary.json",
        {
            "counts": counts,
            "metrics_all": all_summary.to_dict(orient="index"),
            "metrics_protein_coding": coding_summary.to_dict(orient="index"),
            "paired_top5_all": contrasts_all,
            "paired_top5_protein_coding": contrasts_coding,
            "interpretation_boundary": (
                "Post-development sensitivity controls. Full-versus-distance changes do not establish "
                "a causal enhancer mechanism. Protein-coding restriction changes the candidate universe."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true", help="Fit four locked controls over five folds each.")
    args = parser.parse_args()
    self_test()
    cfg = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    base, df, splits = load_scientific_inputs()
    check_design(cfg, base, df)
    if not args.fit:
        print(json.dumps({"preflight": "passed", "loci": df.locus_id.nunique(), "pairs": len(df), "control_fits": 20}))
        return
    out = ROOT / "results" / cfg["run_id"]
    out.mkdir(exist_ok=False)
    status = {"status": "running", "started": datetime.now(timezone.utc).isoformat()}
    write_json(out / "run_status.json", status)
    try:
        write_json(out / "protocol.json", cfg)
        write_json(
            out / "environment.json",
            {
                "python": sys.version,
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ["lightgbm", "numpy", "pandas", "scikit-learn", "pyarrow"]
                },
                "script_sha256": sha(Path(__file__)),
                "protocol_sha256": sha(PROTOCOL),
                "design_sha256": sha(DESIGN),
                "full_fit_predictions_sha256": sha(ROOT / cfg["source_full_fit"] / "oof_predictions.parquet"),
            },
        )
        control_scores = fit_controls(cfg, base, df, splits, out)
        evaluate(cfg, base, df, control_scores, out)
        status.update(status="completed_needs_review", models_fit=20, finished=datetime.now(timezone.utc).isoformat())
    except Exception:
        (out / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        status.update(status="failed", finished=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write_json(out / "run_status.json", status)


if __name__ == "__main__":
    main()
