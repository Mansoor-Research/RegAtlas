"""Verify the supplied reduced-feature fits and reported control metrics; never train."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_final_models import sha
from run_internal_controls import (
    MEASURES,
    check_design,
    load_scientific_inputs,
    method_metrics,
    paired_top5,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/internal_controls_01"


def check_table(saved: pd.DataFrame, actual: pd.DataFrame) -> None:
    order = ["method", "locus_id"]
    saved = saved.sort_values(order).reset_index(drop=True)
    actual = actual.sort_values(order).reset_index(drop=True)
    assert saved[order].equals(actual[order])
    for column in MEASURES:
        assert np.allclose(saved[column], actual[column], rtol=0, atol=1e-12)


def main() -> None:
    cfg = json.loads((ROOT / "config/internal_controls_protocol.json").read_text())
    base, df, _ = load_scientific_inputs()
    check_design(cfg, base, df)
    assert json.loads((OUT / "run_status.json").read_text())["models_fit"] == 20
    assert json.loads((OUT / "protocol.json").read_text()) == cfg
    environment = json.loads((OUT / "environment.json").read_text())
    assert environment["script_sha256"] == sha(ROOT / "scripts/run_internal_controls.py")
    assert environment["protocol_sha256"] == sha(ROOT / "config/internal_controls_protocol.json")
    assert environment["design_sha256"] == sha(ROOT / "internal_controls_design.md")
    assert environment["full_fit_predictions_sha256"] == sha(
        ROOT / "results/final_fit_01/oof_predictions.parquet"
    )

    predictions = pd.read_parquet(OUT / "oof_control_predictions.parquet")
    assert len(predictions) == len(df)
    assert not predictions.duplicated(["locus_id", "gene_id"]).any()
    joined = df.merge(
        predictions, on=["locus_id", "gene_id", "chrom", "label", "outer_fold"],
        validate="one_to_one",
    )
    records = json.loads((OUT / "fit_records.json").read_text())
    assert len(records) == 20
    assert len(list((OUT / "models").glob("*.txt"))) == 20
    for record in records:
        column, fold, features = record["column"], record["fold"], record["features"]
        path = OUT / "models" / f"{column}_fold_{fold}.txt"
        assert sha(path) == record["model_sha256"]
        model = lgb.Booster(model_file=str(path))
        assert model.feature_name() == features
        assert model.current_iteration() == record["best_iteration"]
        test = joined.loc[joined.outer_fold.eq(fold)]
        assert test.locus_id.nunique() == record["test_loci"]
        assert np.array_equal(
            model.predict(test[features], num_threads=1), test[column].to_numpy()
        )

    full = pd.read_parquet(ROOT / "results/final_fit_01/oof_predictions.parquet")
    joined = joined.merge(full[["locus_id", "gene_id", "ranker", "classifier"]],
                          on=["locus_id", "gene_id"], validate="one_to_one")
    columns = ["ranker", "classifier"] + [
        f"{name}_{objective}"
        for name in cfg["feature_sets"] for objective in cfg["model_objectives"]
    ]
    all_metrics = method_metrics(joined, columns, coding=False)
    coding_metrics = method_metrics(
        joined.loc[joined.gene_biotype.eq(cfg["coding_biotype"])], columns, coding=True
    )
    check_table(pd.read_csv(OUT / "per_locus_all.csv"), all_metrics)
    check_table(pd.read_csv(OUT / "per_locus_protein_coding.csv"), coding_metrics)
    summary = json.loads((OUT / "summary.json").read_text())
    for name, frame in [("metrics_all", all_metrics),
                        ("metrics_protein_coding", coding_metrics)]:
        actual = frame.groupby("method")[MEASURES].mean()
        for method, metrics in summary[name].items():
            for measure, reported in metrics.items():
                assert np.isclose(actual.loc[method, measure], reported, rtol=0, atol=1e-12)
    for name, frame, contrasts in [
        ("paired_top5_all", all_metrics, cfg["paired_top5_contrasts"]),
        ("paired_top5_protein_coding", coding_metrics, cfg["coding_subset_top5_contrasts"]),
    ]:
        for (left, right), reported in zip(contrasts, summary[name], strict=True):
            actual = paired_top5(frame, left, right, cfg["bootstrap_seed"], cfg["bootstrap_resamples"])
            assert actual["left"] == reported["left"] and actual["right"] == reported["right"]
            assert np.isclose(actual["delta"], reported["delta"], rtol=0, atol=1e-12)
            assert np.allclose(actual["ci95"], reported["ci95"], rtol=0, atol=1e-12)
    print(json.dumps({"status": "passed", "models_reloaded": 20,
                      "pairs": len(joined), "loci": joined.locus_id.nunique(),
                      "control_metrics": "recalculated"}))


if __name__ == "__main__":
    main()
