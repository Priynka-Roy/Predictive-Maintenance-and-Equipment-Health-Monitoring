"""Frozen development-only experiments, then retrospective scoring of the V1 benchmark."""

import hashlib
import json
import logging
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn

from industrialguard.data import load_raw, resample_sensors
from industrialguard.evaluation import alert_policy, evaluate_v2
from industrialguard.monitoring import (
    available_features,
    build_v2_features,
    calibrate_monitor,
    fit_monitor,
    health_indicators,
    monitor_score,
    score_thresholds,
    selection_objective,
)
from industrialguard.regimes import regime_audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v2"
LOGGER = logging.getLogger(__name__)
DISPLAY = {
    "robust_original": "Robust deviation · original 52 inputs",
    "robust_raw": "Robust deviation · raw sensors",
    "robust_pressure": "Robust deviation · pressure engineering",
    "covariance_pressure": "Regularized Mahalanobis · pressure engineering",
    "regime_pressure": "Regime-relative robust monitoring",
    "isolation_reduced": "Isolation Forest · reduced inputs",
}
COMPLEXITY = {"robust": 1, "regime_robust": 2, "covariance": 2, "isolation": 3}


def save_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def flags_for(frame: pd.DataFrame, bundle: dict) -> pd.Series:
    scores = pd.Series(monitor_score(frame, bundle), index=frame.index)
    return alert_policy(scores, score_thresholds(frame, bundle), bundle["persistence_bins"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/v2.json"
    config = json.loads(config_path.read_text())
    plan_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    failures = pd.read_csv(ROOT / "configs/failures.csv", parse_dates=["start", "end"])
    raw_paths = list((ROOT / "data/raw").glob("*.csv"))
    if len(raw_paths) != 1:
        raise RuntimeError(
            "Download official data with python scripts/download.py before V2 reproduction"
        )
    with raw_paths[0].open("rb") as source:
        raw_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    raw, quality = load_raw(raw_paths[0])
    bins = resample_sensors(raw)
    features = build_v2_features(bins)
    all_names = set().union(
        *(available_features(features, spec["feature_set"]) for spec in config["models"])
    )
    features = features.dropna(subset=sorted(all_names))
    development = features.loc[features.index < pd.Timestamp(config["benchmark_start"])]
    LOGGER.info(
        "Prepared %d common complete rows; model selection restricted to %d pre-May rows",
        len(features),
        len(development),
    )
    results, fitted = [], {}
    for fold in config["folds"]:
        train_end, cal_end, eval_end = (
            pd.Timestamp(fold[k]) for k in ["train_end", "calibration_end", "evaluation_end"]
        )
        if eval_end > pd.Timestamp(config["benchmark_start"]):
            raise ValueError("Development fold must not include the previously observed benchmark")
        train = development[
            (development.index >= pd.Timestamp(fold["train_start"]))
            & (development.index < train_end)
        ]
        calibration = development[(development.index >= train_end) & (development.index < cal_end)]
        # Exclude documented intervals from calibration, never claim other bins are certified normal.
        reference = pd.Series(True, index=calibration.index)
        for event in failures.itertuples():
            reference &= ~((calibration.index >= event.start) & (calibration.index <= event.end))
        calibration = calibration[reference]
        context = development[(development.index >= train_end) & (development.index < eval_end)]
        for spec in config["models"]:
            model = fit_monitor(train, spec, config)
            for policy in config["policies"]:
                experiment_id = f"{spec['id']}_q{policy['quantile']}_p{policy['persistence_bins']}"
                bundle = calibrate_monitor(
                    model, calibration, policy["quantile"], policy["persistence_bins"]
                )
                flags = flags_for(context, bundle)
                evaluation = evaluate_v2(flags, failures, start=cal_end, end=eval_end)
                issued = flags[(flags.index >= cal_end) & (flags.index < eval_end)]
                weekly = issued.resample("7D").mean().dropna()
                regime_metrics = {
                    str(r): {
                        "bins": len(block),
                        "issued_alert_fraction": float(flags.loc[block.index].mean()),
                    }
                    for r, block in context.loc[issued.index].groupby("regime")
                }
                record = {
                    "experiment_id": experiment_id,
                    "fold": fold["id"],
                    "model_id": spec["id"],
                    "model_type": spec["type"],
                    "feature_set": spec["feature_set"],
                    "features": bundle["feature_names"],
                    "feature_count": len(bundle["feature_names"]),
                    "train_start": str(train.index.min()),
                    "train_end_exclusive": str(train_end),
                    "calibration_start": str(train_end),
                    "calibration_end_exclusive": str(cal_end),
                    "evaluation_start": str(cal_end),
                    "evaluation_end_exclusive": str(eval_end),
                    "hyperparameters": {
                        "seed": config["seed"],
                        "trees": config["isolation_trees"] if spec["type"] == "isolation" else None,
                        "subsample": config["isolation_max_samples"]
                        if spec["type"] == "isolation"
                        else None,
                        "covariance_estimator": "winsorized training LedoitWolf"
                        if spec["type"] == "covariance"
                        else None,
                    },
                    "quantile": policy["quantile"],
                    "persistence_bins": policy["persistence_bins"],
                    "threshold": bundle["threshold"],
                    "regime_thresholds": bundle.get("regime_thresholds", {}),
                    "fit_seconds": bundle["fit_seconds"],
                    "weekly_alert_fraction_sd": float(weekly.std(ddof=0)),
                    "complexity": COMPLEXITY[spec["type"]],
                    "regime_metrics": regime_metrics,
                    "evaluation": evaluation,
                    "provenance": "V2 development only; no independent event generalization",
                }
                results.append(record)
                if fold["id"] == "primary_development":
                    fitted[experiment_id] = bundle
            LOGGER.info("Development fold %s completed model %s", fold["id"], spec["id"])
    grouped = {}
    for record in results:
        grouped.setdefault(record["experiment_id"], []).append(record)
    summaries = []
    for experiment_id, records in grouped.items():
        exposure = sum(r["evaluation"]["observed_days"] for r in records)
        unannotated_exposure = sum(r["evaluation"]["unannotated_observed_days"] for r in records)
        fraction = (
            sum(
                r["evaluation"]["alert_bin_fraction"] * r["evaluation"]["observed_days"]
                for r in records
            )
            / exposure
        )
        burden = (
            sum(r["evaluation"]["unannotated_episode_count"] for r in records)
            / unannotated_exposure
        )
        summary = {
            "experiment_id": experiment_id,
            "model_id": records[0]["model_id"],
            "feature_set": records[0]["feature_set"],
            "quantile": records[0]["quantile"],
            "persistence_bins": records[0]["persistence_bins"],
            "issued_alert_fraction": fraction,
            "unannotated_burden": burden,
            "weekly_alert_fraction_sd": float(
                np.mean([r["weekly_alert_fraction_sd"] for r in records])
            ),
            "development_detected_events": sum(r["evaluation"]["detected_events"] for r in records),
            "development_event_count": sum(r["evaluation"]["event_count"] for r in records),
            "complexity": records[0]["complexity"],
        }
        summary["eligible"], summary["selection_objective"] = selection_objective(
            summary, config["selection"]
        )
        summaries.append(summary)
    # This selection is completed and persisted before ANY benchmark scores are computed.
    eligible = [s for s in summaries if s["eligible"]]
    if not eligible:
        raise RuntimeError(
            "No predefined development policy meets the workload/silence constraints; do not silently optimize the benchmark"
        )
    selected = min(eligible, key=lambda s: (s["selection_objective"], s["experiment_id"]))
    per_model = {}
    for spec in config["models"]:
        candidates = [s for s in summaries if s["model_id"] == spec["id"]]
        feasible = [s for s in candidates if s["eligible"]]
        per_model[spec["id"]] = min(
            feasible or candidates, key=lambda s: (s["selection_objective"], s["experiment_id"])
        )
    decision = {
        "plan_sha256": plan_hash,
        "selection_provenance": "Both folds end before May 1; benchmark not supplied to selection",
        "selected_model": selected["model_id"],
        "selected_experiment": selected["experiment_id"],
        "objective": config["selection"],
        "per_model": per_model,
        "summaries": summaries,
    }
    save_json(OUT / "selection.json", decision)
    save_json(OUT / "experiments.json", results)
    pd.DataFrame(summaries).to_csv(OUT / "experiments.csv", index=False)
    LOGGER.info(
        "Frozen development selection: %s. Beginning retrospective benchmark only now.",
        selected["experiment_id"],
    )
    model_bundles, comparison = {}, {}
    monitoring = features.copy()
    for spec in config["models"]:
        model_id = spec["id"]
        chosen = per_model[model_id]
        bundle = fitted[chosen["experiment_id"]]
        model_bundles[model_id] = bundle
        timing = perf_counter()
        scores = monitor_score(features, bundle)
        score_seconds = perf_counter() - timing
        flags = alert_policy(
            pd.Series(scores, index=features.index),
            score_thresholds(features, bundle),
            bundle["persistence_bins"],
        )
        health, attention = health_indicators(scores, bundle)
        for suffix, value in {
            "score": scores,
            "alert": flags,
            "health": health,
            "attention": attention,
        }.items():
            monitoring[f"{model_id}_{suffix}"] = value
        primary = next(
            r
            for r in results
            if r["experiment_id"] == chosen["experiment_id"] and r["fold"] == "primary_development"
        )
        retrospective = evaluate_v2(
            flags,
            failures,
            start=config["benchmark_start"],
            end=features.index.max() + pd.Timedelta("5min"),
        )
        months = {}
        for month in (
            features[features.index >= pd.Timestamp(config["benchmark_start"])]
            .index.to_period("M")
            .unique()
        ):
            months[str(month)] = evaluate_v2(
                flags,
                failures,
                start=max(month.start_time, pd.Timestamp(config["benchmark_start"])),
                end=min((month + 1).start_time, features.index.max() + pd.Timedelta("5min")),
            )
        comparison[model_id] = {
            "display_name": DISPLAY[model_id],
            "model_type": spec["type"],
            "feature_set": spec["feature_set"],
            "features": bundle["feature_names"],
            "feature_count": len(bundle["feature_names"]),
            "threshold": bundle["threshold"],
            "quantile": bundle["quantile"],
            "persistence_bins": bundle["persistence_bins"],
            "eligible": chosen["eligible"],
            "development": primary["evaluation"],
            "retrospective": retrospective,
            "retrospective_months": months,
            "regime_metrics": primary["regime_metrics"],
            "fit_seconds": bundle["fit_seconds"],
            "scoring_all_rows_seconds": score_seconds,
            "scoring_and_evaluation_seconds": perf_counter() - timing,
            "limitations": "Previously inspected benchmark, sparse reports, incomplete normal labels, single compressor; no external validation",
        }
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src/industrialguard").glob("*.py"))
    }
    source_hashes["scripts/run_v2.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "source_sha256": source_hashes,
        "raw_sha256": raw_sha256,
        "failure_reports_sha256": hashlib.sha256(
            (ROOT / "configs/failures.csv").read_bytes()
        ).hexdigest(),
        "data_license": "CC BY 4.0",
        "schema_version": 2,
        "sklearn_version": sklearn.__version__,
        "plan_sha256": plan_hash,
        "files": {},
    }
    bundle = {"schema_version": 2, "selected_model": selected["model_id"], "models": model_bundles}
    joblib.dump(bundle, OUT / "models.joblib", compress=3)
    monitoring.to_parquet(ROOT / "data/demo/v2_monitoring.parquet")
    metrics = {
        "schema_version": 2,
        "selected_model": selected["model_id"],
        "selected_experiment": selected["experiment_id"],
        "config": config,
        "plan_sha256": plan_hash,
        "models": comparison,
        "development_fold_count": len(config["folds"]),
        "evaluation_provenance": {
            "development": "pre-May only",
            "retrospective": "May–September already inspected in V1",
            "external_validation": "Unavailable",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "pipeline_seconds": perf_counter() - started,
        "quality": {"raw_rows": quality["rows"], "common_feature_rows": len(features)},
        "regime_audit_development": regime_audit(development),
    }
    save_json(OUT / "metrics.json", metrics)
    # Development feature audit and descriptive case selection never modify frozen model decisions.
    original = available_features(development, "original")
    correlation = (
        development.loc[development.index < pd.Timestamp("2020-03-01"), original].corr().abs()
    )
    pairs = [
        (a, b, float(correlation.loc[a, b]))
        for i, a in enumerate(original)
        for b in original[i + 1 :]
        if correlation.loc[a, b] >= config["correlation_limit"]
    ]
    save_json(
        OUT / "feature_audit.json",
        {
            "training_only_correlation_limit": config["correlation_limit"],
            "high_correlation_pairs": pairs,
            "frozen_feature_groups": {
                name: available_features(features, name)
                for name in ["original", "raw", "pressure", "reduced"]
            },
        },
    )
    cases = create_cases(monitoring, failures, selected["model_id"])
    save_json(OUT / "cases.json", cases)
    for filename in [
        "models.joblib",
        "metrics.json",
        "selection.json",
        "experiments.json",
        "cases.json",
        "../../data/demo/v2_monitoring.parquet",
    ]:
        manifest["files"][filename] = hashlib.sha256((OUT / filename).read_bytes()).hexdigest()
    save_json(OUT / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "selected_model": selected["model_id"],
                "selected_experiment": selected["experiment_id"],
                "pipeline_seconds": metrics["pipeline_seconds"],
                "models": {
                    k: {
                        "development": v["development"]["detected_events"],
                        "retrospective": v["retrospective"]["detected_events"],
                        "burden": v["retrospective"]["unannotated_alerts_per_observed_day"],
                    }
                    for k, v in comparison.items()
                },
            },
            indent=2,
        )
    )


def create_cases(frame: pd.DataFrame, failures: pd.DataFrame, selected: str) -> list[dict]:
    development = frame[
        (frame.index >= pd.Timestamp("2020-04-01")) & (frame.index < pd.Timestamp("2020-05-01"))
    ]
    keep = pd.Series(True, index=development.index)
    for event in failures.itertuples():
        keep &= ~(
            (development.index >= event.start - pd.Timedelta("24h"))
            & (development.index <= event.end + pd.Timedelta("24h"))
        )
    unusual = development.loc[keep, f"{selected}_score"].idxmax()
    return [
        {
            "title": "April 18 reported air leak",
            "start": "2020-04-17",
            "end": "2020-04-19 23:59",
            "provenance": "V2 development; one reported event",
            "interpretation": "Inspect pressure, current and temperature with multiple model responses. Reported event overlap is descriptive and does not establish predictive generalization.",
        },
        {
            "title": "Unannotated unusual development period",
            "start": str(unusual - pd.Timedelta("3h")),
            "end": str(unusual + pd.Timedelta("3h")),
            "provenance": "V2 development; no documented failure within the case window",
            "interpretation": "Highest selected-model deviation outside reported intervals and their 24-hour context. Lack of a report does not prove a false alarm or healthy condition.",
        },
        {
            "title": "July 15 model-policy comparison",
            "start": "2020-07-14",
            "end": "2020-07-16 23:59",
            "provenance": "Previously inspected V1 benchmark; retrospective illustration only",
            "interpretation": "Compare frozen monitoring policies on the July reported air leak. This case was inspected only after development selection; it cannot establish independent improvement.",
        },
    ]


if __name__ == "__main__":
    main()
