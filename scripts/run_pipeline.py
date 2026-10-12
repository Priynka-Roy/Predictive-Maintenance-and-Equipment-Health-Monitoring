"""Reproduce preparation, EDA, feature engineering, training, evaluation and demo."""

import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from industrialguard.data import ANALOG, SENSORS, UNITS, load_raw, resample_sensors
from industrialguard.evaluation import alert_episodes, evaluate, persistent_alerts
from industrialguard.features import feature_names, make_features
from industrialguard.models import fit_models, score_frame


matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))


def eda(raw, bins, quality):
    out = ROOT / "reports/figures"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for sensor, ax in zip(ANALOG, axes.flat):
        ax.hist(raw[sensor], bins=80, color="#13a89e")
        ax.set(title=sensor, xlabel=UNITS[sensor], ylabel="Observations")
    for ax in list(axes.flat)[len(ANALOG) :]:
        ax.axis("off")
    fig.suptitle("MetroPT-3 • original sensor distributions")
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=150)
    plt.close(fig)
    correlation = bins[SENSORS].corr()
    correlation.to_csv(ROOT / "reports/correlation.csv")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set(
        xticks=range(len(SENSORS)),
        yticks=range(len(SENSORS)),
        xticklabels=SENSORS,
        yticklabels=SENSORS,
        title="Five-minute sensor correlations",
    )
    plt.setp(ax.get_xticklabels(), rotation=75, ha="right")
    fig.colorbar(im, ax=ax, label="Pearson correlation (association)")
    fig.tight_layout()
    fig.savefig(out / "correlations.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for ax, sensor in zip(axes, ["TP3", "Motor_current", "Oil_temperature"]):
        ax.plot(bins.index, bins[sensor], lw=0.4, color="#168c91")
        ax.set(ylabel=f"{sensor} ({UNITS[sensor]})")
    axes[0].set_title("Historical sensor behavior • gaps remain unfilled")
    fig.tight_layout()
    fig.savefig(out / "temporal.png", dpi=150)
    plt.close(fig)
    quality["resampled_bins"] = len(bins)
    quality["unusable_resampled_bins"] = int(bins[SENSORS].isna().any(axis=1).sum())
    save_json(ROOT / "reports/data_quality.json", quality)


def main():
    started = perf_counter()
    config = json.loads((ROOT / "configs/pipeline.json").read_text())
    failures = pd.read_csv(ROOT / "configs/failures.csv", parse_dates=["start", "end"])
    paths = list((ROOT / "data/raw").glob("*.csv"))
    if len(paths) != 1:
        raise RuntimeError("Run python scripts/download.py first; expected one official raw CSV")
    raw, quality = load_raw(paths[0])
    bins = resample_sensors(raw, config["resample"])
    eda(raw, bins, quality)
    engineered = make_features(bins, tuple(config["rolling_windows"]))
    names = feature_names(engineered)
    features = engineered.dropna(subset=names)
    train_end, validation_end = (
        pd.Timestamp(config["train_end"]),
        pd.Timestamp(config["validation_end"]),
    )
    train = features.loc[features.index < train_end, names]
    val = features.loc[(features.index >= train_end) & (features.index < validation_end), names]
    reference = pd.Series(True, index=val.index)
    for event in failures.itertuples():
        reference &= ~((val.index >= event.start) & (val.index <= event.end))
    bundle, candidates = fit_models(train, val[reference], config)
    bundle["training_period"] = [str(train.index.min()), str(train.index.max())]
    bundle["validation_period"] = [str(val.index.min()), str(val.index.max())]
    scored = score_frame(features, bundle)
    for name in ["robust", "isolation"]:
        scored[f"{name}_alert"] = persistent_alerts(
            scored[f"{name}_alert"], config["persistence_bins"]
        )
    scored["documented_failure"] = False
    for event in failures.itertuples():
        scored.loc[
            (scored.index >= event.start) & (scored.index <= event.end), "documented_failure"
        ] = True
    metrics = {
        "selected_model": "isolation",
        "selection_basis": "Minimum validation weekly alert-rate variability among two Isolation Forest candidates; baseline retained for comparison. No test-based selection.",
        "config": config,
        "candidates": candidates,
        "feature_count": len(names),
        "thresholds": bundle["thresholds"],
        "partitions": {},
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "calibration_reference": "March-April validation excluding UCI documented failure intervals; unannotated observations are not certified healthy.",
    }
    partitions = {
        "train": scored[scored.index < train_end],
        "validation": scored[(scored.index >= train_end) & (scored.index < validation_end)],
        "test": scored[scored.index >= validation_end],
    }
    for split, frame in partitions.items():
        metrics["partitions"][split] = {
            "rows": len(frame),
            "start": str(frame.index.min()),
            "end": str(frame.index.max()),
            **{
                name: evaluate(frame[f"{name}_alert"], failures, config["lead_window_hours"])
                for name in ["robust", "isolation"]
            },
        }
    metrics["sensitivity"] = []
    # Validation-only sensitivity table; test period never revisited for alternatives.
    val_scored = scored.loc[val.index]
    val_reference_scores = val_scored.loc[reference, "isolation_score"]
    for quantile in [0.98, 0.99, 0.995, 0.999]:
        for persistence in [1, 3, 6]:
            threshold = float(val_reference_scores.quantile(quantile))
            flags = persistent_alerts(val_scored.isolation_score > threshold, persistence)
            metrics["sensitivity"].append(
                {
                    "quantile": quantile,
                    "persistence_bins": persistence,
                    "threshold": threshold,
                    **evaluate(flags, failures, config["lead_window_hours"]),
                }
            )
    metrics["test_months"] = {
        str(month): {
            name: evaluate(frame[f"{name}_alert"], failures, config["lead_window_hours"])
            for name in ["robust", "isolation"]
        }
        for month, frame in partitions["test"].groupby(partitions["test"].index.to_period("M"))
    }
    timing = perf_counter()
    score_frame(features.iloc[-1000:], bundle)
    metrics["inference_1000_rows_seconds"] = perf_counter() - timing
    metrics["pipeline_seconds"] = perf_counter() - started
    quality["usable_feature_rows"] = len(scored)
    quality["feature_rows_lost_to_gaps_or_warmup"] = len(bins) - len(scored)
    save_json(ROOT / "reports/data_quality.json", quality)
    save_json(ROOT / "artifacts/metrics.json", metrics)
    joblib.dump(bundle, ROOT / "artifacts/model.joblib", compress=3)
    scored.to_parquet(ROOT / "data/demo/monitoring.parquet")
    engineered.to_parquet(ROOT / "data/processed/features.parquet")
    for name in ["robust", "isolation"]:
        alert_episodes(scored[f"{name}_alert"]).to_csv(
            ROOT / f"reports/{name}_alerts.csv", index=False
        )
    failures.to_csv(ROOT / "data/demo/failures.csv", index=False)
    write_report(quality, bins, scored, metrics, failures)
    print(
        json.dumps(
            {
                "quality": {k: v for k, v in quality.items() if k != "sensor_summary"},
                "test": metrics["partitions"]["test"],
                "pipeline_seconds": metrics["pipeline_seconds"],
            },
            indent=2,
        )
    )


def write_report(quality, bins, scored, metrics, failures):
    states = scored.operating_state.value_counts(normalize=True).round(4).to_dict()
    text = f"""# Reproducible statistical EDA report

Generated from the original UCI CSV by `python scripts/run_pipeline.py`.

## Data audit

- Original records: {quality["rows"]:,}; period: {quality["start"]} through {quality["end"]}.
- Median observed interval: {quality["median_interval_seconds"]} seconds; gaps >60 seconds: {quality["gaps_over_60_seconds"]}.
- Missing values by sensor: {quality["missing"]}.
- Digital values outside 0/1: {quality["nonbinary_counts"]}.
- Negative pressure observations: {quality["negative_pressure_counts"]}. These may include sensor offset effects; retained, not labelled failures.
- Five-minute bins: {quality["resampled_bins"]:,}; bins lacking 15 records or complete sensor values: {quality["unusable_resampled_bins"]:,}.
- Usable causal feature rows: {len(scored):,}. Empty bins are not interpolated; windows reset at outages.

## Sensor patterns and regimes

Motor current regime fractions among scored five-minute means: {states}. Thresholds (<1 A off, 1–5.5 A offloaded, >5.5 A loaded) approximate the documented current levels. Mixed-state bins blur instantaneous regimes; these are descriptive categories, not verified machine modes.

Pressure channels co-move; digital states and current reflect compressor load cycling. The CSV includes negative pressure readings and strong distribution asymmetry. A global deviation model can confuse legitimate regime changes and seasonal temperature changes with anomalies. The heatmap in `correlations.png` documents association; repeated time-series rows invalidate independent-observation significance tests.

## Failure-aligned inspection

"""
    for event in failures.itertuples():
        local = bins.loc[event.start - pd.Timedelta("24h") : event.end + pd.Timedelta("24h")]
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for ax, sensor in zip(axes, ["TP3", "Oil_temperature", "Motor_current"]):
            ax.plot(local.index, local[sensor], lw=0.8, color="#168c91")
            ax.axvspan(
                event.start, event.end, alpha=0.15, color="#e98938", label="UCI reported interval"
            )
            ax.set(ylabel=f"{sensor} ({UNITS[sensor]})")
            ax.legend(loc="upper right")
        axes[0].set_title(f"Reported air leak {event.event_id} • observed sensor context")
        fig.tight_layout()
        fig.savefig(ROOT / f"reports/figures/failure_{event.event_id}.png", dpi=150)
        plt.close(fig)
        text += f"\nEvent {event.event_id} ({event.start}–{event.end}): {len(local.dropna()):,} observed bins in the ±24-hour context. See `figures/failure_{event.event_id}.png`. No causal diagnosis is inferred from the patterns.\n"
    text += (
        "\n## Computed chronological test results\n\n```json\n"
        + json.dumps(metrics["partitions"]["test"], indent=2)
        + "\n```\n"
    )
    text += "\nUnannotated alerts per observed day are a proxy for false-alert burden; incomplete failure annotation prevents verified false-positive rates or alert precision. Only three test events exist, so no narrow confidence interval is defensible. Warnings in the 24-hour preceding window can be coincidental, particularly under high alert burden. No RUL or calibrated failure probabilities are estimated.\n"
    (ROOT / "reports/EDA.md").write_text(text)


if __name__ == "__main__":
    main()
