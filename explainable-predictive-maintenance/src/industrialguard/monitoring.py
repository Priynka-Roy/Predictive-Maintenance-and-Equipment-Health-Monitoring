"""V2 training-only reference models and reproducible development selection."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

from industrialguard.data import ANALOG, SENSORS
from industrialguard.features import make_features, feature_names
from industrialguard.models import attention_category, health_score
from industrialguard.regimes import classify_regime

SCHEMA_VERSION = 2
PRESSURE = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "pressure_difference",
    "reservoir_gap",
    "pressure_variation",
    "temperature_change_15min",
    "Motor_current",
    "Oil_temperature",
    "DV_eletric",
]
REDUCED = [
    *ANALOG,
    "reservoir_gap",
    "pressure_difference",
    "pressure_variation",
    "temperature_change_15min",
    "Oil_temperature_mean_3",
    "Motor_current_std_3",
    "DV_eletric",
    "load_transition",
]


def build_v2_features(bins: pd.DataFrame) -> pd.DataFrame:
    result = make_features(bins)
    result["reservoir_gap"] = bins.TP3 - bins.Reservoirs
    result["pressure_variation"] = result.TP3_std_3
    valid = bins[SENSORS].notna().all(axis=1)
    segments = (
        (~valid)
        | (~valid).shift(fill_value=False)
        | (bins.index.to_series().diff() > pd.Timedelta("5min"))
    ).cumsum()
    result["temperature_change_15min"] = bins.Oil_temperature.groupby(segments).diff(3)
    result["regime"] = classify_regime(bins)
    return result


def available_features(frame: pd.DataFrame, feature_set: str) -> list[str]:
    original = feature_names(
        frame.drop(
            columns=["reservoir_gap", "pressure_variation", "temperature_change_15min", "regime"]
        )
    )
    groups = {"original": original, "raw": SENSORS, "pressure": PRESSURE, "reduced": REDUCED}
    if feature_set not in groups:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return list(groups[feature_set])


def robust_reference(frame: pd.DataFrame) -> dict:
    center = frame.median().to_numpy()
    scale = (frame.quantile(0.75) - frame.quantile(0.25)).to_numpy()
    scale = np.maximum(scale, np.maximum(np.abs(center) * 0.01, 0.01))
    return {"center": center, "scale": scale}


def reduced_names(train: pd.DataFrame, names: list[str], limit: float) -> list[str]:
    variable = [c for c in names if train[c].std() > 1e-8]
    corr = train[variable].corr().abs()
    kept = []
    for name in variable:
        if not any(corr.loc[name, previous] >= limit for previous in kept):
            kept.append(name)
    if not kept:
        raise ValueError("No nonconstant model features in training data")
    return kept


def fit_monitor(train: pd.DataFrame, spec: dict, config: dict) -> dict:
    started = perf_counter()
    names = available_features(train, spec["feature_set"])
    if spec["feature_set"] == "reduced":
        names = reduced_names(train, names, config["correlation_limit"])
    selected = train[names]
    if not np.isfinite(selected.to_numpy()).all() or selected.empty:
        raise ValueError("Training requires complete finite feature rows")
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "id": spec["id"],
        "model_type": spec["type"],
        "feature_set": spec["feature_set"],
        "feature_names": names,
        "min_reference_rows": config["regime_min_reference_rows"],
        **robust_reference(selected),
    }
    standardized = (selected.to_numpy() - bundle["center"]) / bundle["scale"]
    if spec["type"] == "covariance":
        # Winsorization limits reference contamination; no Gaussian chi-square limit is assumed.
        bundle["estimator"] = LedoitWolf().fit(np.clip(standardized, -8, 8))
    elif spec["type"] == "isolation":
        bundle["estimator"] = IsolationForest(
            n_estimators=config["isolation_trees"],
            max_samples=config["isolation_max_samples"],
            random_state=config["seed"],
            n_jobs=-1,
        ).fit(standardized)
    elif spec["type"] == "regime_robust":
        bundle["regime_references"] = {
            str(regime): robust_reference(block[names])
            for regime, block in train.groupby("regime")
            if len(block) >= config["regime_min_reference_rows"] and regime != "unknown"
        }
    elif spec["type"] != "robust":
        raise ValueError(f"Unknown model type {spec['type']}")
    bundle["fit_seconds"] = perf_counter() - started
    bundle["training_rows"] = len(train)
    return bundle


def feature_deviations(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    selected = frame[bundle["feature_names"]]
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError("Model inputs must be finite")
    values = np.abs((selected.to_numpy() - bundle["center"]) / bundle["scale"])
    if bundle["model_type"] == "regime_robust":
        for regime, reference in bundle["regime_references"].items():
            mask = (frame.regime == regime).fillna(False).to_numpy(dtype=bool)
            values[mask] = np.abs(
                (selected.to_numpy()[mask] - reference["center"]) / reference["scale"]
            )
    return pd.DataFrame(values, index=frame.index, columns=selected.columns)


def monitor_score(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported monitoring artifact schema")
    kind = bundle["model_type"]
    if kind in {"robust", "regime_robust"}:
        return feature_deviations(frame, bundle).max(axis=1).to_numpy()
    values = frame[bundle["feature_names"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Model inputs must be finite")
    scaled = (values - bundle["center"]) / bundle["scale"]
    if kind == "covariance":
        return bundle["estimator"].mahalanobis(scaled)
    if kind == "isolation":
        return -bundle["estimator"].score_samples(scaled)
    raise ValueError(f"Unknown model type {kind}")


def calibrate_monitor(
    bundle: dict, calibration: pd.DataFrame, quantile: float, persistence: int
) -> dict:
    if not 0 < quantile < 1 or persistence < 1:
        raise ValueError("Invalid alert policy")
    scores = monitor_score(calibration, bundle)
    bundle = dict(bundle)
    bundle.update(
        threshold=float(np.quantile(scores, quantile)),
        score_reference=np.sort(scores),
        category_thresholds=np.quantile(scores, [0.95, 0.99, 0.995]),
        quantile=quantile,
        persistence_bins=persistence,
        calibration_rows=len(calibration),
    )
    # Empirical limits can adapt to descriptive regimes; sparse regimes fall back transparently.
    bundle["regime_calibration_rows"] = {}
    if bundle["model_type"] == "regime_robust":
        bundle["regime_thresholds"] = {}
        for regime, block in calibration.groupby("regime"):
            bundle["regime_calibration_rows"][str(regime)] = len(block)
            if len(block) >= bundle["min_reference_rows"] and regime in bundle["regime_references"]:
                bundle["regime_thresholds"][str(regime)] = float(
                    np.quantile(monitor_score(block, bundle), quantile)
                )
    return bundle


def score_thresholds(frame: pd.DataFrame, bundle: dict) -> pd.Series:
    threshold = pd.Series(bundle["threshold"], index=frame.index)
    for regime, value in bundle.get("regime_thresholds", {}).items():
        threshold.loc[frame.regime == regime] = value
    return threshold


def health_indicators(scores: np.ndarray, bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    return health_score(scores, bundle["score_reference"]), attention_category(
        scores, bundle["category_thresholds"]
    )


def selection_objective(record: dict, selection: dict) -> tuple[bool, float]:
    fraction = record["issued_alert_fraction"]
    burden = record["unannotated_burden"]
    feasible = (
        selection["minimum_issued_alert_fraction"]
        <= fraction
        <= selection["maximum_alert_fraction"]
        and burden <= selection["max_unannotated_episodes_per_observed_day"]
    )
    target = selection["target_alert_fraction"]
    event_credit = (
        min(1, record["development_detected_events"]) * selection["secondary_event_credit"]
    )
    objective = (
        abs(fraction - target) / target
        + selection["stability_weight"] * record["weekly_alert_fraction_sd"] / target
        + 0.10 * burden / selection["max_unannotated_episodes_per_observed_day"]
        + selection["complexity_penalty"] * record["complexity"]
        - event_credit
    )
    return feasible, float(objective)
