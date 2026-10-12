"""Training-only robust scaling and isolation scoring; validation calibration."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def fit_models(train: pd.DataFrame, validation: pd.DataFrame, config: dict) -> tuple[dict, list]:
    names = list(train.columns)
    center = train.median().to_numpy()
    # IQR floor avoids divisions by zero for stable digital signals and near-constant features.
    scale = (train.quantile(0.75) - train.quantile(0.25)).to_numpy()
    scale = np.maximum(scale, np.maximum(np.abs(center) * 0.01, 0.01))
    bundle = {"feature_names": names, "center": center, "scale": scale}
    candidates = []
    fitted = []
    for samples in config["candidate_max_samples"]:
        pipeline = Pipeline(
            [
                ("scale", RobustScaler()),
                (
                    "model",
                    IsolationForest(
                        n_estimators=config["n_estimators"],
                        max_samples=samples,
                        random_state=config["seed"],
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        started = perf_counter()
        pipeline.fit(train)
        score = -pipeline.score_samples(validation)
        threshold = float(np.quantile(score, config["alert_quantile"]))
        # Stability objective measured across contiguous validation calendar weeks;
        # label-free thresholding, never using test failures for tuning.
        flags = pd.Series(score > threshold, index=validation.index)
        weekly = flags.resample("7D").mean()
        objective = float(weekly.std(ddof=0))
        candidates.append(
            {
                "max_samples": samples,
                "fit_seconds": perf_counter() - started,
                "validation_weekly_alert_rate_std": objective,
                "threshold": threshold,
            }
        )
        fitted.append(pipeline)
    best = int(np.argmin([c["validation_weekly_alert_rate_std"] for c in candidates]))
    bundle["isolation"] = fitted[best]
    bundle["selected_max_samples"] = candidates[best]["max_samples"]
    robust = robust_score(validation, bundle)
    isolation = -bundle["isolation"].score_samples(validation)
    bundle["thresholds"] = {
        "robust": float(np.quantile(robust, config["alert_quantile"])),
        "isolation": float(np.quantile(isolation, config["alert_quantile"])),
    }
    bundle["threshold"] = bundle["thresholds"]["isolation"]
    bundle["score_reference"] = np.sort(isolation)
    bundle["category_thresholds"] = np.quantile(isolation, [0.95, 0.99, config["alert_quantile"]])
    bundle["seed"] = config["seed"]
    return bundle, candidates


def deviations(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    selected = frame[bundle["feature_names"]]
    return pd.DataFrame(
        np.abs((selected.to_numpy() - bundle["center"]) / bundle["scale"]),
        columns=selected.columns,
        index=selected.index,
    )


def robust_score(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    return deviations(frame, bundle).max(axis=1).to_numpy()


def health_score(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """100 minus validation percentile: relative typicality, never failure probability."""
    if len(reference) == 0 or not np.isfinite(reference).all():
        raise ValueError("Health calibration requires finite validation scores")
    return 100 * (1 - np.searchsorted(np.sort(reference), scores, side="right") / len(reference))


def attention_category(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.array(["Normal", "Monitor", "Inspect", "High attention"])[
        np.searchsorted(thresholds, scores, side="left")
    ]


def score_frame(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    selected = frame[bundle["feature_names"]]
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError("Model inputs must be finite; exclude incomplete feature windows")
    result = frame.copy()
    result["robust_score"] = robust_score(selected, bundle)
    result["isolation_score"] = -bundle["isolation"].score_samples(selected)
    for name, threshold in bundle["thresholds"].items():
        result[f"{name}_alert"] = result[f"{name}_score"] > threshold
    result["health"] = health_score(result.isolation_score.to_numpy(), bundle["score_reference"])
    result["attention"] = attention_category(
        result.isolation_score.to_numpy(), bundle["category_thresholds"]
    )
    return result
