"""Calibrated artifact consistency and transparent relative health behavior."""

import joblib
import numpy as np
import pandas as pd
import pytest

from industrialguard.models import attention_category, fit_models, health_score, score_frame


@pytest.fixture
def fitted():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.normal(size=(240, 4)),
        index=pd.date_range("2020-01-01", periods=240, freq="2h"),
        columns=["pressure", "temperature", "current", "difference"],
    )
    config = {
        "candidate_max_samples": [32, 64],
        "n_estimators": 10,
        "seed": 42,
        "alert_quantile": 0.995,
    }
    train, validation = frame.iloc[:120], frame.iloc[120:200]
    bundle, candidates = fit_models(train, validation, config)
    return frame, train, validation, bundle, candidates


def test_transformations_use_training_and_thresholds_use_validation(fitted):
    _, train, validation, bundle, candidates = fitted
    np.testing.assert_array_equal(bundle["center"], train.median().to_numpy())
    np.testing.assert_array_equal(
        bundle["isolation"].named_steps["scale"].center_, train.median().to_numpy()
    )
    scores = -bundle["isolation"].score_samples(validation)
    assert bundle["threshold"] == pytest.approx(np.quantile(scores, 0.995))
    assert bundle["selected_max_samples"] in [32, 64]
    assert len(candidates) == 2


def test_model_artifact_roundtrip_and_feature_order(fitted, tmp_path):
    frame, _, _, bundle, _ = fitted
    holdout = frame.iloc[200:]
    original = score_frame(holdout, bundle)
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    restored = score_frame(holdout[holdout.columns[::-1]], joblib.load(path))
    for column in ["robust_score", "isolation_score", "health", "attention", "isolation_alert"]:
        pd.testing.assert_series_equal(original[column], restored[column])
    assert original.health.between(0, 100).all()


def test_nonfinite_and_missing_model_inputs_fail(fitted):
    frame, _, _, bundle, _ = fitted
    invalid = frame.iloc[200:].copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_frame(invalid, bundle)
    with pytest.raises(KeyError):
        score_frame(frame.drop(columns="current"), bundle)


def test_relative_health_monotonic_endpoints_and_invalid_reference():
    score = health_score(np.array([-1, 1, 2, 3, 4]), np.array([3, 1, 2]))
    np.testing.assert_allclose(score, [100, 200 / 3, 100 / 3, 0, 0])
    assert (np.diff(score) <= 0).all()
    for reference in [np.array([]), np.array([np.nan])]:
        with pytest.raises(ValueError):
            health_score(np.array([1]), reference)


def test_attention_boundary_matches_strict_alert_threshold():
    labels = attention_category(np.array([0, 1, 1.1, 2, 2.1, 3, 3.1]), np.array([1, 2, 3]))
    assert list(labels) == [
        "Normal",
        "Normal",
        "Monitor",
        "Monitor",
        "Inspect",
        "Inspect",
        "High attention",
    ]
