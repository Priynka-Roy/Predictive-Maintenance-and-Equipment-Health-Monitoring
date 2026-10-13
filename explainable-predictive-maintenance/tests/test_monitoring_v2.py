"""Hand-calculated, causal, reproducible V2 modeling and policy-selection checks."""

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from industrialguard.data import SENSORS
from industrialguard.monitoring import (
    available_features,
    build_v2_features,
    calibrate_monitor,
    feature_deviations,
    fit_monitor,
    monitor_score,
    reduced_names,
    selection_objective,
)
from industrialguard.regimes import classify_regime

ROOT = Path(__file__).resolve().parents[1]


def frame():
    rng = np.random.default_rng(42)
    index = pd.date_range("2020-02-01", periods=160, freq="5min")
    data = pd.DataFrame({c: rng.normal(5, 1, len(index)) for c in SENSORS}, index=index)
    data["Motor_current"] = 7.0
    data["COMP"] = 0.0
    data["DV_eletric"] = 1.0
    data["sample_count"] = 30
    return data


def config():
    return json.loads((ROOT / "configs/v2.json").read_text())


def test_regime_requires_control_agreement_and_retains_ambiguity():
    data = pd.DataFrame(
        {
            "Motor_current": [0, 4, 7, 4, 7, 0, np.nan],
            "COMP": [1, 1, 0, 0.5, 1, 0, 1],
            "DV_eletric": [0, 0, 1, 0.5, 0, 1, 0],
        }
    )
    assert classify_regime(data).tolist() == [
        "off",
        "unloaded",
        "loaded",
        "transition",
        "ambiguous",
        "ambiguous",
        "unknown",
    ]


def test_v2_features_are_causal_and_restart_across_missing_time():
    data = frame()
    baseline = build_v2_features(data)
    changed = data.copy()
    changed.loc[changed.index[100] :, "Oil_temperature"] += 100
    pd.testing.assert_frame_equal(baseline.iloc[:100], build_v2_features(changed).iloc[:100])
    broken = data.drop(data.index[50:55])
    features = build_v2_features(broken)
    assert pd.isna(features.loc[data.index[55], "temperature_change_15min"])
    assert pd.isna(features.loc[data.index[57], "temperature_change_15min"])
    assert np.isfinite(features.loc[data.index[58], "temperature_change_15min"])


def test_original_representation_retains_exact_v1_names():
    features = build_v2_features(frame())
    assert len(available_features(features, "original")) == 52
    assert "regime" not in available_features(features, "original")
    assert "reservoir_gap" in available_features(features, "pressure")


def test_reduced_representation_removes_training_correlations_and_constants():
    data = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [0, 1, 0, 1], "constant": [1] * 4}
    )
    assert reduced_names(data, list(data), 0.98) == ["a", "c"]


@pytest.mark.parametrize("kind", ["robust", "covariance", "regime_robust", "isolation"])
def test_models_fit_train_only_roundtrip_and_finite_scores(kind, tmp_path):
    features = build_v2_features(frame()).dropna()
    train = features.iloc[:90]
    cal = features.iloc[90:]
    spec = {"id": "test", "type": kind, "feature_set": "pressure"}
    cfg = config()
    cfg["isolation_trees"] = 15
    cfg["isolation_max_samples"] = 64
    bundle = fit_monitor(train, spec, cfg)
    center = bundle["center"].copy()
    calibrated = calibrate_monitor(bundle, cal, 0.98, 1)
    np.testing.assert_equal(center, calibrated["center"])
    np.testing.assert_allclose(center, train[bundle["feature_names"]].median().to_numpy())
    scores = monitor_score(cal, calibrated)
    assert np.isfinite(scores).all()
    assert calibrated["threshold"] == pytest.approx(np.quantile(scores, 0.98))
    path = tmp_path / "model.joblib"
    joblib.dump(calibrated, path)
    np.testing.assert_allclose(scores, monitor_score(cal, joblib.load(path)))
    repeated = fit_monitor(train, spec, cfg)
    np.testing.assert_allclose(scores, monitor_score(cal, repeated))


def test_regime_reference_falls_back_for_unseen_regime():
    features = build_v2_features(frame()).dropna()
    spec = {"id": "regime", "type": "regime_robust", "feature_set": "pressure"}
    bundle = fit_monitor(features, spec, config())
    unseen = features.iloc[:2].copy()
    unseen["regime"] = "unknown"
    actual = feature_deviations(unseen, bundle)
    expected = np.abs(
        (unseen[bundle["feature_names"]].to_numpy() - bundle["center"]) / bundle["scale"]
    )
    np.testing.assert_allclose(actual, expected)


def test_selection_rejects_silence_and_event_credit_is_bounded():
    selection = config()["selection"]
    silent = {
        "issued_alert_fraction": 0,
        "unannotated_burden": 0,
        "weekly_alert_fraction_sd": 0,
        "development_detected_events": 0,
        "complexity": 1,
    }
    assert not selection_objective(silent, selection)[0]
    useful = copy.deepcopy(silent)
    useful.update(issued_alert_fraction=0.01, unannotated_burden=1)
    assert selection_objective(useful, selection)[0]
    objective = selection_objective(useful, selection)[1]
    useful["development_detected_events"] = 99
    assert objective - selection_objective(useful, selection)[1] == pytest.approx(0.05)


def test_unsupported_artifact_version_and_invalid_inputs_fail_helpfully():
    features = build_v2_features(frame()).dropna()
    bundle = fit_monitor(features, {"id": "test", "type": "robust", "feature_set": "raw"}, config())
    bundle["schema_version"] = 99
    with pytest.raises(ValueError, match="schema"):
        monitor_score(features, bundle)


def test_v2_artifacts_configs_selection_and_scores_match_when_available():
    metrics_path = ROOT / "artifacts/v2/metrics.json"
    if not metrics_path.exists():
        pytest.skip("V2 analytical run has not completed")
    metrics = json.loads(metrics_path.read_text())
    assert metrics["config"] == config()
    assert all(
        pd.Timestamp(fold["evaluation_end"]) <= pd.Timestamp(metrics["config"]["benchmark_start"])
        for fold in config()["folds"]
    )
    bundle = joblib.load(ROOT / "artifacts/v2/models.joblib")
    data = pd.read_parquet(ROOT / "data/demo/v2_monitoring.parquet").iloc[::2000]
    for model_id, model in bundle["models"].items():
        np.testing.assert_allclose(monitor_score(data, model), data[f"{model_id}_score"])
    assert json.loads((ROOT / "artifacts/v1/metrics.json").read_text()) == json.loads(
        (ROOT / "artifacts/metrics.json").read_text()
    )


def test_invalid_control_fractions_are_unknown_not_stable_modes():
    data = pd.DataFrame({"Motor_current": [0, 7, -1], "COMP": [2, 0, 1], "DV_eletric": [0, -1, 0]})
    assert classify_regime(data).tolist() == ["unknown", "unknown", "unknown"]
