"""Validate source integrity and enforce prediction-time temporal boundaries."""

import numpy as np
import pandas as pd
import pytest

from industrialguard.data import SENSORS, load_raw, resample_sensors, validate_frame
from industrialguard.features import feature_names, make_features


def test_validation_reports_missing_and_ranges_without_clipping(sensors):
    raw = sensors.reset_index()
    raw.loc[2, "Oil_temperature"] = np.nan
    raw["COMP"] = raw["COMP"].astype(float)
    raw.loc[3, "COMP"] = 0.5
    raw.loc[4, "TP2"] = -2
    quality = validate_frame(raw)
    assert quality["rows"] == 40
    assert quality["missing"]["Oil_temperature"] == 1
    assert quality["nonbinary_counts"]["COMP"] == 1
    assert quality["negative_pressure_counts"]["TP2"] == 1
    assert raw.loc[4, "TP2"] == -2


@pytest.mark.parametrize(
    "problem", ["schema", "duplicate", "reverse", "missing_time", "infinite", "empty"]
)
def test_invalid_source_rejected(sensors, problem):
    raw = sensors.reset_index()
    if problem == "schema":
        raw = raw.drop(columns="TP2")
    elif problem == "duplicate":
        raw.loc[1, "timestamp"] = raw.loc[0, "timestamp"]
    elif problem == "reverse":
        raw = raw.iloc[::-1]
    elif problem == "missing_time":
        raw.loc[0, "timestamp"] = pd.NaT
    elif problem == "infinite":
        raw.loc[0, "TP2"] = np.inf
    else:
        raw = raw.iloc[:0]
    with pytest.raises(ValueError):
        validate_frame(raw)


def test_raw_load_drops_only_extra_index_column(sensors, tmp_path):
    path = tmp_path / "raw.csv"
    sensors.reset_index().to_csv(path, index=True)
    loaded, quality = load_raw(path)
    assert list(loaded.columns) == ["timestamp", *SENSORS]
    assert quality["rows"] == len(sensors)


def test_resampling_is_right_closed_and_masks_low_coverage(sensors):
    raw = sensors.iloc[:31].copy()
    raw.index = pd.date_range("2020-01-01 00:00:10", periods=31, freq="10s", name="timestamp")
    bins = resample_sensors(raw.reset_index())
    assert bins.loc["2020-01-01 00:05", "sample_count"] == 30
    assert bins.loc["2020-01-01 00:05", "TP2"] == 14.5
    assert bins.loc["2020-01-01 00:10", SENSORS].isna().all()
    assert bins.loc["2020-01-01 00:05", "COMP"] == 0.5


def test_features_are_causal_and_match_hand_calculation(sensors):
    actual = make_features(sensors)
    changed = sensors.copy()
    changed.iloc[20:] = changed.iloc[20:] + 1000
    pd.testing.assert_frame_equal(actual.iloc[:20], make_features(changed).iloc[:20])
    assert actual.iloc[2].TP2_mean_3 == 1
    assert actual.iloc[2].TP2_std_3 == pytest.approx(np.sqrt(2 / 3))
    assert actual.iloc[3].TP2_change == 1
    assert actual.iloc[3].pressure_difference == -1
    assert "operating_state" not in feature_names(actual)


def test_feature_windows_restart_after_telemetry_gap(sensors):
    disconnected = sensors.drop(index=sensors.index[8:10])
    features = make_features(disconnected)
    assert pd.isna(features.loc[sensors.index[10], "TP2_change"])
    assert pd.isna(features.loc[sensors.index[11], "TP2_mean_3"])
    assert features.loc[sensors.index[12], "TP2_mean_3"] == 11


def test_all_feature_windows_restart_after_incomplete_bin(sensors):
    incomplete = sensors.copy()
    incomplete.loc[sensors.index[5], "Oil_temperature"] = np.nan
    features = make_features(incomplete)
    assert pd.isna(features.loc[sensors.index[6], "TP2_change"])
    assert pd.isna(features.loc[sensors.index[7], "TP2_mean_3"])
    assert features.loc[sensors.index[8], "TP2_mean_3"] == 7
