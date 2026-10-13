"""Boundary ownership, causal alerts and sparse telemetry."""

import json

import pandas as pd
import pytest

from industrialguard.evaluation import alert_policy, evaluate_v2


def failures(*intervals):
    return pd.DataFrame(
        [(i + 1, a, b) for i, (a, b) in enumerate(intervals)], columns=["event_id", "start", "end"]
    )


def test_global_onset_ownership_crosses_month_once():
    index = pd.date_range("2020-01-31 23:45", periods=8, freq="5min")
    flags = pd.Series([False, True, True, True, True, False, True, False], index=index)
    events = failures((index[2], index[4]))
    jan = evaluate_v2(flags, events, start="2020-01-01", end="2020-02-01")
    feb = evaluate_v2(flags, events, start="2020-02-01", end="2020-03-01")
    assert jan["assigned_event_count"] == jan["detected_events"] == 1
    assert jan["events"][0]["observed_bins"] == 3
    assert feb["assigned_event_count"] == 0
    assert jan["alert_episodes"] == feb["alert_episodes"] == 1
    assert feb["crossing_onset_episodes"] == 1
    assert jan["unannotated_episode_count"] == 0
    assert feb["unannotated_episode_count"] == 1
    json.dumps(jan, allow_nan=False)


def test_warning_definitions_exclude_other_failures():
    index = pd.date_range("2020-01-01", periods=13, freq="5min")
    flags = pd.Series(False, index=index)
    flags.iloc[[1, 2, 4, 6, 7, 9, 10]] = True
    events = failures((index[1], index[2]), (index[10], index[12]))
    result = evaluate_v2(flags, events, lead_hours=1)["events"][1]
    assert result["earliest_warning_lead_hours"] == pytest.approx(30 / 60)
    assert result["nearest_warning_lead_hours"] == pytest.approx(5 / 60)
    assert result["episode_onset_warning_lead_hours"] == pytest.approx(30 / 60)


def test_episode_onset_outside_warning_window_is_not_warning_onset():
    index = pd.date_range("2020-01-01", periods=20, freq="5min")
    result = evaluate_v2(
        pd.Series(True, index=index), failures((index[15], index[18])), lead_hours=1
    )["events"][0]
    assert result["earliest_warning_lead_hours"] == 1
    assert result["nearest_warning_lead_hours"] == pytest.approx(5 / 60)
    assert result["episode_onset_warning_lead_hours"] is None


def test_missing_outage_and_partial_coverage():
    index = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:05", "2020-01-03 00:00"])
    flags = pd.Series([False, None, True], index=index, dtype="boolean")
    events = failures(("2020-01-01", "2020-01-01 00:10"), ("2020-01-02", "2020-01-02 01:00"))
    result = evaluate_v2(flags, events)
    assert result["event_count"] == 1
    assert result["unobservable_events"] == result["partial_events"] == 1
    assert result["observed_days"] == pytest.approx(10 / 1440)
    assert result["calendar_days"] > 2
    assert result["events"][0]["coverage_fraction"] == pytest.approx(1 / 3)


def test_zero_telemetry_is_json_serializable():
    flags = pd.Series([], index=pd.DatetimeIndex([]), dtype=bool)
    result = evaluate_v2(
        flags, failures(("2020-01-01", "2020-01-02")), start="2020-01-01", end="2020-02-01"
    )
    assert result["unobservable_events"] == 1
    assert result["event_detection_rate"] is None
    assert result["alert_bin_fraction"] is None
    json.dumps(result, allow_nan=False)


def test_causal_hysteresis_persistence_and_missing_reset():
    index = pd.date_range("2020-01-01", periods=9, freq="5min")
    scores = pd.Series([4, 4, 2, 4, 4, None, 4, 4, 0], index=index)
    assert alert_policy(scores, 3, 2, 1).tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        False,
    ]
    assert alert_policy(scores, 3, 2).tolist() == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
    ]
    for end in range(1, len(scores)):
        assert alert_policy(scores.iloc[:end], 3, 2, 1).equals(
            alert_policy(scores, 3, 2, 1).iloc[:end]
        )


def test_gap_and_missing_threshold_reset_state():
    index = pd.to_datetime(
        ["2020-01-01 00:00", "2020-01-01 00:05", "2020-01-01 00:30", "2020-01-01 00:35"]
    )
    scores = pd.Series(4.0, index=index)
    assert alert_policy(scores, 3, 2).tolist() == [False, True, False, True]
    thresholds = pd.Series([3, 3, None, 3], index=index)
    assert alert_policy(scores, thresholds, 2).tolist() == [False, True, False, False]
    with pytest.raises(ValueError):
        alert_policy(scores, 3, 0)
    with pytest.raises(ValueError):
        alert_policy(scores, 3, 2, 4)


def test_alerts_only_during_prior_failure_never_qualify_as_warnings():
    index = pd.date_range("2020-01-01", periods=13, freq="5min")
    flags = pd.Series(False, index=index)
    flags.iloc[1:4] = True
    result = evaluate_v2(flags, failures((index[1], index[3]), (index[10], index[12])))
    warning = result["events"][1]
    assert warning["earliest_warning_lead_hours"] is None
    assert warning["nearest_warning_lead_hours"] is None
    assert warning["episode_onset_warning_lead_hours"] is None


def test_event_exactly_at_exclusive_end_belongs_to_next_partition():
    index = pd.date_range("2020-01-01", periods=6, freq="5min")
    flags = pd.Series(False, index=index)
    events = failures((index[3], index[4]))
    left = evaluate_v2(flags, events, start=index[0], end=index[3])
    right = evaluate_v2(flags, events, start=index[3], end=index[-1])
    assert left["assigned_event_count"] == 0
    assert right["assigned_event_count"] == 1


def test_missing_flags_break_episodes_and_exclude_exposure():
    index = pd.date_range("2020-01-01", periods=3, freq="5min")
    flags = pd.Series([True, None, True], index=index, dtype="boolean")
    result = evaluate_v2(flags, failures())
    assert result["alert_episodes"] == 2
    assert result["unannotated_observed_days"] == pytest.approx(10 / 1440)
    assert result["event_detection_rate"] is None


def test_entire_other_failure_episode_excluded_even_after_failure_ends():
    index = pd.date_range("2020-01-01", periods=13, freq="5min")
    flags = pd.Series(False, index=index)
    flags.iloc[1:9] = True
    result = evaluate_v2(flags, failures((index[2], index[3]), (index[10], index[12])))
    warning = result["events"][1]
    assert warning["earliest_warning_lead_hours"] is None
    assert warning["nearest_warning_lead_hours"] is None
    assert warning["episode_onset_warning_lead_hours"] is None
