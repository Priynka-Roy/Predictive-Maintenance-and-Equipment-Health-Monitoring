"""Hand-calculated event overlaps, warning times and observed alert exposure."""

import pandas as pd
import pytest

from industrialguard.evaluation import alert_episodes, evaluate, persistent_alerts


def test_persistence_requires_consecutive_bins_and_resets_across_gap():
    index = pd.to_datetime(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:05",
            "2020-01-01 00:10",
            "2020-01-01 00:30",
            "2020-01-01 00:35",
            "2020-01-01 00:40",
        ]
    )
    flags = pd.Series(True, index=index)
    assert persistent_alerts(flags, 3).tolist() == [False, False, True, False, False, True]
    flags.iloc[1] = False
    assert not persistent_alerts(flags, 3).iloc[2]


def test_episode_boundaries_break_on_gap_and_negative_bin():
    index = pd.to_datetime(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:05",
            "2020-01-01 00:10",
            "2020-01-01 00:15",
            "2020-01-01 00:40",
        ]
    )
    episodes = alert_episodes(pd.Series([True, True, False, True, True], index=index))
    assert episodes.bins.tolist() == [2, 1, 1]
    assert episodes.start.iloc[2] == index[-1]


def test_event_overlap_lead_time_and_unannotated_burden():
    index = pd.date_range("2020-01-01", periods=13, freq="5min")
    flags = pd.Series(False, index=index)
    flags.iloc[[1, 4, 5, 10]] = True
    failures = pd.DataFrame({"event_id": [1], "start": [index[5]], "end": [index[7]]})
    metrics = evaluate(flags, failures, lead_hours=1)
    assert metrics["event_count"] == metrics["detected_events"] == 1
    assert metrics["event_detection_rate"] == 1
    assert metrics["events"][0]["lead_time_hours"] == pytest.approx(20 / 60)
    assert metrics["alert_episodes"] == 3
    assert metrics["unannotated_alert_episodes"] == 2
    assert metrics["unannotated_observed_days"] == pytest.approx(10 * 5 / 1440)
    assert metrics["unannotated_alerts_per_observed_day"] == pytest.approx(57.6)


def test_boundary_events_excluded_and_no_event_rate_is_undefined():
    index = pd.date_range("2020-01-01", periods=5, freq="5min")
    failures = pd.DataFrame(
        {
            "event_id": [1, 2],
            "start": [index[0] - pd.Timedelta("5min"), index[-1]],
            "end": [index[1], index[-1] + pd.Timedelta("5min")],
        }
    )
    metrics = evaluate(pd.Series(False, index=index), failures)
    assert metrics["event_count"] == 0
    assert metrics["event_detection_rate"] is None
    assert metrics["unannotated_alert_episodes"] == 0


def test_alert_denominator_counts_observed_bins_not_outage_wall_time():
    index = pd.to_datetime(["2020-01-01", "2020-01-03"])
    failures = pd.DataFrame(columns=["event_id", "start", "end"])
    metrics = evaluate(pd.Series([True, True], index=index), failures)
    assert metrics["unannotated_observed_days"] == pytest.approx(10 / 1440)
    assert metrics["alert_episodes"] == 2


def test_event_entirely_inside_outage_is_unobservable_not_a_missed_detection():
    index = pd.to_datetime(["2020-01-01", "2020-01-03"])
    failures = pd.DataFrame({"event_id": [1], "start": ["2020-01-02"], "end": ["2020-01-02 01:00"]})
    metrics = evaluate(pd.Series(False, index=index), failures)
    assert metrics["event_count"] == 0
    assert metrics["unobservable_events"] == 1
    assert metrics["event_detection_rate"] is None
    assert metrics["events"][0]["observed_bins"] == 0


def test_boundary_event_still_excluded_from_unannotated_alert_burden():
    index = pd.date_range("2020-01-01", periods=5, freq="5min")
    failures = pd.DataFrame(
        {"event_id": [1], "start": [index[0] - pd.Timedelta("5min")], "end": [index[1]]}
    )
    flags = pd.Series([True, True, False, False, False], index=index)
    metrics = evaluate(flags, failures)
    assert metrics["event_count"] == 0
    assert metrics["unannotated_alert_episodes"] == 0
    assert metrics["unannotated_observed_days"] == pytest.approx(3 * 5 / 1440)
