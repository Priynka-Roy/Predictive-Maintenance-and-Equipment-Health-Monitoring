"""Application artifact validation and all eight workspaces with real demo outputs."""

import json
from pathlib import Path

import numpy as np

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.main import PAGES, alert_episodes, load_monitoring, timeseries

ROOT = Path(__file__).resolve().parents[1]


def test_monitoring_loader_rejects_invalid_artifacts(tmp_path):
    frame = pd.DataFrame({"x": [1]}, index=pd.date_range("2020-01-01", periods=1))
    path = tmp_path / "missing.parquet"
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="Missing monitoring columns"):
        load_monitoring(str(path), path.stat().st_mtime)
    path = tmp_path / "no_time.parquet"
    frame.reset_index(drop=True).to_parquet(path)
    with pytest.raises(ValueError, match="timestamp"):
        load_monitoring(str(path), path.stat().st_mtime)


def test_ui_alert_episode_summary_obeys_gap_boundary():
    index = pd.to_datetime(["2020-01-01", "2020-01-01 00:05", "2020-01-02"], format="mixed")
    frame = pd.DataFrame({"isolation_alert": [True, True, True], "health": [5, 2, 8]}, index=index)
    result = alert_episodes(frame)
    assert result.Observations.tolist() == [2, 1]
    assert result["Lowest relative health"].tolist() == [2, 8]


def test_chart_sensor_labels_include_units(sensors):
    figure = timeseries(sensors, ["TP2", "Motor_current"], "Test chart")
    assert [trace.name for trace in figure.data] == ["TP2 (bar)", "Motor_current (A)"]
    assert "timezone unspecified" in figure.layout.xaxis.title.text


def test_complete_streamlit_application_all_eight_sections():
    for relative in [
        "data/demo/monitoring.parquet",
        "artifacts/model.joblib",
        "artifacts/metrics.json",
    ]:
        assert (ROOT / relative).is_file(), f"Missing deployment artifact: {relative}"
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45).run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 4
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {list(app.exception)}"
        assert not app.error, f"{page}: {list(app.error)}"
        assert app.title[0].value == page
        if page != "About the Project":
            assert len(app.get("plotly_chart")) > 0 or page == "Maintenance Decision Support"
        if page == "Explainable AI":
            assert len(app.dataframe) > 0
            assert not app.warning
        if page == "Model Performance":
            assert len(app.dataframe) >= 2
            expected = json.loads((ROOT / "artifacts/metrics.json").read_text())["partitions"][
                "test"
            ]
            comparison = app.dataframe[0].value.set_index("model")
            assert set(comparison.index) == {"robust", "isolation"}
            for name in comparison.index:
                assert comparison.loc[name, "detected_events"] == expected[name]["detected_events"]
                assert comparison.loc[name, "event_count"] == expected[name]["event_count"]
                assert comparison.loc[name, "unannotated_alerts_per_observed_day"] == pytest.approx(
                    expected[name]["unannotated_alerts_per_observed_day"]
                )
    app.sidebar.radio[0].set_value("Sensor Explorer").run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Select at least one sensor" in item.value for item in app.info)


@pytest.mark.parametrize("problem", ["duplicate", "nonfinite", "health_range"])
def test_monitoring_loader_rejects_invalid_scores_and_duplicate_timestamps(tmp_path, problem):
    frame = pd.read_parquet(ROOT / "data/demo/monitoring.parquet").iloc[:3].copy()
    if "timestamp" in frame:
        frame = frame.set_index("timestamp")
    if problem == "duplicate":
        frame.index = [frame.index[0]] * len(frame)
    elif problem == "nonfinite":
        frame.iloc[0, frame.columns.get_loc("isolation_score")] = np.inf
    else:
        frame.iloc[0, frame.columns.get_loc("health")] = 101
    path = tmp_path / f"{problem}.parquet"
    frame.to_parquet(path)
    with pytest.raises(ValueError):
        load_monitoring(str(path), path.stat().st_mtime)
