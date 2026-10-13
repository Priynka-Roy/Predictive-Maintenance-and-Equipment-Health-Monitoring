"""V2 presentation adapters must preserve model identity and artifact integrity."""

import json

import numpy as np
import pandas as pd
import pytest

from app.main import ROOT, load_model, load_monitoring, select_model_frame


def test_model_selection_uses_all_outputs_from_same_model():
    frame = pd.DataFrame(
        {
            "robust_score": [9.0],
            "isolation_score": [2.0],
            "chosen_score": [7.0],
            "chosen_alert": [True],
            "chosen_health": [1.0],
            "chosen_attention": ["Inspect"],
        }
    )
    selected = select_model_frame(frame, "chosen")
    assert selected.isolation_score.iloc[0] == 7.0
    assert selected.isolation_alert.iloc[0]
    assert selected.health.iloc[0] == 1.0
    assert selected.attention.iloc[0] == "Inspect"
    assert frame.isolation_score.iloc[0] == 2.0
    with pytest.raises(ValueError, match="Missing selected-model"):
        select_model_frame(frame, "missing")


def test_model_integrity_rejects_bad_hash_before_deserialization(tmp_path):
    path = tmp_path / "models.joblib"
    path.write_bytes(b"not a pickle")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"models.joblib": "0" * 64}}))
    with pytest.raises(ValueError, match="integrity"):
        load_model(str(path), path.stat().st_mtime)


def test_real_v2_monitoring_and_model_selection_when_available():
    path = ROOT / "data/demo/v2_monitoring.parquet"
    if not path.exists():
        pytest.skip("V2 offline artifacts not generated yet")
    frame = load_monitoring(str(path), path.stat().st_mtime)
    metrics = json.loads((ROOT / "artifacts/v2/metrics.json").read_text())
    for model_id in metrics["models"]:
        selected = select_model_frame(frame, model_id)
        assert np.isfinite(selected.isolation_score).all()
        assert selected.health.between(0, 100).all()
    model_path = ROOT / "artifacts/v2/models.joblib"
    assert load_model(str(model_path), model_path.stat().st_mtime)["schema_version"] == 2


def test_v2_sidebar_switches_models_and_guided_cases_when_available():
    from streamlit.testing.v1 import AppTest

    path = ROOT / "artifacts/v2/metrics.json"
    if not path.exists():
        pytest.skip("V2 offline artifacts not generated yet")
    metrics = json.loads(path.read_text())
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45).run()
    assert not app.exception
    assert not app.error
    for model_id, info in metrics["models"].items():
        app.sidebar.selectbox[0].set_value(model_id).run()
        assert not app.exception
        assert not app.error
        assert any(info["display_name"] in item.value for item in app.caption)
    for model_id in ["covariance_pressure", "regime_pressure"]:
        if model_id in metrics["models"]:
            app.sidebar.selectbox[0].set_value(model_id).run()
            for page in ["Anomaly Detection", "Explainable AI"]:
                app.sidebar.radio[0].set_value(page).run()
                assert not app.exception
                assert not app.error
    app.sidebar.radio[0].set_value("Explainable AI").run()
    assert not app.exception
    assert not app.error
    case_selector = next(item for item in app.selectbox if item.label == "Guided historical case")
    if len(case_selector.options) > 1:
        case_selector.set_value(case_selector.options[1]).run()
        assert not app.exception
        assert len(app.dataframe) > 0


def test_v2_maintenance_handles_zero_alert_historical_day_when_available():
    from streamlit.testing.v1 import AppTest

    path = ROOT / "data/demo/v2_monitoring.parquet"
    if not path.exists():
        pytest.skip("V2 offline artifacts not generated yet")
    metrics = json.loads((ROOT / "artifacts/v2/metrics.json").read_text())
    frame = load_monitoring(str(path), path.stat().st_mtime)
    counts = frame[f"{metrics['selected_model']}_alert"].resample("D").agg(["sum", "count"])
    day = counts.loc[(counts["sum"] == 0) & (counts["count"] > 0)].index[0].date()
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45).run()
    app.sidebar.date_input[0].set_value((day, day))
    app.sidebar.radio[0].set_value("Maintenance Decision Support").run()
    assert not app.exception
    assert not app.error
    assert any("No persistent alert episodes" in item.value for item in app.info)


def test_v2_comparison_tables_show_actual_event_counts():
    from streamlit.testing.v1 import AppTest

    metrics = json.loads((ROOT / "artifacts/v2/metrics.json").read_text())
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Model Performance").run()
    tables = [
        item.value
        for item in app.dataframe
        if {"Model", "Detected events", "Unannotated episodes"}.issubset(item.value.columns)
    ]
    assert len(tables) == 2
    for partition, table in zip(["development", "retrospective"], tables):
        assert len(table) == len(metrics["models"])
        for info in metrics["models"].values():
            result = info[partition]
            row = table.loc[table.Model == info["display_name"]].iloc[0]
            assert (
                row["Detected events"] == f"{result['detected_events']} / {result['event_count']}"
            )
            assert row["Unannotated episodes"] == result["unannotated_episode_count"]
