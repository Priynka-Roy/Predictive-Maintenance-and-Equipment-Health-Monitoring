"""The committed demonstration reproduces scores from the shipped trained model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from industrialguard.models import score_frame

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_model_reproduces_precomputed_demo_scores():
    bundle = joblib.load(ROOT / "artifacts/model.joblib")
    demo = pd.read_parquet(ROOT / "data/demo/monitoring.parquet")
    if "timestamp" in demo.columns:
        demo = demo.set_index("timestamp")
    # Cover the beginning, middle and end rather than a single favorable case.
    selected = demo.iloc[np.unique(np.linspace(0, len(demo) - 1, 30).astype(int))]
    scored = score_frame(selected, bundle)
    for column in ["robust_score", "isolation_score", "health"]:
        np.testing.assert_allclose(scored[column], selected[column], atol=1e-10)
    np.testing.assert_array_equal(scored.attention, selected.attention)
    metrics = json.loads((ROOT / "artifacts/metrics.json").read_text())
    assert metrics["thresholds"] == bundle["thresholds"]
    assert metrics["feature_count"] == len(bundle["feature_names"])
    assert demo.index.is_monotonic_increasing and demo.index.is_unique
