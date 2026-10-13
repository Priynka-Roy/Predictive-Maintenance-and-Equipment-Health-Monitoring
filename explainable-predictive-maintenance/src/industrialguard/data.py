"""Schema and chronology checks for the original UCI sensor release."""

from pathlib import Path

import numpy as np
import pandas as pd

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]
SENSORS = ANALOG + DIGITAL
UNITS = {
    **dict.fromkeys(ANALOG[:5], "bar"),
    "Oil_temperature": "°C",
    "Motor_current": "A",
    **dict.fromkeys(DIGITAL, "binary"),
}


def validate_frame(frame: pd.DataFrame) -> dict:
    """Reject unusable schema/chronology; report range issues without clipping sensors."""
    missing = set(["timestamp", *SENSORS]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    t = pd.to_datetime(frame.timestamp, errors="raise")
    if t.isna().any() or not t.is_monotonic_increasing or t.duplicated().any():
        raise ValueError("Timestamps must be nonmissing, unique and chronological")
    values = frame[SENSORS].apply(pd.to_numeric, errors="raise")
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Infinite sensor values are invalid")
    if values.empty:
        raise ValueError("No sensor observations")
    delta = t.diff().dt.total_seconds().dropna()
    return {
        "rows": len(frame),
        "start": str(t.min()),
        "end": str(t.max()),
        "missing": values.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "median_interval_seconds": float(delta.median()) if len(delta) else None,
        "gaps_over_60_seconds": int((delta > 60).sum()),
        "largest_gap_seconds": float(delta.max()) if len(delta) else None,
        "nonbinary_counts": {
            c: int((values[c].notna() & ~values[c].isin([0, 1])).sum()) for c in DIGITAL
        },
        "negative_pressure_counts": {c: int((values[c] < 0).sum()) for c in ANALOG[:5]},
        "sensor_summary": values.describe(percentiles=[0.01, 0.5, 0.99]).to_dict(),
    }


def load_raw(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="raise")
    quality = validate_frame(frame)
    return frame[["timestamp", *SENSORS]], quality


def resample_sensors(frame: pd.DataFrame, frequency: str = "5min") -> pd.DataFrame:
    """Right-closed trailing bins: a timestamp denotes when a bin becomes available.

    Empty/low coverage bins remain missing. No interpolation across outages.
    Digital means are duty fractions, not invented binary observations.
    """
    source = frame.set_index("timestamp")[SENSORS]
    grouped = source.resample(frequency, label="right", closed="right")
    result = grouped.mean()
    result["sample_count"] = grouped.size()
    result.loc[result.sample_count < 15, SENSORS] = np.nan
    return result.rename_axis("timestamp")
