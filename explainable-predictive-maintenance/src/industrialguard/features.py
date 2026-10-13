"""Causal trailing features with resets at missing bins and telemetry outages."""

import pandas as pd

from industrialguard.data import ANALOG, SENSORS


def make_features(frame: pd.DataFrame, windows: tuple[int, ...] = (3, 12)) -> pd.DataFrame:
    valid = frame[SENSORS].notna().all(axis=1)
    gap = frame.index.to_series().diff() > pd.Timedelta("5min")
    segment = ((~valid) | (~valid).shift(fill_value=False) | gap).cumsum()
    result = frame.copy()
    result["pressure_difference"] = frame.TP2 - frame.TP3
    for sensor in ANALOG:
        groups = frame[sensor].groupby(segment)
        result[f"{sensor}_change"] = groups.diff()
        for window in windows:
            result[f"{sensor}_mean_{window}"] = groups.transform(
                lambda s: s.rolling(window, min_periods=window).mean()
            )
            result[f"{sensor}_std_{window}"] = groups.transform(
                lambda s: s.rolling(window, min_periods=window).std(ddof=0)
            )
    result["load_transition"] = frame.DV_eletric.groupby(segment).diff().abs()
    result["operating_state"] = pd.cut(
        frame.Motor_current,
        [-float("inf"), 1, 5.5, float("inf")],
        labels=["off", "offloaded", "loaded"],
    ).astype("string")
    result.loc[~valid, "operating_state"] = pd.NA
    return result


def feature_names(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame if c not in {"sample_count", "operating_state"}]
