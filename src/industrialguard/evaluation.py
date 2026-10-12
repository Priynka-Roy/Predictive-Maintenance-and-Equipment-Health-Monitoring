"""Event overlaps and proxy alert burden on observed, unannotated telemetry."""

import pandas as pd


def persistent_alerts(flags: pd.Series, bins: int = 3) -> pd.Series:
    """A flag fires only after consecutive positive bins; resets across telemetry gaps."""
    segments = (flags.index.to_series().diff() > pd.Timedelta("5min")).cumsum()
    return (
        flags.astype(int)
        .groupby(segments)
        .transform(lambda x: x.rolling(bins, min_periods=bins).sum().eq(bins))
        .astype(bool)
    )


def alert_episodes(flags: pd.Series) -> pd.DataFrame:
    gap = flags.index.to_series().diff() > pd.Timedelta("5min")
    start = flags & (~flags.shift(fill_value=False) | gap)
    group = start.cumsum()
    rows = []
    for _, points in flags[flags].groupby(group[flags]):
        rows.append({"start": points.index.min(), "end": points.index.max(), "bins": len(points)})
    return pd.DataFrame(rows, columns=["start", "end", "bins"])


def evaluate(flags: pd.Series, failures: pd.DataFrame, lead_hours: int = 24) -> dict:
    events = failures.copy()
    for col in ["start", "end"]:
        events[col] = pd.to_datetime(events[col])
    all_events = events.copy()
    # Only fully contained documented events belong to this chronological partition.
    events = events[(events.start >= flags.index.min()) & (events.end <= flags.index.max())]
    episodes = alert_episodes(flags)
    annotated = pd.Series(False, index=flags.index)
    rows = []
    for event in all_events.itertuples():
        annotated |= (flags.index >= event.start) & (flags.index <= event.end)
    for event in events.itertuples():
        period = (flags.index >= event.start) & (flags.index <= event.end)
        annotated |= period
        detected = bool(flags[period].any())
        preceding = flags[
            (flags.index >= event.start - pd.Timedelta(hours=lead_hours))
            & (flags.index < event.start)
        ]
        warning = preceding[preceding]
        rows.append(
            {
                "event_id": int(event.event_id),
                "start": str(event.start),
                "end": str(event.end),
                "observed_bins": int(period.sum()),
                "detected_during_event": detected,
                "lead_time_hours": float((event.start - warning.index[0]).total_seconds() / 3600)
                if len(warning)
                else None,
            }
        )
    # Episode overlapping any documented interval is excluded from unannotated burden.
    overlap = pd.Series(False, index=episodes.index)
    for event in all_events.itertuples():
        overlap |= (episodes.start <= event.end) & (episodes.end >= event.start)
    unannotated = int((~overlap).sum())
    observed_days = float((~annotated).sum() * 5 / (60 * 24))
    return {
        "event_count": sum(r["observed_bins"] > 0 for r in rows),
        "unobservable_events": sum(r["observed_bins"] == 0 for r in rows),
        "detected_events": sum(r["detected_during_event"] for r in rows),
        "event_detection_rate": sum(r["detected_during_event"] for r in rows)
        / sum(r["observed_bins"] > 0 for r in rows)
        if any(r["observed_bins"] > 0 for r in rows)
        else None,
        "alert_episodes": len(episodes),
        "unannotated_alert_episodes": unannotated,
        "unannotated_observed_days": observed_days,
        "unannotated_alerts_per_observed_day": unannotated / observed_days
        if observed_days
        else None,
        "alert_bin_fraction": float(flags.mean()),
        "events": rows,
    }
