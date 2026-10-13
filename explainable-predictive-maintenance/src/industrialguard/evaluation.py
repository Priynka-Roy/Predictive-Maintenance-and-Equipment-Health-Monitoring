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


def alert_policy(
    scores: pd.Series,
    high_threshold: float | pd.Series,
    persistence_bins: int = 3,
    low_threshold: float | pd.Series | None = None,
) -> pd.Series:
    """Causal consecutive-entry policy, optionally retaining alerts until the low threshold.

    Missing scores/thresholds and gaps longer than five minutes reset state. Without
    hysteresis an alert remains true only while scores exceed the high threshold.
    Equality is not an exceedance. Threshold series must use the score index.
    """
    if persistence_bins < 1 or int(persistence_bins) != persistence_bins:
        raise ValueError("persistence_bins must be a positive integer")
    _validate_timeline(scores)
    high = (
        high_threshold.reindex(scores.index)
        if isinstance(high_threshold, pd.Series)
        else pd.Series(high_threshold, index=scores.index)
    )
    low = (
        low_threshold.reindex(scores.index)
        if isinstance(low_threshold, pd.Series)
        else pd.Series(low_threshold, index=scores.index)
    )
    if low_threshold is not None and (low > high).any():
        raise ValueError("low_threshold must not exceed high_threshold")
    count, active, previous = 0, False, None
    result = []
    for timestamp, score, entry, exit_value in zip(scores.index, scores, high, low, strict=True):
        gap = previous is not None and timestamp - previous > pd.Timedelta("5min")
        previous = timestamp
        missing = (
            pd.isna(score) or pd.isna(entry) or (low_threshold is not None and pd.isna(exit_value))
        )
        if gap or missing:
            count, active = 0, False
        if missing:
            result.append(False)
            continue
        if active and low_threshold is not None:
            active = bool(score > exit_value)
            if not active:
                count = 0
        else:
            count = count + 1 if score > entry else 0
            active = count >= persistence_bins
        result.append(active)
    return pd.Series(result, index=scores.index, name="alert", dtype=bool)


def _validate_timeline(series: pd.Series) -> None:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("A DatetimeIndex is required")
    if not series.index.is_monotonic_increasing or not series.index.is_unique:
        raise ValueError("Timestamps must be sorted and unique")


def evaluate_v2(
    flags: pd.Series,
    failures: pd.DataFrame,
    lead_hours: float = 24,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> dict:
    """Evaluate onset-owned events/episodes in [start,end), preserving global context.

    Failure intervals are closed; observed exposure is nonmissing five-minute bins.
    Supply full-history flags to avoid inventing episode onsets at partition boundaries.
    """
    _validate_timeline(flags)
    if lead_hours < 0:
        raise ValueError("lead_hours must be nonnegative")
    if len(flags) == 0 and (start is None or end is None):
        raise ValueError("Empty telemetry requires explicit start and end")
    start = pd.Timestamp(start) if start is not None else flags.index.min()
    end = pd.Timestamp(end) if end is not None else flags.index.max() + pd.Timedelta("5min")
    if end <= start:
        raise ValueError("end must follow start")
    events = failures.copy()
    for col in ("start", "end"):
        events[col] = pd.to_datetime(events[col], format="mixed")
    if events[["start", "end"]].isna().any().any() or (events.end < events.start).any():
        raise ValueError("Failure intervals must have valid ordered endpoints")
    observed = flags.notna()
    positive = flags.fillna(False).astype(bool)
    episodes = alert_episodes(positive)
    annotated = pd.Series(False, index=flags.index)
    for event in events.itertuples():
        annotated |= (flags.index >= event.start) & (flags.index <= event.end)
    owned_events = events[(events.start >= start) & (events.start < end)]
    owned_episodes = episodes[(episodes.start >= start) & (episodes.start < end)]
    rows = []
    for event in owned_events.itertuples():
        period = (flags.index >= event.start) & (flags.index <= event.end)
        observed_bins = int((period & observed).sum())
        expected_bins = len(
            pd.date_range(event.start.ceil("5min"), event.end.floor("5min"), freq="5min")
        )
        coverage = (
            "unobservable"
            if observed_bins == 0
            else "partial"
            if observed_bins < expected_bins
            else "complete"
        )
        # Reject whole episodes associated with any other documented failure.
        other_events = events[events.event_id != event.event_id]
        eligible_episodes = episodes.copy()
        other_overlap = pd.Series(False, index=episodes.index)
        for other in other_events.itertuples():
            other_overlap |= (episodes.start <= other.end) & (episodes.end >= other.start)
        eligible_episodes = eligible_episodes[~other_overlap]
        eligible_flags = positive.copy()
        for excluded in episodes[other_overlap].itertuples():
            eligible_flags.loc[excluded.start : excluded.end] = False
        warning_mask = (
            (flags.index >= event.start - pd.Timedelta(hours=lead_hours))
            & (flags.index < event.start)
            & eligible_flags
            & observed
            & ~annotated
        )
        warning = flags.index[warning_mask]
        onset_candidates = eligible_episodes[
            (eligible_episodes.start >= event.start - pd.Timedelta(hours=lead_hours))
            & (eligible_episodes.start < event.start)
        ]
        eligible_onsets = []
        for onset in onset_candidates.start:
            if not bool(((events.start <= onset) & (events.end >= onset)).any()):
                eligible_onsets.append(onset)

        def lead(timestamp):
            return (
                float((event.start - timestamp).total_seconds() / 3600)
                if timestamp is not None
                else None
            )

        earliest = warning[0] if len(warning) else None
        nearest = warning[-1] if len(warning) else None
        onset = min(eligible_onsets) if eligible_onsets else None
        rows.append(
            {
                "event_id": int(event.event_id),
                "start": str(event.start),
                "end": str(event.end),
                "observed_bins": observed_bins,
                "expected_bins": expected_bins,
                "coverage": coverage,
                "coverage_fraction": min(observed_bins / expected_bins, 1.0)
                if expected_bins
                else None,
                "detected_during_event": bool(positive[period].any()),
                "earliest_warning_time": str(earliest) if earliest is not None else None,
                "nearest_warning_time": str(nearest) if nearest is not None else None,
                "warning_episode_onset": str(onset) if onset is not None else None,
                "earliest_warning_lead_hours": lead(earliest),
                "nearest_warning_lead_hours": lead(nearest),
                "episode_onset_warning_lead_hours": lead(onset),
                "lead_time_hours": lead(earliest),
            }
        )
    overlap = pd.Series(False, index=owned_episodes.index)
    for event in events.itertuples():
        overlap |= (owned_episodes.start <= event.end) & (owned_episodes.end >= event.start)
    unannotated = int((~overlap).sum())
    partition = (flags.index >= start) & (flags.index < end)
    observed_days = float((partition & observed).sum() * 5 / 1440)
    unannotated_days = float((partition & observed & ~annotated).sum() * 5 / 1440)
    calendar_days = float((end - start).total_seconds() / 86400)
    observable_count = sum(row["observed_bins"] > 0 for row in rows)
    detected = sum(row["detected_during_event"] for row in rows)
    return {
        "start": str(start),
        "end_exclusive": str(end),
        "assigned_event_count": len(rows),
        "event_count": observable_count,
        "unobservable_events": len(rows) - observable_count,
        "partial_events": sum(row["coverage"] == "partial" for row in rows),
        "complete_events": sum(row["coverage"] == "complete" for row in rows),
        "detected_events": detected,
        "event_detection_rate": detected / observable_count if observable_count else None,
        "alert_episodes": len(owned_episodes),
        "crossing_onset_episodes": int(((episodes.start < start) & (episodes.end >= start)).sum()),
        "unannotated_episode_count": unannotated,
        "unannotated_alert_episodes": unannotated,
        "observed_days": observed_days,
        "calendar_days": calendar_days,
        "unannotated_observed_days": unannotated_days,
        "unannotated_alerts_per_observed_day": unannotated / unannotated_days
        if unannotated_days
        else None,
        "unannotated_alerts_per_calendar_day": unannotated / calendar_days,
        "alert_bin_fraction": float(positive[partition & observed].mean())
        if observed_days
        else None,
        "events": rows,
    }
