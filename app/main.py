"""Presentation layer for the precomputed MetroPT monitoring demonstration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SENSORS = {
    "TP2": "bar",
    "TP3": "bar",
    "H1": "bar",
    "DV_pressure": "bar",
    "Reservoirs": "bar",
    "Motor_current": "A",
    "Oil_temperature": "°C",
    "COMP": "state",
    "DV_eletric": "state",
    "Towers": "state",
    "MPG": "state",
    "LPS": "state",
    "Pressure_switch": "state",
    "Oil_level": "state",
    "Caudal_impulses": "state",
}
PAGES = [
    "Overview",
    "Sensor Explorer",
    "Anomaly Detection",
    "Equipment Health",
    "Explainable AI",
    "Maintenance Decision Support",
    "Model Performance",
    "About the Project",
]
COLORS = {
    "Normal": "#40c4aa",
    "Monitor": "#e8c76b",
    "Inspect": "#ee9967",
    "High attention": "#e86f78",
}


@st.cache_data(show_spinner=False)
def load_monitoring(path: str, modified: float) -> pd.DataFrame:
    """Load a validated timestamp-indexed monitoring table; never train models."""
    del modified
    frame = pd.read_parquet(path)
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        frame = frame.set_index("timestamp")
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Monitoring data must contain a timestamp column or datetime index.")
    if frame.empty or frame.index.hasnans:
        raise ValueError("Monitoring data is empty or contains invalid timestamps.")
    required = set(SENSORS) | {
        "robust_score",
        "isolation_score",
        "robust_alert",
        "isolation_alert",
        "health",
        "attention",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing monitoring columns: {', '.join(sorted(missing))}")
    if frame.index.has_duplicates:
        raise ValueError("Monitoring timestamps must be unique.")
    scores = frame[["robust_score", "isolation_score", "health"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not frame.health.between(0, 100).all():
        raise ValueError(
            "Monitoring scores must be finite and relative health must be between 0 and 100."
        )
    return frame.sort_index()


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified: float) -> dict[str, Any]:
    del modified
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_metrics(path: str, modified: float) -> dict[str, Any]:
    del modified
    return json.loads(Path(path).read_text())


def chart(figure: go.Figure) -> None:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=45, b=20),
        font=dict(family="Arial", color="#cbd5e1"),
        height=390,
        legend=dict(orientation="h", y=1.15),
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#253346")
    st.plotly_chart(figure, use_container_width=True)


def timeseries(frame: pd.DataFrame, columns: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    palette = ["#40c4aa", "#6da8ed", "#e8c76b", "#b199eb"]
    plotted = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="5min"))
    for i, name in enumerate(columns):
        label = f"{name} ({SENSORS[name]})" if name in SENSORS else name.replace("_", " ")
        fig.add_trace(
            go.Scatter(
                x=plotted.index,
                y=plotted[name],
                name=label,
                line=dict(color=palette[i % len(palette)], width=1.6),
            )
        )
    fig.update_layout(
        title=title, xaxis_title="Timestamp (dataset local time; timezone unspecified)"
    )
    return fig


def alert_episodes(frame: pd.DataFrame, column: str = "isolation_alert") -> pd.DataFrame:
    """Summarize contiguous alert observations within the selected sample."""
    active = frame[column].fillna(False).astype(bool)
    gap = frame.index.to_series().diff() > pd.Timedelta("5min")
    groups = (active.ne(active.shift(fill_value=False)) | gap).cumsum()
    rows = []
    for _, block in frame.loc[active].groupby(groups[active]):
        rows.append(
            {
                "Start": block.index.min(),
                "End": block.index.max(),
                "Observations": len(block),
                "Lowest relative health": float(block.health.min()),
            }
        )
    return pd.DataFrame(rows)


def overview(frame: pd.DataFrame, full: pd.DataFrame, metrics: dict[str, Any]) -> None:
    st.markdown("### One compressor. A clearer view of its condition.")
    st.write(
        "Explore real MetroPT-3 Air Production Unit measurements through statistical monitoring, "
        "machine learning, and transparent maintenance attention indicators."
    )
    latest = frame.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("Latest relative health", f"{latest.health:.0f} / 100")
    cols[1].metric("Attention", str(latest.attention))
    cols[2].metric("Selected alert observations", f"{int(frame.isolation_alert.sum()):,}")
    cols[3].metric("Alert share", f"{frame.isolation_alert.mean():.1%}")
    chart(timeseries(frame, ["health"], "Relative health across the selected period"))
    left, right = st.columns([3, 2])
    with left:
        daily = frame.isolation_alert.resample("D").agg(["sum", "count"])
        chart(
            px.bar(
                daily,
                x=daily.index,
                y="sum",
                title="Alert observations by day",
                labels={"sum": "Alert observations", "index": "Date"},
                color_discrete_sequence=["#6da8ed"],
            )
        )
    with right:
        st.markdown("#### Monitoring context")
        st.write(
            f"**Equipment:** one metro-train air compressor\n\n"
            f"**Demo observations:** {len(full):,}\n\n"
            f"**Selected period:** {frame.index.min():%d %b %Y} → {frame.index.max():%d %b %Y}"
        )
        st.info(
            "Health is a relative anomaly indicator. An alert identifies unusual behavior; "
            "it does not establish a failure or its cause."
        )
    st.caption(
        "Counts and charts use the precomputed five-minute demonstration. See Model Performance for held-out evaluation artifacts."
    )


def explorer(frame: pd.DataFrame) -> None:
    st.caption(
        "Signals are five-minute means. Digital state means represent the fraction of samples active in each bin."
    )
    available = [name for name in SENSORS if name in frame]
    selected = st.multiselect(
        "Sensor signals",
        available,
        default=[name for name in ["TP2", "TP3", "Motor_current"] if name in available],
    )
    if not selected:
        st.info("Select at least one sensor to inspect its measurements.")
        return
    for sensor in selected:
        fig = timeseries(frame, [sensor], f"{sensor} · {SENSORS[sensor]}")
        if st.session_state.get("show_alerts", True):
            alerts = frame[frame.isolation_alert.astype(bool)]
            fig.add_trace(
                go.Scatter(
                    x=alerts.index,
                    y=alerts[sensor],
                    mode="markers",
                    name="Isolation Forest alert",
                    marker=dict(color="#e86f78", size=5),
                )
            )
        chart(fig)
    st.markdown("#### Selected-period statistics")
    st.dataframe(frame[selected].describe().T, width="stretch")
    with st.expander("Sensor distributions"):
        for sensor in selected:
            chart(
                px.histogram(
                    frame,
                    x=sensor,
                    nbins=50,
                    title=f"{sensor} distribution · {SENSORS[sensor]}",
                    color_discrete_sequence=["#6da8ed"],
                )
            )
    if len(selected) > 1:
        chart(
            px.imshow(
                frame[selected].corr(),
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                title="Sensor correlation · association, not causation",
            )
        )


def detection(frame: pd.DataFrame, model: dict[str, Any]) -> None:
    st.write(
        "Compare a transparent robust-deviation baseline with Isolation Forest. "
        "Both use historical features and thresholds fixed before final test evaluation."
    )
    for name, score in [
        ("Robust statistical baseline", "robust_score"),
        ("Isolation Forest", "isolation_score"),
    ]:
        fig = timeseries(frame, [score], name)
        threshold = model.get("thresholds", {}).get(
            "robust" if score == "robust_score" else "isolation"
        )
        if threshold is not None and np.isscalar(threshold):
            fig.add_hline(
                y=float(threshold),
                line_dash="dash",
                line_color="#e86f78",
                annotation_text="Fixed threshold",
            )
        chart(fig)
    st.dataframe(
        pd.DataFrame(
            {
                "Model": ["Robust baseline", "Isolation Forest"],
                "Alert observations": [
                    int(frame.robust_alert.sum()),
                    int(frame.isolation_alert.sum()),
                ],
                "Alert share": [frame.robust_alert.mean(), frame.isolation_alert.mean()],
            }
        ),
        hide_index=True,
    )
    st.caption(
        "Model score scales differ. Their numerical magnitudes cannot be compared directly. "
        "Alerts require three consecutive flagged five-minute bins; incomplete feature windows are excluded."
    )


def health(frame: pd.DataFrame) -> None:
    latest = frame.iloc[-1]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(latest.health),
            title={"text": "Latest relative health"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#40c4aa"},
                "steps": [
                    {"range": [0, 0.5], "color": "#61343e"},
                    {"range": [0.5, 1], "color": "#594639"},
                    {"range": [1, 5], "color": "#56513c"},
                    {"range": [5, 100], "color": "#254a48"},
                ],
            },
        )
    )
    chart(fig)
    st.write(
        f"Latest category: **{latest.attention}**. The score ranks unusual behavior relative to the "
        "validation score distribution. It is not a calibrated failure probability or remaining useful life estimate."
    )
    chart(timeseries(frame, ["health"], "Health indicator history"))
    counts = (
        frame.attention.value_counts().rename_axis("Attention").reset_index(name="Observations")
    )
    chart(
        px.bar(
            counts,
            x="Attention",
            y="Observations",
            color="Attention",
            color_discrete_map=COLORS,
            title="Maintenance attention distribution",
        )
    )
    st.caption(
        "Attention is assigned at validation score percentiles 95, 99, and 99.5. "
        "Approximate corresponding health boundaries are 5, 1, and 0.5. "
        "Higher health means more typical behavior; a low percentile health alone is not evidence of damage."
    )


def explanations(frame: pd.DataFrame, model: dict[str, Any]) -> None:
    candidates = frame.nlargest(min(100, len(frame)), "isolation_score").sort_index()
    stamp = st.selectbox(
        "Inspect an observation (100 most unusual in selected period)",
        candidates.index,
        format_func=lambda value: value.strftime("%Y-%m-%d %H:%M"),
    )
    row = frame.loc[stamp]
    features = model.get("feature_names", [])
    center = model.get("robust_center", model.get("center"))
    scale = model.get("robust_scale", model.get("scale"))
    if (
        center is None
        or scale is None
        or not features
        or any(name not in frame for name in features)
    ):
        st.warning("Feature reference statistics are unavailable in this model artifact.")
        return
    deviations = np.abs(
        (row[features].to_numpy(dtype=float) - np.asarray(center)) / np.asarray(scale)
    )
    table = pd.DataFrame(
        {
            "Feature": features,
            "Absolute robust deviation": deviations,
            "Observed": row[features].to_numpy(dtype=float),
            "Training center": np.asarray(center),
        }
    )
    table = table.sort_values("Absolute robust deviation", ascending=False)
    chart(
        px.bar(
            table.head(12).iloc[::-1],
            x="Absolute robust deviation",
            y="Feature",
            orientation="h",
            color_discrete_sequence=["#40c4aa"],
            title="Largest deviations from training reference",
        )
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.info(
        "These are robust feature deviations, not additive Isolation Forest contributions. "
        "They explain unusual measured patterns relative to the training reference, not proven physical causes. "
        "Correlated and derived features can describe the same underlying signal."
    )


def maintenance(frame: pd.DataFrame, metrics: dict[str, Any]) -> None:
    st.write(
        "Use alert episodes to prioritize review of the monitored compressor. "
        "Inspection decisions require equipment history, operating context, and qualified engineering judgment."
    )
    episodes = alert_episodes(frame)
    st.dataframe(episodes.tail(50).iloc[::-1], hide_index=True, width="stretch")
    st.caption(
        "Episodes group adjacent five-minute alert rows and reset across gaps. These are model alerts, not confirmed incidents."
    )
    st.markdown("#### Illustrative review-cost scenario")
    cols = st.columns(3)
    review_cost = cols[0].number_input(
        "Cost per alert review (€)", min_value=0.0, value=150.0, step=25.0
    )
    interruption_hours = cols[1].number_input(
        "Hours per review", min_value=0.0, value=0.5, step=0.25
    )
    hourly_cost = cols[2].number_input(
        "Assumed interruption cost (€/h)", min_value=0.0, value=300.0, step=50.0
    )
    st.metric(
        "Illustrative selected-period review burden",
        f"€{len(episodes) * (review_cost + interruption_hours * hourly_cost):,.0f}",
    )
    st.caption(
        "User-editable assumptions, not observed costs. No failure avoidance or financial return is estimated."
    )
    st.markdown("#### Score threshold sensitivity · selected sample")
    quantiles = [0.90, 0.95, 0.975, 0.99, 0.995]
    sensitivity = pd.DataFrame(
        {
            "Sample score quantile": quantiles,
            "Score threshold": [frame.isolation_score.quantile(q) for q in quantiles],
        }
    )
    sensitivity["Alert observations"] = [
        int((frame.isolation_score > value).sum()) for value in sensitivity["Score threshold"]
    ]
    st.dataframe(sensitivity, hide_index=True)
    st.caption(
        "Exploratory sample quantiles illustrate workload only. They do not modify the deployed threshold or evaluation results."
    )
    if metrics.get("sensitivity"):
        st.markdown("#### Offline validation trade-offs")
        tested = pd.DataFrame(metrics["sensitivity"]).drop(columns=["events"], errors="ignore")
        tested["Persistence (bins)"] = tested.persistence_bins.astype(str)
        chart(
            px.line(
                tested,
                x="quantile",
                y="unannotated_alerts_per_observed_day",
                color="Persistence (bins)",
                markers=True,
                title="Validation threshold and persistence sensitivity",
                labels={
                    "quantile": "Validation score quantile",
                    "unannotated_alerts_per_observed_day": "Unannotated episodes / observed day",
                },
            )
        )
        st.caption(
            "Computed on validation data, without final test threshold optimization. Lower alert burden can also miss reported failures."
        )
    st.info(
        "Monitor persistent changes, check operating-state transitions, and verify sensor quality before "
        "escalating an anomaly for inspection. These are decision-support prompts, not certified maintenance instructions."
    )


def performance(metrics: dict[str, Any]) -> None:
    st.write(
        "Evaluation artifacts were computed offline using chronological train, validation, and held-out test periods."
    )
    partitions = metrics.get("partitions", {})
    test = partitions.get("test", {})
    results = {
        name: test[name] for name in ["robust", "isolation"] if isinstance(test.get(name), dict)
    }
    if not results:
        st.error(
            "The metrics artifact does not contain chronological held-out model results. Regenerate artifacts with the offline pipeline."
        )
        return
    st.markdown("#### Final held-out results")
    cols = st.columns(4)
    for column, name, label in [
        (cols[0], "robust", "Robust detected events"),
        (cols[1], "isolation", "Isolation Forest detected events"),
    ]:
        value = results.get(name, {})
        column.metric(label, f"{value.get('detected_events', 0)} / {value.get('event_count', 0)}")
    cols[2].metric("Held-out observations", f"{test.get('rows', 0):,}")
    latency = metrics.get("inference_1000_rows_seconds")
    cols[3].metric(
        "Inference / 1,000 rows",
        f"{latency * 1000:.1f} ms" if latency is not None else "Unavailable",
    )
    robust = results.get("robust", {})
    isolation = results.get("isolation", {})
    st.warning(
        f"Isolation Forest detected {isolation.get('detected_events', 0)} of {isolation.get('event_count', 0)} "
        f"held-out documented events; the robust baseline detected {robust.get('detected_events', 0)} of "
        f"{robust.get('event_count', 0)}. The selected model's low alert burden is not evidence of useful failure detection. "
        "Neither model demonstrates reliable early warning on this test period."
    )
    st.caption(
        f"Test period: {test.get('start', 'unknown')} → {test.get('end', 'unknown')}. "
        "Evaluation is fixed across sidebar date selections."
    )
    if isinstance(results, dict) and results:
        flat = pd.DataFrame(
            [
                dict(
                    model=name,
                    **{
                        key: item
                        for key, item in value.items()
                        if not isinstance(item, (dict, list))
                    },
                )
                for name, value in results.items()
                if isinstance(value, dict)
            ]
        )
        st.dataframe(flat, hide_index=True, width="stretch")
        if {"model", "unannotated_alerts_per_observed_day"}.issubset(flat.columns):
            chart(
                px.bar(
                    flat,
                    x="model",
                    y="unannotated_alerts_per_observed_day",
                    color_discrete_sequence=["#6da8ed"],
                    title="Held-out unannotated alert burden",
                    labels={"unannotated_alerts_per_observed_day": "Episodes per observed day"},
                )
            )
        event_rows = [
            dict(model=name, **event)
            for name, value in results.items()
            if isinstance(value, dict)
            for event in value.get("events", [])
        ]
        if event_rows:
            st.markdown("#### Held-out event detection and retrospective lead time")
            st.dataframe(pd.DataFrame(event_rows), hide_index=True, width="stretch")
    st.markdown("#### Chronological validation methodology")
    split_rows = [
        {
            "Partition": name,
            "Observations": value.get("rows"),
            "Start": value.get("start"),
            "End": value.get("end"),
        }
        for name, value in partitions.items()
    ]
    st.dataframe(pd.DataFrame(split_rows), hide_index=True, width="stretch")
    st.write(metrics.get("selection_basis", ""))
    st.caption(metrics.get("calibration_reference", ""))
    if metrics.get("candidates"):
        st.markdown("#### Candidate comparison · validation only")
        st.dataframe(pd.DataFrame(metrics["candidates"]), hide_index=True, width="stretch")
    monthly_rows = [
        dict(
            Month=month,
            Model=name,
            **{key: value for key, value in result.items() if not isinstance(value, (dict, list))},
        )
        for month, models in metrics.get("test_months", {}).items()
        for name, result in models.items()
    ]
    if monthly_rows:
        chart(
            px.line(
                pd.DataFrame(monthly_rows),
                x="Month",
                y="unannotated_alerts_per_observed_day",
                color="Model",
                markers=True,
                title="Unannotated alert burden across held-out months",
                labels={"unannotated_alerts_per_observed_day": "Episodes / observed day"},
            )
        )
    with st.expander("Complete computed evaluation artifact"):
        st.json(metrics, expanded=False)
    st.markdown("#### Historical failure context")
    event_path = ROOT / "configs" / "failures.csv"
    if event_path.exists():
        events = pd.read_csv(event_path)
        st.dataframe(events, hide_index=True, width="stretch")
        events["start"] = pd.to_datetime(events["start"])
        events["end"] = pd.to_datetime(events["end"])
        events["Event"] = "Report " + events.event_id.astype(str)
        chart(
            px.timeline(
                events,
                x_start="start",
                x_end="end",
                y="Event",
                color_discrete_sequence=["#e86f78"],
                title="Documented historical failure periods",
            )
        )
    st.warning(
        "Few documented failures and incomplete maintenance labels limit generalization. "
        "Alerts outside documented windows are proxy false alerts, not verified false positives. "
        "Correlated observations do not support independent-observation confidence intervals. "
        "Any observed lead times are retrospective event summaries, not guaranteed early warning."
    )


def about() -> None:
    st.markdown("### Explainable Predictive Maintenance & Equipment Health Monitoring")
    st.write(
        "IndustrialGuard AI is an applied statistics and machine learning portfolio project using one real "
        "metro train Air Production Unit. It demonstrates transferable condition-monitoring methods for "
        "industrial equipment without claiming deployment or validation at any industrial company."
    )
    st.markdown("#### Data and attribution")
    st.markdown(
        "[MetroPT-3 · UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) "
        "· [Dataset DOI](https://doi.org/10.24432/C5VW3R) · "
        "[Research publication](https://www.nature.com/articles/s41597-022-01877-3). "
        "Data is licensed under CC BY 4.0. The compact demonstration derives from original measurements."
    )
    st.markdown("#### Methodology")
    st.write(
        "Timestamp validation and resampling → causal rolling features → chronological separation → "
        "robust statistical baseline and Isolation Forest → validation threshold selection → "
        "held-out evaluation → relative health and feature-deviation explanations."
    )
    st.markdown("#### Technology")
    st.write(
        "Python · pandas · NumPy · scikit-learn · Streamlit · Plotly · Parquet · pytest · Ruff · GitHub Actions"
    )
    st.markdown("#### Limitations")
    st.write(
        "Single equipment item; limited failure reports; regime changes; retrospective evaluation; "
        "sampled demonstration; no calibrated failure probabilities, remaining useful life predictions, "
        "or proven causal diagnosis. A field deployment would require domain validation and ongoing monitoring."
    )
    repo_path = ROOT / "artifacts" / "repository.json"
    if repo_path.exists():
        url = json.loads(repo_path.read_text()).get("url")
        if url:
            st.link_button("View GitHub repository", url)


def main() -> None:
    st.set_page_config(
        page_title="IndustrialGuard AI",
        page_icon="⚙",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """<style>
        .stApp { background:#101925; color:#e6edf5; }
        [data-testid='stSidebar'] { background:#152131; }
        [data-testid='stMetric'] { background:#192638; padding:20px; border-radius:10px; border:1px solid #2c3b50; }
        [data-testid='stMetricLabel'] { color:#9dafc3; }
        h1 { letter-spacing:-1px; } .block-container { padding-top:2.2rem; max-width:1500px; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("## ⚙ IndustrialGuard AI")
    st.sidebar.caption("INDUSTRIAL CONDITION INTELLIGENCE")
    page = st.sidebar.radio("Workspace", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("MetroPT-3 · single compressor\n\nRetrospective demonstration")
    st.title(page)
    st.caption("INDUSTRIALGUARD AI / AIR PRODUCTION UNIT / METROPT-3")
    if page == "About the Project":
        about()
        return
    data_path = ROOT / "data" / "demo" / "monitoring.parquet"
    model_path = ROOT / "artifacts" / "model.joblib"
    metrics_path = ROOT / "artifacts" / "metrics.json"
    try:
        full = load_monitoring(str(data_path), data_path.stat().st_mtime)
        model = load_model(str(model_path), model_path.stat().st_mtime)
        metrics = load_metrics(str(metrics_path), metrics_path.stat().st_mtime)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Monitoring artifacts could not be loaded: {error}")
        st.info(
            "Run the reproducible offline pipeline described in README.md to prepare the demo and model artifacts. "
            "The application does not train models during startup."
        )
        return
    limits = (full.index.min().date(), full.index.max().date())
    dates = st.sidebar.date_input(
        "Historical period", value=limits, min_value=limits[0], max_value=limits[1]
    )
    st.sidebar.checkbox("Highlight alerts", value=True, key="show_alerts")
    if not isinstance(dates, (list, tuple)) or len(dates) != 2:
        st.info("Select both a start and end date.")
        return
    mask = (full.index.date >= dates[0]) & (full.index.date <= dates[1])
    frame = full.loc[mask]
    if frame.empty:
        st.info("No observations in this range. Choose another historical period.")
        return
    if page == "Overview":
        overview(frame, full, metrics)
    elif page == "Sensor Explorer":
        explorer(frame)
    elif page == "Anomaly Detection":
        detection(frame, model)
    elif page == "Equipment Health":
        health(frame)
    elif page == "Explainable AI":
        explanations(frame, model)
    elif page == "Maintenance Decision Support":
        maintenance(frame, metrics)
    elif page == "Model Performance":
        performance(metrics)
    st.divider()
    st.caption(
        "Research and portfolio demonstration · Anomalies are not automatically failures · No live equipment connection"
    )


if __name__ == "__main__":
    main()
