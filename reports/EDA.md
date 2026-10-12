# Reproducible statistical EDA report

Generated from the original UCI CSV by `python scripts/run_pipeline.py`.

## Data audit

- Original records: 1,516,948; period: 2020-02-01 00:00:00 through 2020-09-01 03:59:50.
- Median observed interval: 10.0 seconds; gaps >60 seconds: 331.
- Missing values by sensor: {'TP2': 0, 'TP3': 0, 'H1': 0, 'DV_pressure': 0, 'Reservoirs': 0, 'Oil_temperature': 0, 'Motor_current': 0, 'COMP': 0, 'DV_eletric': 0, 'Towers': 0, 'MPG': 0, 'LPS': 0, 'Pressure_switch': 0, 'Oil_level': 0, 'Caudal_impulses': 0}.
- Digital values outside 0/1: {'COMP': 0, 'DV_eletric': 0, 'Towers': 0, 'MPG': 0, 'LPS': 0, 'Pressure_switch': 0, 'Oil_level': 0, 'Caudal_impulses': 0}.
- Negative pressure observations: {'TP2': 1275474, 'TP3': 0, 'H1': 237996, 'DV_pressure': 1444001, 'Reservoirs': 0}. These may include sensor offset effects; retained, not labelled failures.
- Five-minute bins: 61,393; bins lacking 15 records or complete sensor values: 10,880.
- Usable causal feature rows: 47,951. Empty bins are not interpolated; windows reset at outages.

## Sensor patterns and regimes

Motor current regime fractions among scored five-minute means: {'offloaded': 0.5168, 'off': 0.4364, 'loaded': 0.0468}. Thresholds (<1 A off, 1–5.5 A offloaded, >5.5 A loaded) approximate the documented current levels. Mixed-state bins blur instantaneous regimes; these are descriptive categories, not verified machine modes.

Pressure channels co-move; digital states and current reflect compressor load cycling. The CSV includes negative pressure readings and strong distribution asymmetry. A global deviation model can confuse legitimate regime changes and seasonal temperature changes with anomalies. The heatmap in `correlations.png` documents association; repeated time-series rows invalidate independent-observation significance tests.

## Failure-aligned inspection


Event 1 (2020-04-18 00:00:00–2020-04-18 23:59:00): 809 observed bins in the ±24-hour context. See `figures/failure_1.png`. No causal diagnosis is inferred from the patterns.

Event 2 (2020-05-29 23:30:00–2020-05-30 06:00:00): 566 observed bins in the ±24-hour context. See `figures/failure_2.png`. No causal diagnosis is inferred from the patterns.

Event 3 (2020-06-05 10:00:00–2020-06-07 14:30:00): 887 observed bins in the ±24-hour context. See `figures/failure_3.png`. No causal diagnosis is inferred from the patterns.

Event 4 (2020-07-15 14:30:00–2020-07-15 19:00:00): 542 observed bins in the ±24-hour context. See `figures/failure_4.png`. No causal diagnosis is inferred from the patterns.

## Computed chronological test results

```json
{
  "rows": 27585,
  "start": "2020-05-01 22:25:00",
  "end": "2020-09-01 04:00:00",
  "robust": {
    "event_count": 3,
    "unobservable_events": 0,
    "detected_events": 1,
    "event_detection_rate": 0.3333333333333333,
    "alert_episodes": 5,
    "unannotated_alert_episodes": 3,
    "unannotated_observed_days": 93.39930555555556,
    "unannotated_alerts_per_observed_day": 0.03212015316554519,
    "alert_bin_fraction": 0.0007250317201377561,
    "events": [
      {
        "event_id": 2,
        "start": "2020-05-29 23:30:00",
        "end": "2020-05-30 06:00:00",
        "observed_bins": 79,
        "detected_during_event": false,
        "lead_time_hours": null
      },
      {
        "event_id": 3,
        "start": "2020-06-05 10:00:00",
        "end": "2020-06-07 14:30:00",
        "observed_bins": 552,
        "detected_during_event": false,
        "lead_time_hours": null
      },
      {
        "event_id": 4,
        "start": "2020-07-15 14:30:00",
        "end": "2020-07-15 19:00:00",
        "observed_bins": 55,
        "detected_during_event": true,
        "lead_time_hours": null
      }
    ]
  },
  "isolation": {
    "event_count": 3,
    "unobservable_events": 0,
    "detected_events": 0,
    "event_detection_rate": 0.0,
    "alert_episodes": 0,
    "unannotated_alert_episodes": 0,
    "unannotated_observed_days": 93.39930555555556,
    "unannotated_alerts_per_observed_day": 0.0,
    "alert_bin_fraction": 0.0,
    "events": [
      {
        "event_id": 2,
        "start": "2020-05-29 23:30:00",
        "end": "2020-05-30 06:00:00",
        "observed_bins": 79,
        "detected_during_event": false,
        "lead_time_hours": null
      },
      {
        "event_id": 3,
        "start": "2020-06-05 10:00:00",
        "end": "2020-06-07 14:30:00",
        "observed_bins": 552,
        "detected_during_event": false,
        "lead_time_hours": null
      },
      {
        "event_id": 4,
        "start": "2020-07-15 14:30:00",
        "end": "2020-07-15 19:00:00",
        "observed_bins": 55,
        "detected_during_event": false,
        "lead_time_hours": null
      }
    ]
  }
}
```

Unannotated alerts per observed day are a proxy for false-alert burden; incomplete failure annotation prevents verified false-positive rates or alert precision. Only three test events exist, so no narrow confidence interval is defensible. Warnings in the 24-hour preceding window can be coincidental, particularly under high alert burden. No RUL or calibrated failure probabilities are estimated.
