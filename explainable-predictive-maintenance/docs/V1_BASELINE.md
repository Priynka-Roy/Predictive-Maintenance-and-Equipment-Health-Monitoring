# Version 1 frozen baseline

Version 1 is the published historical implementation at commit `7a5747c`, retained by the immutable `v1.0.0` tag. Its outputs in `artifacts/metrics.json`, `artifacts/model.joblib`, `data/demo/monitoring.parquet`, and original reports remain unchanged. Historical documentation is retained in `docs/v1/`. Recomputations with Version 2 evaluation definitions are distinct results, not replacements for this record.

## Method and configuration

The UCI MetroPT-3 2020 release contains 1,516,948 records from one compressor and 15 sensor channels. Five-minute means require at least 15 source records and complete channels. Empty/invalid windows are excluded, without interpolation. Features reset after gaps and use the current and preceding observations only.

The 52 inputs comprise 15 sensor levels, seven analog first differences, trailing 3-bin and 12-bin means and standard deviations for seven analog channels, pressure difference TP2−TP3, and an outlet-control transition magnitude. February supplies training; March–April supplies validation; May–September supplies the original final test. March–April documented event observations are excluded from the calibration reference; other observations are unannotated, not certified healthy.

The statistical baseline is the maximum absolute deviation from training medians divided by floored training IQRs. Isolation Forest uses training-fitted RobustScaler, 200 trees and seed 42. Candidates use `max_samples` 256 and 1024. The 256 candidate was selected by the smallest standard deviation of validation weekly alert fractions: 0.004229 versus 0.004625. This objective does not measure maintenance usefulness.

Each model uses its 99.5th-percentile validation reference score, followed by three consecutive above-threshold five-minute observations. Thresholds: robust 940.519367; Isolation Forest 0.723676. Persistence resets over telemetry gaps. Health is a relative Isolation Forest score percentile, not a failure probability or remaining life.

## Frozen historical evaluation

| Model | Detected original test intervals | Total alert episodes | Unannotated episodes | Episodes per unannotated observed day |
|---|---:|---:|---:|---:|
| Robust deviation | 1/3 | 5 | 3 | 0.032120 |
| Isolation Forest | 0/3 | 0 | 0 | 0.000000 |

The original test contains 27,585 usable observations, from 2020-05-01 22:25 through 2020-09-01 04:00, and 93.399306 unannotated observed telemetry days. The robust detector overlaps the July 15 report only. Neither model demonstrates a qualifying warning during the 24 hours before any of the three original test events. Original validation detection is 0/1 for both models.

These numbers reproduce the stored Version 1 evaluator. It includes only events fully contained between a partition’s first and last observation, so some boundary and outage cases are mishandled. Version 2 addresses those definitions without revising this frozen historical table.

## Reasons for Version 2

Silence can win the weekly-stability objective while missing faults. Calibration and model selection share the March–April period, with only one documented interval. Current-only regimes are descriptive rather than validated states; correlated and redundant features can dominate deviations. Boundary events, lead-time windows and outage exposure need explicit treatment. May–September has already been inspected, so future comparisons there are retrospective, not independent validation. Version 2 improves selection, evaluation and explanations without manufacturing a fresh holdout or a reliable failure-prediction claim.
