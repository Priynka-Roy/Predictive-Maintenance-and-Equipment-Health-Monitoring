# Model card

## Purpose and artifact

Historical anomaly monitoring for one MetroPT-3 compressor APU. The persisted bundle in `artifacts/model.joblib` contains ordered feature names, training medians and scales, the selected scaler/Isolation Forest pipeline, validation thresholds, and the health-score reference. `artifacts/metrics.json` is the source of computed results. This is a portfolio model, not a certified maintenance system.

## Data and fitting

The official UCI 2020 CSV has 1,516,948 rows and 15 sensor channels. Observed timestamps span 2020-02-01 00:00:00–2020-09-01 03:59:50; median spacing is 10 seconds. Causal five-minute aggregation and rolling features yield 47,951 complete rows and 52 model inputs. No null sensors were observed, but 331 timestamp gaps exceed one minute.

Training is February; validation is March–April; test starts May. Reported April event windows are excluded from calibration. RobustScaler and model fitting use training only. The forest uses 200 trees, seed 42, and subsample size 256, selected from 256/1,024 by the smallest validation weekly raw-alert variability. Threshold quantile is 0.995; issued alerts need three consecutive scored five-minute windows. Features reset at missing bins and outages.

The robust baseline uses the maximum training median/IQR-standardized deviation. It is retained even though the forest is the display model. Selection is within the forest family and does not justify claiming superiority over the baseline.

## Final-test evidence

| Frozen policy | Test events detected | Alert episodes | Unannotated episodes | Unannotated alerts / observed day |
|---|---:|---:|---:|---:|
| Robust deviation baseline | 1/3 (33.3%) | 5 | 3 | 0.03212 |
| Isolation Forest (selected on validation) | 0/3 (0%) | 0 | 0 | 0.00000 |

Test observations: **27,585** usable five-minute rows, 2020-05-01 22:25:00–2020-09-01 04:00:00. Non-event observed exposure: **93.399 telemetry days**. Neither method issued a qualifying warning within 24 hours before any final-test event. The baseline detected only the July interval during the reported event. **Earlier failure detection was not demonstrated.**

The forest's zero alerts also mean zero detected events. Its conservative validation-stability selection is a poor failure-sensitive policy in this holdout; quiet operation is not evidence of success. The frozen test is retained without tuning it to improve the displayed results.

Validation itself had one event detected by neither model. The selected forest issued two validation episodes; the baseline issued ten. Candidate settings and validation-only quantile/persistence sensitivity are recorded in the metrics artifact. Only three test intervals prevent a reliable generalization estimate; no independent-row confidence interval is reported.

## Score meaning and explanations

Anomaly scores are larger for unusual patterns. Relative health is 100 minus the validation empirical score percentile. Attention categories use 95th/99th/99.5th validation percentiles. These are relative indicators, not calibrated failure probabilities. Standardized input deviations identify associated features, not exact forest contributions or physical causes.

## Computational evidence and reproducibility

The recorded run used Python 3.12.14, scikit-learn 1.6.1, pandas 2.2.3, and NumPy 2.2.6. One recorded pipeline run took 25.81 seconds and 1,000-row scoring took 0.0130 seconds on the development machine. These are measured local timings, not hardware-independent benchmarks or service guarantees. Rebuild with `python scripts/run_pipeline.py`; settings are in `configs/pipeline.json`.

## Risks and prohibited claims

No RUL, calibrated probability, fleet generalization, real-time monitoring, confirmed false-alarm precision, causal diagnosis, prevented downtime, or production validation is supported. Legitimate regime changes and sensor offsets can trigger deviations. Unreported event labels are incomplete. See [limitations](LIMITATIONS.md), [methodology](METHODOLOGY.md), and [data attribution](DATA_CARD.md).
