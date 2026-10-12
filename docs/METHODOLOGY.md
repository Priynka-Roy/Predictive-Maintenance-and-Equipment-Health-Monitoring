# Methodology

## Observation and feature availability

The unit of analysis is a five-minute, right-labelled, right-closed telemetry window. Its timestamp is the earliest time the complete window can be scored. Analogue means retain their measurement units; digital means become active-duty fractions. A window with fewer than 15 readings is unavailable. Empty intervals are not interpolated. Range anomalies are reported, not automatically clipped away.

Features include all 15 signals, TP2 minus TP3, analogue first differences, 15-minute and one-hour trailing means and population standard deviations, and changes in the outlet-valve duty fraction. Incomplete rolling windows are excluded from model scoring. Features reset after invalid bins or telemetry gaps; they cannot borrow observations across an outage. Current-based states (off below 1 A, offloaded from 1–5.5 A, loaded above 5.5 A) are approximate summaries of window-average current, not independently verified engineering labels.

## Chronological design

Training precedes March 1, 2020. Validation covers March 1 through April 30. Final test starts May 1. Boundary timestamps and settings are recorded in `configs/pipeline.json`. Trailing features may use earlier historical observations across a partition boundary because those observations were available at inference time. Transformers and model parameters are fitted only on training observations. Reported failures are not used to fit the unsupervised detectors.

April's one reported interval belongs to validation, and three later intervals belong to test. This is a single chronological holdout on one machine, not repeated independent validation across a fleet. The final test is used only after the model and alert threshold are selected.

## Baseline and model selection

The statistical baseline takes the largest absolute training-standardized deviation across features. Centering uses the training median; scale uses the training interquartile range, floored at the larger of 1% of the center magnitude or 0.01. This is an interpretable anomaly statistic rather than a parametric hypothesis test; no normality claim is made.

Isolation Forest uses a persisted `RobustScaler` and 200 trees with seed 42. Two candidate subsample sizes (256 and 1,024) are compared. Selection minimizes the standard deviation of contiguous validation-week alert fractions, using a candidate-specific 99.5th-percentile threshold from unannotated validation windows. Documented April failure windows are excluded from calibration and the selection reference; the remaining windows are not certified healthy. This label-free stability heuristic limits tuning; it is not evidence of optimal failure detection. The threshold controls the empirical raw-bin budget in validation, not the future false-positive rate. The baseline is calibrated to the same quantile for comparison.

## Alerts and evaluation

An alert requires three consecutive above-threshold five-minute windows. Gaps reset persistence. Alert episodes group consecutive persistent flags. The timestamp of the first persistent flag is the operational alert time; the first two elevated windows do not count as an issued alert.

An event is detected if at least one persistent alert overlaps its documented interval. Only events fully contained in the partition enter its event denominator. A preceding alert within 24 hours is reported separately as descriptive lead time. The earliest qualifying preceding flag can have a large lead time even if it was unrelated to the later failure; this measure cannot establish predictive specificity.

An episode that overlaps no documented event contributes to **unannotated alert burden**. The denominator is non-event observed-bin exposure divided by 288 bins/day. This differs from calendar-day exposure and from confirmed false alarms: unreported anomalies, inspection needs, and failures may exist. The analysis intentionally avoids accuracy, PR-AUC, and a claimed true false-positive rate because normal ground truth is incomplete.

## Relative health and explanations

For an Isolation Forest score `s`, relative health equals `100 × (1 − empirical validation CDF(s))`. Higher values indicate more typical behavior relative to validation. A health value near zero indicates an unusual validation-relative score; it is not a failure probability or an absolute physical health measurement.

Attention categories use the 95th, 99th, and 99.5th validation score percentiles: Normal, Monitor, Inspect, and High attention. These category thresholds are separate from the three-window alert rule. Sensor explanations show training-standardized deviations and associated historical measurements. They explain unusual input values, not exact additive contributions to the forest, causal fault mechanisms, or certified diagnoses.

## Sensitivity and uncertainty

The generated report compares alert-budget quantiles, persistence rules, and subsample candidates using validation only. Test-month breakdowns describe temporal behavior under the frozen policy and are not separate tuning opportunities. With only three final-test intervals, event-level uncertainty is too weak for a robust population estimate. Correlated minute-level observations are not treated as independent trials to manufacture narrow confidence intervals. No confidence interval is claimed for generalization to other machines.
