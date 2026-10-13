# IndustrialGuard AI: turning compressor telemetry into reviewable maintenance evidence

## Problem and relevance

A maintenance analyst needs more than an anomaly flag: they need to see the measurements, understand the alert policy, assess the review burden, and know whether historical faults were actually detected. A metro compressor APU provides a concrete industrial setting because air supply supports train subsystems. This project develops those analytical and engineering skills in a public, reproducible portfolio. Mining, machinery, marine and energy applications motivate transferability, without asserting that any named employer supplied data or validated the results.

## Why this dataset

The official UCI MetroPT-3 release combines real multivariate telemetry with reported air-leak periods. Downloaded originals were preserved; retrieval hashes and timestamps were recorded. The actual CSV contains 1,516,948 records with 15 signals from 2020-02-01 00:00:00 through 2020-09-01 03:59:50. It has no null sensor readings, yet 331 gaps exceed 60 seconds. The largest gap is 48.03 hours. This distinction shaped the pipeline: missing time was not silently filled.

Source investigation found inconsistent row-count and sampling claims in the PDF and UCI page, an inconsistent maintenance date, and a related paper describing a different 2022 release. The project uses observed CSV timestamps and the four 2020 reported intervals; it does not import the paper's 2022 events. [Data card and references](DATA_CARD.md).

## Statistical challenges and choices

Sensor values reflect operating regimes, strong temporal dependence, and correlated pressure channels. Some pressure readings are negative; they were investigated and retained rather than silently relabelled as failure or clipped. Only four reported intervals exist and normal-operation labels are incomplete. Supervised failure classification, RUL prediction, precision claims, and narrow independent-row confidence intervals would therefore overstate the evidence.

The implemented alternative uses five-minute trailing windows and 52 causal inputs: sensor levels, pressure differential, rolling means and variation, first differences, and an outlet-duty change indicator. Windows reset across outages. Train-only transformations and February/March–April/May-onward separation prevent future fitting leakage. Current-based regimes are approximate descriptions of window means rather than trusted equipment-state labels.

## Modeling and engineering

An interpretable maximum robust-deviation baseline is compared with a 200-tree Isolation Forest. Two forest subsample settings are assessed using validation-week alert-rate stability. Reported validation fault windows are excluded from threshold calibration; the remainder is described as unannotated rather than certified normal. Frozen 99.5th-percentile thresholds and three-window persistence generate operational alerts.

The Python package separates data validation, causal features, scoring, health conversion, and evaluation. The command-line pipeline generates data audits, correlations, distributions, event-aligned figures, saved model artifacts and a compact real-data demo. Streamlit provides eight sections with interactive Plotly charts, score histories, explanation cases, event metrics and editable illustrative decision costs. Heavy analysis runs explicitly, while the app reads cached precomputed outputs.

## Actual holdout results

| Frozen policy | Test events detected | Alert episodes | Unannotated episodes | Unannotated alerts / observed day |
|---|---:|---:|---:|---:|
| Robust deviation baseline | 1/3 (33.3%) | 5 | 3 | 0.03212 |
| Isolation Forest (selected on validation) | 0/3 (0%) | 0 | 0 | 0.00000 |

Test observations: **27,585** usable five-minute rows, 2020-05-01 22:25:00–2020-09-01 04:00:00. Non-event observed exposure: **93.399 telemetry days**. Neither method issued a qualifying warning within 24 hours before any final-test event. The baseline detected only the July interval during the reported event. **Earlier failure detection was not demonstrated.**

The forest's zero alerts also mean zero detected events. Its conservative validation-stability selection is a poor failure-sensitive policy in this holdout; quiet operation is not evidence of success. The frozen test is retained without tuning it to improve the displayed results.

These findings are useful precisely because they do not support an inflated predictive-maintenance claim. Stability optimization and aggressive persistence produced a quiet forest but poor event sensitivity. The baseline found one event with low unannotated burden, yet it missed the other two and did not warn in advance. No threshold was retuned using the final test to make the result look better.

## Business interpretation

The current forest policy should not be presented as adequate failure warning. The dashboard instead lets an analyst inspect relative deviations, review baseline evidence, and understand the consequences of a conservative alert budget. Unannotated episodes express inspection workload under incomplete labels, not verified wasted work. Cost controls illustrate assumptions; the data contains no observed inspection costs, prevented downtime or realized savings.

## Lessons and next work

The strongest lessons are methodological: verify the exact data release, distinguish nulls from time gaps, enforce causal availability, preserve a held-out history, report failures of the selected policy, and evaluate exposure and event counts honestly. A production-oriented continuation would start with expert-reviewed fault and normal labels, operating-regime conditioning, multiple machines and prospective warnings. More model complexity alone would not resolve those evidence gaps.

The project demonstrates a complete analytical application and a critical assessment of its utility. Deployment preparation and publication status are documented separately; a repository and screenshots do not imply a verified live deployment.
