# Version 2 computed results

These are actual outputs from the official MetroPT-3 source. [Metrics](../artifacts/v2/metrics.json), [selection](../artifacts/v2/selection.json), [48-record experiment CSV](../artifacts/v2/experiments.csv), [experiment JSON](../artifacts/v2/experiments.json) and [artifact hashes](../artifacts/v2/manifest.json) make the tables checkable. Plan SHA-256: `5e8b9c422e53c945093b054253040cc1c463f49c6fe5f65db6f07cc5dc4f4a1e`.

## Selection and comparison

The development-only workload objective selects **`covariance_pressure_q0.98_p1`**: 12 pressure-engineered/context inputs, empirical 98th-percentile calibration limit and one-bin persistence. Its pooled development objective is 0.231281, issued-alert fraction 0.011226 and unannotated burden 0.938532 episodes per observed day. It is eligible under the predeclared workload limits, but detects **0/1** April events. Eligibility and a workload objective do not establish failure-warning adequacy.

The table reports each model’s development-selected policy, fitted on February and calibrated on March. Development columns are April only; selection aggregates both pre-May folds, so pooled selection values differ from April values.

| Model | Quantile / persistence bins | April detected | April unannotated episodes/day | Retrospective detected | Retrospective unannotated episodes | Retrospective episodes/day |
|---|---|---:|---:|---:|---:|---:|
| `robust_original` | 0.98 / 1 | 0/1 | 1.16032 | 1/3 | 15 | 0.16060 |
| `robust_raw` | 0.98 / 1 | 0/1 | 1.16032 | 1/3 | 15 | 0.16060 |
| `robust_pressure` | 0.98 / 1 | 0/1 | 1.16032 | 1/3 | 15 | 0.16060 |
| `covariance_pressure` | 0.98 / 1 | 0/1 | 1.16032 | 1/3 | 15 | 0.16060 |
| `regime_pressure` | 0.98 / 3 | 0/1 | 0.96693 | 1/3 | 3 | 0.03212 |
| `isolation_reduced` | 0.995 / 1 | 1/1 | 0.38677 | 2/3 | 22 | 0.23555 |

The selected covariance and three global robust variants have matching event-detection and alert-episode totals, but their bin-level alert masks are not identical. Covariance differs from original-feature robust monitoring in 7 April bins and 2 retrospective bins; raw/pressure robust variants differ from the original in 2 retrospective bins. Covariance adds a regularized correlated-distance interpretation without improving detected-event counts here. Extra original features do not improve these selected-policy event detections.

Reduced Isolation Forest detects the April report and two retrospective intervals, with greater retrospective burden than the frozen V1 policy. Its evidence is exploratory; it was not chosen by the predefined workload objective. The regime-relative policy lowers retrospective burden versus the new covariance policy, without additional detected events or a preceding July warning. More operational context does not automatically improve sensitivity.

## Timing, coverage and V1 comparison

The new covariance/global-robust policy has a July 15 qualifying first warning at 13:50, **40 minutes** before the 14:30 reported onset; the closest qualifying bin is 14:10, **20 minutes** before onset. The eligible episode onset is also 13:50. Neither it nor any other selected per-model policy has a qualifying warning before the May or June interval. Regime-relative and reduced-forest selected policies have no qualifying preceding warnings for any retrospective event. These descriptive retrospective warnings may be coincidental and are not independent early-warning validation.

The retrospective window has 27,585 usable five-minute observations, 95.78125 observed days and 93.399306 unannotated observed days, versus 123.170139 calendar days. June coverage is partial (552/631 expected bins); May and July are fully observed on the evaluation grid. April coverage is also partial (276/288). Partial intervals stay in the observable denominator; outages are not inferred as healthy.

Frozen V1 robust results remain **1/3 detected**, 3 unannotated episodes, 0.032120 episodes/day and no preceding warning. Frozen V1 forest remains **0/3**, no alerts and no warning. V2 covariance remains 1/3 detected but increases unannotated episodes to 15 (0.160601/day), with the descriptive July warning. Reduced forest has 2/3 retrospective detections and 22 unannotated episodes (0.235548/day). These are policy/evaluator/version comparisons, not proof of generalization or universally improved maintenance utility.

## Feature, regime and computation evidence

The 52-input training audit finds nearly redundant pressure and digital signals: TP3/Reservoirs correlation is 0.999993 and COMP/MPG is 1.0. The reduced forest retains 12 inputs after training-only filtering. Exact pairs/groups are in [feature audit](../artifacts/v2/feature_audit.json).

The pre-May scored-history regime audit counts off 10,656, unloaded 4,792, loaded 978, transition 3,736 and ambiguous 204 bins. Stable bins represent 80.654% of that history. Current-label agreement on stable bins is 100% by construction; it does not independently validate physical states. Among bins with unambiguous control fractions, current/control load agreement is 99.800%; the recorded cross-tab exposes disagreements rather than treating the derived label as ground truth. Mean COMP/outlet-complement disagreement is 0.009159. The two-fold experiment registry contains 48 records and documents fit/inference timings. This recorded run completes in 28.97 seconds on the execution host; timing is hardware dependent.

## Case evidence and limits

[Guided cases](../artifacts/v2/cases.json) cover April 18’s reported air leak, April 29 19:10–April 30 01:10’s unannotated unusual period, and July 15’s retrospective policy comparison. The unannotated case is selected outside reported intervals and their 24-hour context; lack of a report does not prove a false alarm.

May–September was inspected in V1 and remains retrospective. No external independent validation exists. There is one development event and three recurring benchmark intervals on one machine; no reliable generalization precision, calibrated failure probability, remaining life or realized savings can be inferred. The upgrade improves scientific provenance, edge-case evaluation and review tools; it does not establish reliable predictive maintenance.
