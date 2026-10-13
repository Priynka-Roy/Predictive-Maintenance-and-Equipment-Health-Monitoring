# IndustrialGuard AI: turning compressor telemetry into reviewable maintenance evidence

A maintenance analyst needs measurements, model context and a manageable inspection workload, alongside honest evidence about documented faults. IndustrialGuard AI develops those skills on one real metro compressor APU. Mining, machinery, marine and energy employers are the intended audience; they did not supply data or validate the project.

## Data and original finding

The UCI MetroPT-3 CSV contains 1,516,948 readings and 15 channels from February–September 2020. Provenance checks distinguish stored nulls from time gaps and retain negative pressure readings rather than silently inventing labels. Source sampling/row-count discrepancies and the inconsistent maintenance date are documented in the [data card](DATA_CARD.md). A related 2022 paper concerns a different release and is not used to relabel this history.

V1 produced a complete causal-feature/Streamlit workflow but weak event sensitivity. Its robust detector overlaps one of three original test events; Isolation Forest overlaps none; neither demonstrates a preceding 24-hour warning. A low weekly alert-rate variability objective favored a quiet forest without ensuring useful sensitivity. Those results are frozen in [V1 baseline](V1_BASELINE.md), not overwritten by the upgrade.

## Version 2 engineering and statistical response

V2 keeps the existing application and introduces time-separated calibration/development, an earlier walk-forward fold, controlled original/raw/pressure/reduced feature groups, regularized multivariate distance and descriptive regime reference monitoring. Its predeclared quantile/persistence grid and workload objective expose sensitivity/burden trade-offs. Silence cannot win by default; the single April event receives only secondary selection credit. Evaluation handles boundary events, outages, warning definitions and episode attribution explicitly, with hand-calculated regression cases.

Training-fitted transformations and causal gap-reset windows guard against future fitting leakage. This does not restore independent validation: May–September was already examined in V1. [Computed V2 results](V2_RESULTS.md) report development and retrospective evidence separately, without relabeling observations as a new holdout. The selected covariance policy detects 0/1 April intervals and 1/3 retrospective intervals, with 15 unannotated episodes; reduced Isolation Forest detects 2/3 retrospectively with 22 unannotated episodes but is not selected by the workload objective. A July 40-minute preceding covariance warning remains retrospective and potentially coincidental. No external independent validation is available.

## Guided historical evidence

The app provides a documented historical air-leak case, an unannotated unusual-operation case and comparisons between implemented models. A reported fault interval supports historical overlap analysis; an unannotated alert supports review workload analysis. Neither establishes the physical cause, and a preceding alert may be coincidental. Case timestamps and actual outputs are in `artifacts/v2/cases.json`, allowing readers to check each story against telemetry.

Deviations show which inputs differ from training references. Regime-relative references add operational context. For covariance and forest models, displayed reference deviations are contextual evidence, not additive model contributions. The covariance model additionally exposes exact signed distance components, including negative terms caused by correlated inputs; these are statistical evidence rather than physical causes. These distinctions help an analyst formulate inspection questions without presenting correlations as causal explanations.

## Practical interpretation

The eight-section Streamlit app reads lightweight persisted outputs, offers selected-model visibility, compares versioned metrics and includes editable illustrative inspection/downtime costs. It does not monitor a live machine or demonstrate prevented downtime. Relative health is calibration typicality, not RUL or failure probability. The strongest portfolio outcome is a reproducible, reviewable analytical system with candid evidence limits.

Further progress requires expert-reviewed labels, verified physical states, independent machines, prospective warnings and real decision-cost evidence. More model complexity alone cannot supply those missing observations. [Deployment status](DEPLOYMENT.md) distinguishes local operation and GitHub publication from verified public hosting.
