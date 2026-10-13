# Version 2 experiments and provenance

The experiment registry is generated from `configs/v2.json`; changes to that plan define a new experiment. It records model/feature configuration, seed, dates, fitted feature names/counts, threshold, persistence, computation and evaluation metrics. Results are computed from the official source, not entered by hand.

## Predeclared comparison

| Model ID | Family | Feature group |
|---|---|---|
| `robust_original` | Global robust max deviation | Original 52 |
| `robust_raw` | Global robust max deviation | Original 15 sensor levels |
| `robust_pressure` | Global robust max deviation | 12 pressure/engineering/context inputs |
| `covariance_pressure` | Ledoit–Wolf distance | Same pressure group |
| `regime_pressure` | Regime-relative robust deviation and empirical limits | Same pressure group |
| `isolation_reduced` | Fixed Isolation Forest | Reduced candidate list with training-only constant/correlation filtering |

Each model is crossed with four policies (quantile 0.98/0.995 × persistence 1/3 bins) and two chronological development folds: 48 development records. These are controlled comparisons, not 48 independent datasets. All candidates use the same complete causal-feature history to avoid changing exposure merely by selecting a smaller representation. This conservative common eligibility loses some rows that a standalone raw-only model could otherwise score.

Early fold: train February, calibrate March 1–20, evaluate March 20–April 1. Primary fold: train February, calibrate March, evaluate April. Intervals are half-open. Development records can differ in calibration thresholds even for the same trained model. Complexity and workload evidence are considered alongside the one primary-development event. The [methodology](V2_METHODOLOGY.md) specifies selection and ineligible per-model comparisons; [metric definitions](EVALUATION_DEFINITIONS.md) define event/episode ownership and exposure.

## Provenance boundaries

After development-only policy selection, the primary-fold fitted/calibrated bundle scores previously inspected May–September. Retrospective comparisons are exported separately and never passed to the selection objective. Frozen V1 methods/metrics remain at their original paths and in the V1 snapshot. Because V2 event definitions correct edge cases, any V1-policy recomputation with the V2 evaluator must be labeled distinctly from original V1 metrics.

No external independent validation was obtained. Digital/current regime agreement is a consistency audit, not field validation. Training correlations and feature reductions are associations, not physical causal claims. Computation timings depend on hardware and are reproducibility metadata rather than universal runtime promises.

## Reproduce

From the Git root after installing the project:

```bash
python explainable-predictive-maintenance/scripts/download.py
python explainable-predictive-maintenance/scripts/run_v2.py
```

The [JSON registry](../artifacts/v2/experiments.json) contains all 48 computed fold-level development records. The [CSV comparison](../artifacts/v2/experiments.csv) contains the 24 policy summaries aggregated across the two folds. [Selection](../artifacts/v2/selection.json) records the 24 two-fold policy summaries and per-model choices. See [computed results](V2_RESULTS.md) for policy ranking and actual measured numbers.
