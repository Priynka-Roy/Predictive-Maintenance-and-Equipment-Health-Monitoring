# IndustrialGuard AI model card

## Purpose and data

The application supports explainable historical monitoring and maintenance review on one UCI MetroPT-3 compressor. It does not estimate calibrated failure risk, remaining useful life or a certified intervention. Four reported air-leak intervals are incomplete fault evidence; unannotated time is not known healthy. Named industrial employers supplied no data and have not validated this work.

## Versioned methods

V1 uses 52 causal inputs, a maximum robust-deviation baseline and a training-scaled 200-tree Isolation Forest. Its original benchmark detects 1/3 and 0/3 intervals respectively, without qualifying preceding warnings. Original artifacts and metrics are preserved. See [frozen baseline](V1_BASELINE.md).

V2 implements global robust models over original/raw/pressure groups, Ledoit–Wolf regularized covariance on pressure-engineered features, regime-relative robust monitoring and Isolation Forest on a training-filtered reduced group. Robust centers/IQRs fit February only with scale floors. Covariance fits clipped training standardized inputs (−8 to 8); inference departures remain unclipped. Its score is squared distance, with empirical calibration limits rather than a Gaussian probability claim. Reduced inputs remove near-constant features and sequentially filter absolute training correlations at 0.98; feature order therefore matters. Isolation Forest uses 200 trees, max_samples 256 and seed 42.

Pressure inputs include sensor pressures, TP2−TP3, TP3−Reservoirs, trailing TP3 variability, lagged oil-temperature change, current, oil temperature and outlet duty. See `monitoring.py` and the generated experiment registry for exact names/counts. Causal features reset at telemetry gaps.

## Regime assumptions

Current cutoffs 1 A and 5.5 A are corroborated with COMP/DV_eletric duty fractions: off/unloaded require COMP ≥0.8 and outlet duty ≤0.2; loaded requires COMP ≤0.2 and duty ≥0.8. Mixed duty fractions between 0.2 and 0.8 indicate transition; disagreements are ambiguous; missing state inputs are unknown. Agreement with current labels on corroborated bins is true by construction, not independent validation. References can fit sufficiently populated transition/ambiguous groups too. Unknown or sparse training groups use global reference; insufficient calibration groups use the global threshold. The audit reports population and disagreement evidence.

## Policies and selection

Quantile/persistence choices and chronological folds are declared in `configs/v2.json`. Calibration determines score limits; development determines eligible policies and the selected portfolio display. May–September cannot influence ranking and remains retrospective. [Methodology](V2_METHODOLOGY.md) and [experiments](V2_EXPERIMENTS.md) document workload assumptions and a secondary credit for the single development event. [Results](V2_RESULTS.md) record the actual selected policy and trade-offs: covariance/pressure inputs, quantile 0.98 and one-bin persistence. It detects 0/1 April and 1/3 retrospective intervals; reduced Isolation Forest detects 2/3 retrospectively but is not selected by the workload objective. These weak/sparse results do not validate failure warning.

## Explanations and limitations

Robust deviations are differences from global or regime-relative training references. Reference deviations are contextual associations, not additive score decompositions or exact forest attribution. For covariance only, the app additionally shows signed components `r_j × (precision @ r)_j` that sum to squared distance; correlated inputs can produce negative components. These are statistical components, not physical causes. Relative health uses the empirical calibration score distribution, while attention categories summarize score ranges separately from persistence-based issued alerts. Normal means typical of that reference, not proven healthy. No external independent validation, reliable confidence interval or observed economic benefit is available.
