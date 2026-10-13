# Version 2 methodology

Version 2 compares practical monitoring alternatives on the existing MetroPT-3 history. It improves experimental discipline and reviewability; it does not establish reliable predictive maintenance. No new external validation data are available. May–September was inspected in Version 1 and is a retrospective benchmark.

## Frozen protocol

The executable plan is `configs/v2.json`. All intervals use naive source timestamps and half-open calendar boundaries. Both development folds train on February 1–March 1. The early fold calibrates March 1–20 and evaluates March 20–April 1; the primary fold calibrates March 1–April 1 and evaluates April 1–May 1. Thus calibration and evaluation are time-separated. The early fold contains no reported event and tests workload stability; the primary fold contains one April event, too little evidence for reliable generalization. Folds reuse training and some calibration history, so they are not independent replications.

Training-only fits determine centers, scales, covariance, correlation filtering and regime references. Calibration-only score references determine thresholds and relative health. Features use current/past telemetry, reset at gaps and never use future windows. Boundary-crossing evaluation is specified in [metric definitions](EVALUATION_DEFINITIONS.md).

## Controlled models and features

| Candidate | Representation | Purpose and assumptions |
|---|---|---|
| Robust original | V1 52 inputs | Preserve the interpretable global reference with a new selection policy. |
| Robust raw | Original sensor levels | Test the effect of removing rolling/difference redundancy. |
| Robust pressure | Pressure engineering and causal dynamics | Test engineering features with a transparent deviation score. |
| Regularized covariance | Pressure engineering | Capture correlated departures with shrinkage; distance is not a chi-square failure probability. |
| Regime-relative robust | Pressure engineering | Compare observations to training references in corroborated operating groups; sparse/ambiguous states need fallback. |
| Isolation Forest reduced | Training correlation-filtered representation | Compare nonlinear isolation after reducing redundant inputs, with fixed 200 trees, max_samples 256 and seed 42. |

Robust scaling limits domination by units and outliers but does not guarantee robustness to every shift. Covariance regularization handles correlated features and unstable inversion; it is not proof of multivariate normality. Operating groups must be interpreted through the generated digital/current audit, not treated as ground truth physical modes. Aggregated digital fractions blur switching.

## Policies and selection

Each candidate is assessed with calibration score quantiles 0.98/0.995 and persistence 1/3 bins. This grid is declared before retrospective scoring. Persistence issues alerts only after the required consecutive usable above-threshold bins and resets over missing time.

Eligibility requires development unannotated episode burden at most 2 per observed day, issued alert fraction at least 0.001, and alert fraction at most 0.20. Silence is ineligible. Among eligible policies the predeclared objective combines distance from a 0.01 alert fraction, alert-rate stability (weight 0.05), model complexity (penalty 0.002), a burden penalty (0.10 × burden/budget), and capped secondary event credit (at most 0.05). The one April event cannot be the sole objective. Workload targets are transparent portfolio assumptions, not an industrial service-level agreement. If no globally eligible policy exists, the run stops instead of declaring a winner. For an individual comparison model without an eligible candidate, its least-objective policy may be displayed with explicit ineligibility; it cannot win global selection.

The policy is selected on development folds only. May–September scores and event metrics cannot alter features, thresholds, persistence or ranking. Retrospective changes can reflect evaluation fixes as well as monitoring changes; frozen V1 results remain separately identified.

## Interpretation

Health describes calibration-relative typicality; attention categories guide review, not failure probabilities or remaining life. Standardized deviations describe unusual measurements relative to training references. For covariance and forest models, these are contextual evidence rather than additive score decompositions, causal explanations or exact model attribution. Engineering hypotheses require independent inspection evidence. Unannotated episodes indicate review workload under incomplete labels, not verified false positives. Editable cost assumptions are illustrative and do not establish realized savings.
