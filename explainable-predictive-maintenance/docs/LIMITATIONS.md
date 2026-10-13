# Limitations and responsible interpretation

- **One equipment history:** one metro compressor APU does not validate other assets or a fleet. Metso, Sandvik, John Deere and Wärtsilä are audience examples; they supplied no data and have not validated this project.
- **Sparse reports:** four intervals, one in primary development and three in the old benchmark, may describe recurring issues rather than independent failures. Unannotated time is not certified healthy.
- **Retrospective evidence:** May–September was already inspected in V1. V2 comparisons there cannot establish independent generalization. No genuinely new external validation is available.
- **Dependent observations:** five-minute rows and overlapping chronological folds are dependent. Row counts are not failure sample sizes. No narrow confidence intervals or independent-fold significance claims are justified.
- **Selection assumptions:** workload limits and target alert fraction are predefined portfolio choices. A secondary credit for one April event is fragile and can still influence selection. Development-only selection improves provenance, not statistical certainty.
- **Anomaly and warning:** event overlap is historical association. A preceding warning may be coincidental, especially for a high-burden policy. Unannotated episodes are not confirmed false positives.
- **Aggregation and regimes:** five-minute means hide transients and mixed states. Digital/current agreement supports descriptive regimes, not certified physical labels; sparse or ambiguous groups fall back to global references.
- **Covariance and shift:** regularization stabilizes distance estimation but does not make telemetry Gaussian. Score quantiles and relative-health references need not retain their calibration meaning under future shift.
- **Exposure:** observed usable bins determine monitoring exposure. Gaps and excluded feature warm-up do not count as observed healthy time. Calendar time and exposure differ; no alarms are inferred in outages.
- **Explanation:** deviation and contribution charts identify statistical evidence and associations. Correlated inputs can repeat information; neither reference comparison nor covariance contribution proves a physical cause. Forest deviations are not exact forest feature attribution.
- **Health and economics:** relative health is neither calibrated failure risk nor RUL. Cost controls are illustrative user assumptions, with no observed savings, prevented downtime or ROI.
- **Historical app:** cached, precomputed outputs do not constitute live monitoring, online adaptation or production validation. A GitHub repository and screenshots do not imply a deployed public app.

Next evidence priorities: expert-reviewed normal/fault labels, prospectively collected data, independent machines, verified equipment-state signals and real decision costs. More model complexity cannot replace those data.
