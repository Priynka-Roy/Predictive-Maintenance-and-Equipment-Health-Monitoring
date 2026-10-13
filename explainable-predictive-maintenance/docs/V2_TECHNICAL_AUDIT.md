# Version 2 technical audit

This audit inspects the published V1 source, configuration, stored metrics, application, tests and deployment files. Priorities reflect consequences for scientific interpretation and operation. V1 findings remain valid historical observations; implementation fixes are tracked through V2 code, tests and generated reports.

| Priority | Finding and consequence | V2 response |
|---|---|---|
| P0 | `evaluation.py` admits only fully contained intervals. Boundary events disappear; events wholly inside outages can be excluded by observation endpoints. | Explicit calendar evaluation windows, events owned once by onset and observable/unobservable denominators. |
| P0 | Partition-local evaluation loses warning context; episode clipping can change onset interpretation. | Distinguish earliest/closest qualifying warning, episode onset and during-event overlap; test crossings and gaps. |
| P0 | May–September was already examined in V1. Calling it a new holdout would misstate independence. | Frozen V1 table; label V2 May–September comparisons retrospective; no benchmark-driven policy selection. |
| P1 | Minimum weekly alert-rate variability rewards an almost silent forest and does not address failure sensitivity. | Development-only burden-constrained selection, anti-silence eligibility, stability/complexity trade-offs and secondary review of the single event. |
| P1 | March–April calibrates thresholds and evaluates candidates in the same history. | Time-separated training, calibration and development plus an additional walk-forward development fold. |
| P1 | The 52-input representation repeats correlated pressure/rolling information; global scaling mixes operating modes. | Controlled original, reduced and pressure-engineered feature comparisons; regularized covariance and regime-relative monitoring. |
| P1 | Average-current cutoffs alone label regimes without digital-signal corroboration. | Training-defined digital/current agreement rules, transition and ambiguous states, and published regime audit. |
| P1 | The 99.5th percentile and three-bin persistence are conservative, potentially losing event sensitivity. | Predeclared calibration quantiles and persistence grid; select only using development evidence. |
| P1 | Scores/health and explanations are forest-centric despite poor V1 sensitivity. | Explicit selected-model display, calibration-relative typicality, regime reference deviations and model-specific explanation limits. |
| P1 | Tests cover basic gaps but need hand-calculated boundary, outage, repeated-episode and lead-time cases. | Expanded evaluation and train-only/causality regression tests. |
| P2 | Application exposes eight useful sections but lacks guided versioned comparison and case evidence. | Preserve navigation; add frozen V1, V2 development/retrospective provenance, comparison and real historical cases. |
| P2 | Heavy pipeline and app artifacts share paths; rerunning V1 would overwrite historical outputs. | Separate V2 artifacts/reports/demo and retain immutable V1 outputs; document explicit rebuild commands. |
| P2 | User moved project into a subdirectory while Git/CI/runtime config remain at root. | Preserve relocation, adjust root CI/install/run/deployment paths and verify from repository root. |
| P2 | Documentation and career language need to distinguish monitoring from validated prediction. | Updated cards, limitations, methods, case study, experiment/results tables and deployment status. |

Existing strengths retained: source provenance/hash checks, schema validation, no interpolation over outages, causal trailing features, train-only fitting, compact persisted demo, eight-section Streamlit architecture, automated checks and honest V1 weak results. Residual limits include one machine, sparse imperfect annotations, temporal dependence, unvalidated physical regimes and no external independent validation.
