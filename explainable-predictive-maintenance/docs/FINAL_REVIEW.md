# Version 2 review and delivery status

## Scientific review

V1 source is preserved by `v1.0.0`; metrics/model/config and historical documentation are snapshot separately and original metric/demo paths are unchanged. V2 separates train/calibration/development and selection from the already inspected retrospective period. Six monitoring/feature candidates and four policies across two folds produce 48 development records. The selection registry is persisted before benchmark scoring, with a plan hash and artifact manifest.

The selected regularized covariance policy detects 0/1 April and 1/3 retrospective events, with 15 unannotated benchmark episodes. Reduced forest detects 2/3 retrospectively but is not selected by the predefined workload objective. A 40-minute preceding July covariance warning is retrospective and potentially coincidental. No external independent validation, calibrated failure probability, RUL or realized savings is available. The stronger contribution is experimental provenance and reviewability, not validated failure prediction.

## Software and application review

The existing eight-section Streamlit architecture is retained, with model comparison, selected-model clarity, frozen V1/experimental V2 provenance, historical guided cases, training/regime reference deviations and exact signed covariance components. Startup reads compact precomputed outputs without downloading or retraining. Empty-alert cases remain valid UI states. CPU batch generation, versioned artifacts, logged progress and pinned dependencies support reproduction. GitHub Actions accounts for the user’s project-subdirectory relocation.

Analytical verification covers training-only fits, feature causality, telemetry gaps, boundary/month event ownership, outages, partial coverage, warning timing and episode attribution. Final lint, automated application tests, artifact/hash checks, local browser evidence and publication outcomes are recorded in the delivery report; The complete local suite passed 62 tests with no warnings; Ruff passed. The independent evaluation reviewer recomputed selection and all six model metrics, confirmed exact matches, verified month-level episode sums, and found no blocking leakage defect.

## Delivery and operation

Existing public repository: [Priynka-Roy/industrialguard-explainable-predictive-maintenance](https://github.com/Priynka-Roy/industrialguard-explainable-predictive-maintenance). Development branch: `feature/industrialguard-v2`. Publication uses the existing history, without force-pushing or replacing V1. [Pull request #1](https://github.com/Priynka-Roy/industrialguard-explainable-predictive-maintenance/pull/1) passed both GitHub Actions checks and was merged into `main` on October 13, 2026. The completed V2 is preserved by `v2.0.0`; `v1.0.0` remains unchanged. No V2 history rewrite or force-push was used.

[Local app](http://localhost:8501) requires Streamlit running on the reader’s machine. No public hosted URL has been verified. Streamlit Community Cloud still requires the owner’s sign-in, repository authorization and terms acceptance. [Deployment instructions](DEPLOYMENT.md) provide the subdirectory entry point and installation commands.

## Residual limitations and next evidence

One machine, few imperfect fault reports, correlated observations, approximate regimes and previously inspected benchmark data limit inference. Workload targets are portfolio assumptions. Expert-reviewed labels, prospective collection, independent equipment and real maintenance costs are the next evidence priorities. [Full limitations](LIMITATIONS.md), [computed results](V2_RESULTS.md) and [metric definitions](EVALUATION_DEFINITIONS.md) explain the practical boundaries.

## Final perspectives

- Applied statistics: no blocking leakage defect; empirical limits avoid an unsupported Gaussian assumption, and sparse dependent failures prevent precise generalization claims.
- Industrial analytics: guided real cases connect pressure/current/temperature patterns with monitoring context; labels and equipment regimes remain approximate.
- ML engineering: 62 local and CI tests, portable source, versioned artifacts, data/config/source hashes and an independently reproduced selection support maintainability.
- Reliability: the selected model misses the development event and raises benchmark workload; no certified inspection, failure probability or cost-saving claim is supported.
- Hiring perspective: the contribution is reproducible statistical engineering and honest evaluation; external equipment validation is the principal next evidence requirement.
