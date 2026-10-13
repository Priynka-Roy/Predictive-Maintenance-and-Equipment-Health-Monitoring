# Methodology guide

The current methodology is [Version 2](V2_METHODOLOGY.md), with [explicit metric definitions](EVALUATION_DEFINITIONS.md), [experiment provenance](V2_EXPERIMENTS.md) and [computed results](V2_RESULTS.md). The immutable historical methodology and metrics are documented in [Version 1 baseline](V1_BASELINE.md) and `docs/v1/`.

Version 2 separates February training, March calibration and April development, and adds an earlier chronological development fold. Candidate/policy selection uses development evidence only. Previously inspected May–September data provide a retrospective comparison, not a fresh holdout. No external independent validation is available. Neither unusual scores nor relative health indicators are calibrated failure probabilities.
