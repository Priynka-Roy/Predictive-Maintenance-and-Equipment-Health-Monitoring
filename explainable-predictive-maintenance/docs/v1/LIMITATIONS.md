# Limitations and responsible interpretation

- **One equipment history:** the dataset concerns one metro APU. Performance does not validate mining machines, marine engines, agricultural equipment, other APUs, or a fleet. The target-company examples describe transferable skills only.
- **Sparse and imperfect reports:** four 2020 intervals are reported, three in the final test. They may not be independent. Absence of an annotation does not prove healthy operation. Maintenance dates and source row counts contain inconsistencies documented in the data card.
- **Anomaly ≠ failure:** normal operating transitions, environmental conditions, sensor artifacts, or changing load can trigger unusual scores. Event overlap is retrospective evidence of association; it does not prove reliable early warning.
- **No failure probability or RUL:** the health indicator is an empirical score percentile. There is no calibrated probability, survival model, or validated remaining-life estimate.
- **Aggregation:** five-minute means reduce compute and deployment size but can obscure short transients. Digital duty fractions and average-current regimes do not retain every state transition.
- **Observed exposure:** alert burden is measured per observed telemetry day. No alert inference is made during gaps; downtime and train operating time are not independently recorded.
- **Small tuning search:** two Isolation Forest subsample settings and a label-free weekly-stability objective are practical choices. They do not establish global optimality; a stable detector can still miss events.
- **Distribution shift:** validation score quantiles do not guarantee a future alert budget. A full validation ECDF makes the health score relative to that specific history.
- **Explanation scope:** robust deviations identify associated inputs. They are not SHAP decompositions of the forest or proven physical causes. Correlated sensor features can repeat the same underlying pattern.
- **Illustrative economics:** dashboard cost inputs are user assumptions. No actual downtime costs, prevented failures, return on investment, or realized savings are observed.
- **Historical demonstration:** the app displays precomputed historical outputs; it does not monitor a live asset or implement online model updates, authentication, enterprise observability, or production incident handling.

Future work should obtain expert-reviewed normal and fault labels; evaluate prospective warnings on multiple independent machines; model operating-regime conditioning; quantify decision costs with real operational evidence; assess drift; and add time-aware uncertainty only when enough events and assets are available.
