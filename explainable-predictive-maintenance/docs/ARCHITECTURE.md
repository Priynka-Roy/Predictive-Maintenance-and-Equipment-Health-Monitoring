# Architecture

A small Python package serves an explicit CPU batch pipeline and an eight-section Streamlit app. Heavy ingestion, fitting and experiments run offline. The app caches compact committed historical artifacts and does not download or train during startup.

```text
Official UCI source → provenance/schema/time checks → causal five-minute features
                                                          ↓
                       February training → March calibration → development folds
                                                          ↓
                   candidate models × predefined threshold/persistence policies
                                                          ↓
                      development-only selection → retrospective May–September
                                                          ↓
                   versioned models + metrics + experiment registry + demo
                                                          ↓
                                   Streamlit historical monitoring and review
```

| Component | Responsibility |
|---|---|
| `scripts/download.py`, `data.py` | Official retrieval, immutable raw files, coverage checks and causal resampling |
| `features.py`, `regimes.py` | Gap-reset temporal/engineering inputs and corroborated descriptive groups |
| `models.py` | Historical V1 model implementation |
| `monitoring.py` | V2 reference fits, statistical/isolation alternatives and interpretations |
| `evaluation.py` | Gap-aware persistence, episode attribution, event/warning/exposure definitions |
| `scripts/run_v2.py`, `configs/v2.json` | Frozen development protocol, experiment selection and separate V2 export |
| `app/main.py`, `streamlit_app.py` | Cached interactive historical presentation |
| `artifacts/`, `data/demo/`, `reports/` | Frozen V1 outputs plus separate versioned V2 outputs |
| `docs/v1/`, `artifacts/v1/`, `v1.0.0` | Historical documentation, artifact snapshot and tagged source |
| `tests/`, root `.github/workflows/` | Analytical regression, app smoke and continuous checks |

The project directory is `explainable-predictive-maintenance/` under the Git root. Scripts resolve artifact paths relative to that project, without developer-specific absolute paths. Raw and full processed outputs remain local. Joblib bundles are trusted repository artifacts; loading an untrusted pickle is outside the supported workflow. CI installs the project subdirectory and uses small fixtures rather than downloading the full source on every commit.
