# Architecture

The batch pipeline and Streamlit presentation share a small Python package. Heavy data processing runs explicitly from the command line. The app reads compact precomputed outputs and never downloads the full dataset or retrains on startup.

```text
Official UCI ZIP → immutable local raw CSV + description PDF
                 ↓ schema / time / range / missingness checks
             five-minute trailing sensor windows
                 ↓ causal features and chronological partitions
          robust baseline + scaled Isolation Forest candidates
                 ↓ validation selection and frozen thresholds
              test scoring and event / alert analysis
                 ↓
      model bundle + compact demo + reports + figures
                 ↓
        eight-section Streamlit application
```

| Component | Responsibility |
|---|---|
| `scripts/download.py` | Official retrieval and provenance hashes |
| `scripts/run_pipeline.py` | Reproducible batch orchestration |
| `configs/pipeline.json` | Dates, windows, seed, candidates, alert policy |
| `src/industrialguard/data.py` | Schema validation and right-closed resampling |
| `src/industrialguard/features.py` | Causal rolling features and operating-state summaries |
| `src/industrialguard/models.py` | Training-only preprocessing, anomaly scores and health |
| `src/industrialguard/evaluation.py` | Persistent alerts, episodes, event overlap and exposure |
| `app/main.py` | Cached interactive historical demonstration |
| `streamlit_app.py` | Portable hosting entry point |
| `artifacts/` | Persisted model and metadata |
| `data/demo/` | Compact attributed historical data |
| `reports/` | Computed validation, EDA and performance evidence |
| `tests/` | Schema, feature causality, model, evaluation and app checks |

Paths resolve from the repository, not from a developer's home folder. Raw and processed full-data outputs stay local. Deployment includes compact inputs, saved artifacts, and reports. Joblib model bundles are trusted repository artifacts and must not be replaced with untrusted uploaded pickle files.

GitHub Actions installs the project with development dependencies, runs Ruff and pytest. It does not redownload the 208 MB source on every commit. Full real-data reproduction is an explicit local workflow; tests use small fixtures for fast deterministic checks.
