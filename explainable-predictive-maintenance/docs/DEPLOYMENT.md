# Deployment and local operation

## Run from the Git repository root

The project now lives in `explainable-predictive-maintenance/`; Git, GitHub Actions and the virtual environment remain at the repository root. Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './explainable-predictive-maintenance[dev]'
streamlit run explainable-predictive-maintenance/streamlit_app.py
```

On Windows activate `.venv\Scripts\activate`. Open [the local application](http://localhost:8501) after starting Streamlit. This address refers to each reader’s own computer and is not a hosted public demo. Startup reads committed historical artifacts; it does not download or retrain the full dataset.

## Reproduce Version 2

```bash
python explainable-predictive-maintenance/scripts/download.py
python explainable-predictive-maintenance/scripts/run_v2.py
python -m ruff check explainable-predictive-maintenance
python -m pytest explainable-predictive-maintenance/tests -q
```

The official download needs internet; modeling is CPU based. Reserve several GB of memory for the roughly 208 MiB uncompressed CSV and intermediates. V2 settings are in `configs/v2.json`; outputs are separate from frozen V1 metrics/model/demo. The historical `scripts/run_pipeline.py` is the V1 workflow and overwrites V1 output paths if executed: use a disposable checkout of `v1.0.0` for historical reproduction rather than rerunning it in the preserved V2 checkout.

## Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) as the repository owner and authorize GitHub access.
2. Choose the existing `Priynka-Roy/industrialguard-explainable-predictive-maintenance` repository and the published V2 branch.
3. Set the main file to `explainable-predictive-maintenance/streamlit_app.py` and Python to 3.11. Runtime requirements are in the project subdirectory beside the entry point.
4. Include committed `data/demo/`, model artifacts and required reports; exclude raw/full processed data, virtual environments, caches and secrets. The root `.streamlit/config.toml` configures appearance. No API secret is needed for historical monitoring.
5. Open the public page and navigate all eight sections, models, cases and explanations; advertise a public URL only after verification.

No public deployment URL has been verified. The owner’s interactive sign-in and terms acceptance are necessary for hosted publication; GitHub publication and local operation do not constitute a public deployment.
