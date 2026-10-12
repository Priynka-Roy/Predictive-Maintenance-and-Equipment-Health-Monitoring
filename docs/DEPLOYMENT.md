# Deployment and local operation

## Local demonstration

Use Python 3.11 and a virtual environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
streamlit run streamlit_app.py
```

On Windows activate `.venv\Scripts\activate` instead. The committed demo uses historical MetroPT-3 outputs and requires no raw download. A full rebuild requires the official source ZIP, more memory and an explicit batch command.

## Streamlit Community Cloud

1. Push the public repository, including `data/demo/`, `artifacts/`, and required `reports/` outputs. Exclude `data/raw/`, full processed data, virtual environments, caches, and secrets.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/), authorize the GitHub repository, and choose **Create app**.
3. Select the repository and branch; set the main file to `streamlit_app.py` and Python to **3.11** in advanced settings.
4. Deploy with the root `requirements.txt`. The app resolves repository paths and `.streamlit/config.toml` controls presentation. No API secrets are needed for the historical demo.
5. Verify the public page, navigate all eight sections, select a time period and explanation case, and confirm charts and metrics render. Record the verified URL only after this check.

The project prepares these deployment inputs. A repository URL alone is not proof of deployment; a live URL should only be advertised once independently opened and verified. If Community Cloud needs a user sign-in or GitHub authorization, that interactive step remains with the account owner.

## Reproduce the analysis

```bash
python scripts/download.py
python scripts/run_pipeline.py
python -m ruff check .
python -m pytest -q
```

The pipeline covers preparation, feature generation, fitting, chronological evaluation, report generation, and demo export in one command. It uses settings in `configs/pipeline.json`; changing those settings defines a new experiment. Retain provenance and reports when comparing runs.

Source retrieval needs internet access. Training is CPU based and has no GPU requirement. The full CSV is roughly 208 MiB uncompressed; reserve additional memory for parsing and resampling and disk space for the archive and outputs. Exact runtime depends on hardware and is recorded in generated run outputs where available.
