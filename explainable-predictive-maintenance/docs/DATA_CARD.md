# MetroPT-3 data card

## Provenance and attribution

IndustrialGuard AI uses the **UCI MetroPT-3 release (dataset 791)**: one train's air-compressor Air Production Unit (APU), observed during real operation in 2020. It is not a simulated fleet. No data was supplied by Metso, Sandvik, John Deere, or Wärtsilä; those industries motivate transferable monitoring skills only.

Source: [UCI MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset). Dataset DOI: [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R). Original measurements and redistributed demo derivatives are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The demo changes the source by aggregating time windows, generating historical features, computing model scores, and selecting a compact subset. Software uses the MIT license.

Suggested attribution: Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). *MetroPT-3 Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW3R.

The download script records retrieval information and file hashes. Raw files are retained locally and excluded from Git. The source CSV contains 1,516,948 observations, an index, a timestamp, and 15 sensor channels. The UCI page lists the CSV as approximately 208.2 MB. The retrieved CSV is 218,300,507 bytes (approximately 208.2 MiB), with SHA-256 `db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24`. Retrieval was recorded at 2026-10-12 22:20:57 UTC (October 13 locally in Helsinki). Actual timestamps run from 2020-02-01 00:00:00 to 2020-09-01 03:59:50. Median stored spacing is 10 seconds; there are 331 gaps over one minute, with a largest gap of 172,918 seconds. No null sensor values or duplicate complete rows were observed. Exact audit details are in `reports/provenance.json` and `reports/data_quality.json`.

## Signal dictionary

| CSV field | Unit | Interpretation in source documentation |
|---|---|---|
| TP2 | bar | Compressor pressure |
| TP3 | bar | Pneumatic-panel pressure |
| H1 | bar | Pressure associated with cyclonic-separator discharge |
| DV_pressure | bar | Dryer discharge pressure; source links zero to loaded operation |
| Reservoirs | bar | Downstream reservoir pressure |
| Motor_current | A | One phase of the motor current |
| Oil_temperature | °C | Compressor oil temperature |
| COMP | binary signal | Intake-valve control, active during off/offloaded operation |
| DV_eletric | binary signal | Compressor outlet control; original spelling is retained |
| Towers | binary signal | Active drying tower |
| MPG | binary signal | Command for starting the compressor under load |
| LPS | binary signal | Low-pressure switch |
| Pressure_switch | binary signal | Dryer discharge detection |
| Oil_level | binary signal | Low-oil-level detection |
| Caudal_impulses | digital signal | Air-flow pulse signal |

These descriptions support interpretation, not a certified physical diagnosis. The ingestion schema checks actual field spellings. Source timestamps have no explicit timezone offset; the analysis preserves naive source time and does not imply UTC or silently localize it.

## Documented 2020 events

| Event | Start | End | Source report |
|---|---|---|---|
| 1 | 2020-04-18 00:00 | 2020-04-18 23:59 | Air leak, high stress |
| 2 | 2020-05-29 23:30 | 2020-05-30 06:00 | Air leak, high stress |
| 3 | 2020-06-05 10:00 | 2020-06-07 14:30 | Air leak, high stress |
| 4 | 2020-07-15 14:30 | 2020-07-15 19:00 | Air leak, high stress |

The raw dataset has no per-row failure target. These intervals are historical report annotations used for retrospective evaluation, not proof that every other minute is normal. The source repeats event number #1 for the May interval; this project assigns a stable chronological event ID 2 while preserving its time interval. Four reported intervals may represent recurring issues on the same equipment rather than independent failures.

## Documentation discrepancies and decisions

1. **Sampling:** UCI describes both 1 Hz acquisition and 0.1 Hz stored observations. The bundled PDF additionally states 15,169,480 rows at 1 Hz, ten times the website and actual distributed CSV count. The pipeline measures timestamp deltas in the actual CSV and reports the observed cadence. It does not fabricate missing 1 Hz observations. Acquisition rate and distributed-file cadence can differ; the available material does not establish the exact downsampling procedure.
2. **Maintenance date:** UCI attaches “Maintenance on 30Apr at 12:00” to the May 29–30 event. This is inconsistent with the event chronology. The report text is preserved as uncertain; neither April 30 nor an assumed May 30 repair is used as a modeling label.
3. **Related paper:** [Veloso et al., Scientific Data 9, 764 (2022)](https://doi.org/10.1038/s41597-022-01877-3) describes a different MetroPT release on Zenodo: January–June 2022, 10,979,547 observations, 20 variables including GPS, and three failures. That paper provides APU context, but its 2022 labels, variable count, and sampling claims are not transferred to this 2020 UCI release. Full text is also available through [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9747912/).
4. **Missingness:** UCI's “no missing values” claim concerns stored records. A CSV without nulls can still contain gaps in time. Both are checked separately.

## Appropriate and inappropriate use

Appropriate: research, reproducible historical condition monitoring, alert-budget analysis, and explanation of signal deviations. Inappropriate: calibrated failure probabilities, remaining-life claims, safety-critical control, conclusions about other machines without validation, or interpreting all unreported alerts as confirmed false positives.

## Version 2 evaluation provenance

Version 2 adds no external dataset and does not relabel a previously inspected period as independent validation. February training, March calibration and April development remain within this one equipment history. May–September is a retrospective benchmark. Digital/current agreement and ambiguous/transition groups are descriptive derived annotations; they do not modify the source failure reports or establish physical mode truth. Original source files and frozen V1 metrics remain preserved.
