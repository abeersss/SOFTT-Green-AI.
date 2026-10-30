# SOFTT Green AI experiments

Reproducible research for **Accuracy Budgets and Routing Overhead in Green Edge AI for Environmental Sensor Networks** by Abeer Saleh Alshammari.

**Status: initial experimental research draft, not submitted or accepted.** The contribution and methodology require author review. This project is separate from FedTRA and Turbo-NAFS.

## Run the experiment

Use Python 3.12 and an isolated environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_data.py
python experiment.py
```

The download script fetches the original public archive, verifies SHA-256, and stores it as `data/beijing.zip`. No credentials or Kaggle account are required. Data are not included in Git to avoid duplicating the upstream archive.

## Data provenance

S. Chen, **Beijing Multi-Site Air Quality**, UCI Machine Learning Repository, 2017. DOI: [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G). License: **CC BY 4.0**.

Official source: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data

SHA-256 of the downloaded archive:
`b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`

The source has 420,768 hourly records across 12 monitoring stations. The experiment retains 409,180 current/next-hour pairs with observed PM2.5 and an exactly one-hour target interval. It forecasts next-hour PM2.5 from current observations and historical lags.

## Chronological methodology

| Target partition | Period | Eligible pairs |
|---|---|---:|
| Training | Before 2015 | 188,119 |
| Validation | 2015 | 102,193 |
| Test | 2016–2017 | 118,868 |

Models: persistence; median-imputed/scaled ridge; histogram gradient boosting; and a volatility-gated ridge/boosting cascade. Thresholds are selected on validation MAE allowances of 0%, 2%, 5%, and 10%. Validation allowances are not guaranteed future error bounds. No future target or backward-filled value is used as a feature.

Controls include 30 random-routing seeds matched to realized test invocation probability, station-level metrics, and 2,000 paired daily-cluster bootstrap resamples. Random gates are descriptive controls, not independent model-training replications. The test set has already been inspected; changes inspired by its results require transparent exploratory labeling.

## Original measured results

| Policy | Test MAE | Test RMSE | Heavy evaluations | Median batch ms |
|---|---:|---:|---:|---:|
| Persistence | 10.07169 | 20.04812 | 0% | 0.57 |
| Ridge | 10.23332 | 19.15266 | 0% | 44.69 |
| Always boosting | 9.74849 | 20.13070 | 100% | 664.03 |
| Cascade with 2% validation allowance | 9.90215 | 20.16604 | 82.78% | 689.98 |

Errors are in µg/m³. Times are CPU batch measurements for the entire test set, not per-message edge-device latency. Shared-host variability limits small timing comparisons. Timings exclude feature construction, loading, transmission, and training. Persistence timing includes array extraction.

**The cascade avoids 17.22% of heavy evaluations but has no observed median throughput gain.** Its allocation also underperforms matched-rate random routing on MAE. With a 5% validation allowance, the selected policy is ridge alone: 14.86× faster standalone batch inference and 4.97% higher test MAE than boosting, with lower RMSE. Negative findings are retained.

**No physical energy or carbon measurement was performed.** The manuscript's energy model is an explicit sensitivity analysis. Avoided evaluations and faster runtime must not be relabeled as measured joule or carbon savings. No B5G radio or live network was evaluated.

## Files and reproduction outputs

- `experiment.py`: complete executed training, validation, inference timing, controls, and uncertainty analysis.
- `download_data.py`: checksum-verified UCI download.
- `results/metrics.csv`, `station_metrics.csv`, `validation_sweep.csv`: original tabular outputs.
- `results/metadata.json`, `numerical_versions.json`: dataset checksum, model/split details, bootstrap intervals, and numerical environment.
- `results/run.log`: original execution output.
- `results/test_predictions_sample.csv`: first 100 saved predictions for easy inspection; **not used to recompute pooled metrics**.
- `experiment.py` regenerates the full 118,868-row `results/test_predictions.csv.gz` locally. Its checksum is recorded in `results/full_predictions_manifest.json`.
- `build_paper.py`, `manuscript_content.json`, `SOFTT_Green_AI.tex`: draft-generation source and manuscript text.
- `IEEEtran.cls` is required for PDF compilation; install the IEEEtran package through your TeX distribution (or use Overleaf).
- `Claude_Review_Prompt.txt`: detailed independent methods/reference QA instructions.

Generate the editable draft and plots with `python build_paper.py`; compile `SOFTT_Green_AI.tex` twice with `pdflatex`. The paper-building script embeds original-run narrative numbers. **After rerunning experiments, revise every affected narrative value and derived comparison before regenerating a revised paper.** Do not silently mix timings from different hosts.

## QA status and limitations

Saved full predictions were checked against every reported pooled MAE and RMSE. The four-page A4 IEEE PDF and editable Word review copy were visually inspected; PDF fonts were checked for embedding. Conference submission, PDF eXpress validation, similarity checking, and independent peer review have not been performed. Only one dataset and one primary chronological split were used. The gate-first implementation, stronger learned gates, additional rolling origins, and hardware power measurement remain untested.

ChatGPT with Codex assisted with experimental code, plotting, and drafting. Claude has not yet reviewed this version. AI assistance and author responsibility must be disclosed accurately in any submitted manuscript. References must be independently checked before submission. There is no guarantee of acceptance.
