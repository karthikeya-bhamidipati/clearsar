# ClearSAR — SAR-to-Optical Cloud Removal (Asia Subset)

An M.Sc.-level deep learning project: reconstructing cloud-covered Sentinel-2 optical
imagery using paired Sentinel-1 SAR data, with a controlled comparison of three
architectures (CNN U-Net, hybrid Swin-Transformer U-Net, pure ViT), plus an
uncertainty-quantification extension.

**Goal:** complete, hands-on understanding of deep learning — not a publication.

> **Current phase: [plan v4](docs/project_plan_v4.md)** — build *one* model step by step (residual
> pass-through, many more training scenes, better losses, global attention) until its reconstructions are as
> close to the truth as the data allows, with a prediction → result → explanation for every stage in
> [`docs/lab_notes.md`](docs/lab_notes.md). The 26-run architecture comparison below is Phase 1 ("Stage 0:
> exploration").

Full plan, reasoning, experiment list, week-by-week schedule and glossary:
**[`docs/project_plan.md`](docs/project_plan.md)** — read this first.

How everything fits together end to end (data flow, folder roles, model
internals, the Kaggle/GitHub workflow): **[`docs/architecture.md`](docs/architecture.md)**.

**Everything explained in one document** — every model, hyper-parameter, metric, experiment, result
and likely examiner question: **[`docs/ClearSAR_Project_Guide.docx`](docs/ClearSAR_Project_Guide.docx)**.

---

## Try it in two minutes (trained models included)

The repository contains the complete output of the final Kaggle run in [`outputs/`](outputs/): all
results tables and figures, training logs and **all 26 trained checkpoints** — no training or data
download needed.

```bash
git clone https://github.com/karthikeya-bhamidipati/clearsar.git
cd clearsar
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Results (final run, 15 000 steps per model, 2 × T4, 6.1 h)

| Model | PSNR Asia (dB) | PSNR other continents (dB) | SSIM Asia | SAM Asia (°) |
|---|---|---|---|---|
| Cloudy input (identity) | 21.18 | 20.95 | 0.740 | 10.45 |
| Tiny CNN baseline | 27.01 | 28.01 | 0.844 | 9.45 |
| Model A — U-Net (3 seeds) | 26.33 ± 0.06 | 27.82 ± 0.30 | 0.847 | 9.43 |
| Model B — Swin U-Net (2 seeds) | 26.41 ± 0.20 | 27.68 ± 0.39 | 0.851 | 9.55 |
| Model C — Pure ViT (2 seeds) | 26.90 ± 0.36 | 27.19 ± 0.41 | 0.844 | 8.91 |
| U-Net 3-seed ensemble | 26.56 | 28.26 | 0.857 | 9.08 |

- All L1-trained models recover **5–7 dB** over the cloudy image.
- No architecture is clearly best overall; differences are of the size of seed noise and far smaller than
  scene-to-scene variation — a 3-layer CNN is competitive on PSNR.
- **Under thick cloud (60–100 %) the ViT is 2–4 dB better** in Asia — its effective receptive field is
  56 px versus 21 px for the U-Net.
- A random patch split would inflate PSNR by **+5.1 dB** (leakage check).
- Only the learned variance head gives reasonably calibrated uncertainty (coverage 0.83–0.89 vs 0.08 for
  MC dropout).

Full analysis: chapter 13 of the project guide, or the **Experiments** page of the app.

## Reproduce from scratch

```bash
# 1. set up the environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. find out which SEN12MS-CR scenes are in Asia (streams small files only)
cd src/data
python find_region_scenes.py scan --out ../../data/scans/scenes.csv --continent Asia

# 3. extract a capped, manageable subset (run this on Kaggle with internet, not locally)
python find_region_scenes.py extract --csv ../../data/scans/scenes.csv \
    --continent Asia --max-scenes 18 \
    --extra ROIs1868_summer:7 ROIs1970_fall:23 \
    --out ../../data/raw
```

### Faster alternative: pre-mirrored Kaggle datasets (recommended)

The TUM server is slow and rate-limited regardless of client, so on Kaggle, attach
these four community mirrors as Inputs instead of streaming from TUM — together
they're the complete SEN12MS-CR dataset (~169 scenes, all 4 seasons):

- Spring: [rajaryan1726/sen12mscr-spring-triplets](https://www.kaggle.com/datasets/rajaryan1726/sen12mscr-spring-triplets)
- Summer: [shinrapb/sen12ms-cr-summer](https://www.kaggle.com/datasets/shinrapb/sen12ms-cr-summer)
- Fall: [shinrapb/sen12ms-cr-fall](https://www.kaggle.com/datasets/shinrapb/sen12ms-cr-fall)
- Winter: [bigoone/sen12ms-cr-winter](https://www.kaggle.com/datasets/bigoone/sen12ms-cr-winter)

Then run the local variants, which read the already-attached files directly —
no network involved:

```bash
cd src/data
python find_region_scenes.py scan-local --out ../../data/scans/scenes.csv --continent Asia
python find_region_scenes.py extract-local --csv ../../data/scans/scenes.csv \
    --continent Asia --max-scenes 18 --out ../../data/raw
```

See `docs/project_plan.md` §2 for the full data pipeline, §3–6 for models and
experiments, and §7 for the week-by-week schedule.

## Training on Kaggle (the complete project in one 12-hour session)

[`notebooks/kaggle_pipeline.py`](notebooks/kaggle_pipeline.py) runs **every experiment in
`docs/project_plan.md`** in one session: index the attached SEN12MS-CR files, split **by scene**
(Asia → train/val/test, one scene per other continent → out-of-continent test), cache the
patches once, benchmark the GPU, pick **one step budget for all runs** so everything fits, train
the jobs below (one per GPU in parallel on T4 ×2), then ensemble + receptive-field analysis and
export everything.

Jobs, in priority order (if time runs short, the last ones are skipped — never the core):

| Experiment | Runs |
|---|---|
| core | `A_unet` U-Net · `B_swin` Swin-bottleneck U-Net · `C_vit` pure ViT (all L1) |
| baselines | `identity` (output = cloudy input) · `tiny_cnn` (3 conv layers) |
| E1 learning curves | `A/B/C_f25`, `A/B/C_f50` (25 % and 50 % of the training patches) |
| leakage check | `A_random` (random patch split instead of scene split) |
| E2 domain shift | every model is evaluated on the out-of-continent test set |
| E3 U-Net probes | `A_noskip` · `A_l2` · effective receptive field (post-hoc) |
| E4 GAN | `A_gan` (λ = 100) · `A_gan10` (λ = 10); λ = ∞ is `A_unet` |
| E5 Swin probes | `B_noshift` · `B_win8` · `B_win16` (= global attention) |
| E6 ViT probes | `C_nopos` · `C_p16` |
| E7 uncertainty | `A_unc` variance head · `A_mcd` MC dropout (20 passes) · `A_ens` 3-seed ensemble |
| seeds | `A_unet_s43`, `A_unet_s44`, `B_swin_s43`, `C_vit_s43` |

**On Kaggle**

1. New notebook → Settings: **Accelerator: GPU T4 ×2**, **Internet: on**. Add the four
   datasets above as inputs.
2. Smoke test (≈30 min, same code path with a tiny budget) — one cell, run interactively:
   ```python
   !git clone -q https://github.com/karthikeya-bhamidipati/clearsar.git /kaggle/working/clearsar
   !CLEARSAR_QUICK=1 python /kaggle/working/clearsar/notebooks/kaggle_pipeline.py
   ```
   It must end with `DONE - download ...` and a RUN STATUS list with every job `done`.
3. Full run — replace the cell with
   ```python
   !rm -rf /kaggle/working/*
   !git clone -q https://github.com/karthikeya-bhamidipati/clearsar.git /kaggle/working/clearsar
   !python /kaggle/working/clearsar/notebooks/kaggle_pipeline.py
   ```
   then **Save Version → Save & Run All (Commit)**. It stops itself after ~11 h (before Kaggle's
   12 h limit; change with `CLEARSAR_HOURS=…`).
4. When the version finishes: Output tab → download `clearsar_output.zip` → extract into
   `outputs/` in this folder (ignored by git) → `streamlit run app.py`.

Every setting can be overridden with a `CLEARSAR_*` environment variable — see `CONFIG` at the
top of the script.

## Frontend

```bash
pip install -r requirements.txt
streamlit run app.py
```

| Section | Page | What it shows |
|---|---|---|
| Deep learning | **Home** | drag-to-compare cloudy input vs reconstruction, headline numbers, the three architectures, data map |
| | **Model comparison** | one chart (PSNR / SSIM / SAM / FID, with seed error bars), leaderboard, cloud-cover and country breakdowns |
| | **Experiments** | one card per experiment: question, prediction, measured change, automatic verdict (Supported / Contradicted / Within noise), and a write-up scaffold |
| | **Explore patches** | every model on held-out patches, against the cloudy input or the truth, with error and uncertainty maps |
| Expert system | **Advisor** | decision (trust · model · fit for purpose · action) with certainty factors, visual explanation, how / why-not reasoning, facts, downloadable report |
| | **Knowledge base** | architecture diagram, what-if simulator, all rules and thresholds |
| About | **How it works** | the pipeline and its limitations |

## Expert system (ClearSAR Advisor)

A rule-based, explainable decision-support system on top of the networks — see
[`docs/expert_system.md`](docs/expert_system.md). Facts come from the input patch, the models' outputs
(disagreement, uncertainty, radar reliance via occlusion, band saliency) and the experiment results; a
YAML knowledge base of 35 IF-THEN rules with certainty factors is chained by a forward-chaining MYCIN-style
engine (`src/expert/`). Every decision comes with the rules that produced it, the alternatives considered and
why other rules did not fire.

```python
from src.expert.advisor import Advisor
advice = Advisor("outputs/checkpoints", "outputs/results").advise(x, use_case="vegetation_monitoring", region="in_region")
advice.verdict("trust"), advice.verdict("action"), advice.result.fired
```

## Tests

```bash
python -m pytest tests -q                       # models, preprocessing, metrics vs scikit-image, splits
python tests/make_synthetic_data.py --out /tmp/synth   # fake SEN12MS-CR for an end-to-end smoke run
```

## Project layout

```
clearsar/
├── README.md                  <- you are here
├── app.py                     <- Streamlit frontend entry point (pages in frontend/)
├── requirements.txt
├── .gitignore
├── docs/
│   ├── project_plan.md        <- the full plan (read first)
│   └── setup_guide.md         <- git/GitHub/Kaggle setup instructions
├── data/
│   ├── scans/                 <- scenes.csv etc. (small, tracked in git)
│   └── raw/                   <- extracted GeoTIFFs (NOT tracked — see .gitignore)
├── src/
│   ├── data/                  <- scene finder, preprocessing, dataset/dataloader
│   ├── models/                <- U-Net, Swin-U-Net, ViT, PatchGAN, uncertainty heads
│   ├── training/               <- training loop, AMP, checkpointing, LR range test
│   ├── evaluation/             <- metrics, tiled evaluation, plots, receptive fields
│   └── expert/                 <- ClearSAR Advisor: knowledge base, inference engine, XAI
├── notebooks/kaggle_pipeline.py <- the Kaggle training/evaluation script (logic lives in src/)
├── tests/                      <- unit tests + synthetic-data generator
├── outputs/                    <- final Kaggle run: results, logs, all 26 checkpoints (tracked)
└── experiments/
    ├── checkpoints/            <- model weights (NOT tracked)
    ├── logs/                   <- training logs, metrics CSVs (tracked)
    └── results/                <- final tables, plots, figures for the write-up (tracked)
```

## Status

- [x] Data pipeline: index, scene split, preprocessing, cache, baselines
- [x] Model A (U-Net), GAN ablation, uncertainty heads
- [x] Model B (Swin-bottleneck U-Net)
- [x] Model C (pure ViT)
- [x] All experiments E1–E7, leakage check, seeds — final Kaggle run 2026-10-21/22
- [x] Frontend (Streamlit) and expert system (ClearSAR Advisor)
- [x] Project guide with results (`docs/ClearSAR_Project_Guide.docx`)
- [ ] Final write-up: prediction → result → explanation for each experiment
