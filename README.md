# ClearSAR — SAR-to-Optical Cloud Removal (Asia Subset)

An M.Sc.-level deep learning project: reconstructing cloud-covered Sentinel-2 optical
imagery using paired Sentinel-1 SAR data, with a controlled comparison of three
architectures (CNN U-Net, hybrid Swin-Transformer U-Net, pure ViT), plus an
uncertainty-quantification extension.

**Goal:** complete, hands-on understanding of deep learning — not a publication.

Full plan, reasoning, experiment list, week-by-week schedule and glossary:
**[`docs/project_plan.md`](docs/project_plan.md)** — read this first.

How everything fits together end to end (data flow, folder roles, model
internals, the Kaggle/GitHub workflow): **[`docs/architecture.md`](docs/architecture.md)**.

---

## Quick start

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

## Training on Kaggle (the whole comparison in one run)

[`notebooks/kaggle_pipeline.py`](notebooks/kaggle_pipeline.py) does everything: index the
attached SEN12MS-CR files, split **by scene** (Asia → train/val/test, one scene per other
continent → out-of-continent test), cache the patches once, train all five runs under the
same protocol, evaluate, and export tables, figures, checkpoints and frontend samples.

| Run | What it is |
|---|---|
| `A_unet` | Model A — U-Net, L1 |
| `B_swin` | Model B — U-Net with a shifted-window Swin bottleneck, L1 |
| `C_vit` | Model C — pure ViT (patch 8, learned positional embeddings), L1 |
| `A_gan` | Model A + PatchGAN, L1 × 100 + adversarial (E4) |
| `A_unc` | Model A + Laplace variance head (E7) |
| `identity` | baseline: output = cloudy input (every model must beat it) |

1. Kaggle notebook → **Accelerator: GPU T4**, **Internet: on**, add the four datasets above.
2. One cell:
   ```python
   !git clone -q https://github.com/karthikeya-bhamidipati/clearsar.git /kaggle/working/clearsar
   !python /kaggle/working/clearsar/notebooks/kaggle_pipeline.py
   ```
   (or paste the whole script into a cell). **Save Version → Save & Run All.**
   Default budget: 15,000 steps per run (plan §4); set `CLEARSAR_QUICK=1` for a ~15 min smoke run:
   `!CLEARSAR_QUICK=1 python /kaggle/working/clearsar/notebooks/kaggle_pipeline.py`
3. Download `/kaggle/working/clearsar_output.zip` from the notebook's Output tab and
   extract it into `outputs/` in this folder (ignored by git).

Every setting (patch caps, steps, runs, paths) can be overridden with a `CLEARSAR_*`
environment variable — see the `CONFIG` block at the top of the script.

## Frontend

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tabs: **Overview** (headline numbers, map of scenes per split), **Results** (in-domain vs
out-of-continent, cloud-cover bins, per-country incl. India, uncertainty calibration),
**Training** (validation curves, parameter counts, GPU time), **Gallery** (every model on
the same test patches) and **Live inference** (run any checkpoint on CPU on a gallery patch
or your own S1 + S2 GeoTIFFs, download the georeferenced prediction).

## Tests

```bash
python -m pytest tests -q                       # models, preprocessing, metrics vs scikit-image, splits
python tests/make_synthetic_data.py --out /tmp/synth   # fake SEN12MS-CR for an end-to-end smoke run
```

## Project layout

```
clearsar/
├── README.md                  <- you are here
├── app.py                     <- Streamlit frontend (reads outputs/)
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
│   └── evaluation/             <- PSNR/SSIM/SAM/UCE/AUSE, plotting
├── notebooks/kaggle_pipeline.py <- the Kaggle training/evaluation script (logic lives in src/)
├── tests/                      <- unit tests + synthetic-data generator
├── outputs/                    <- extracted Kaggle output (NOT tracked)
└── experiments/
    ├── checkpoints/            <- model weights (NOT tracked)
    ├── logs/                   <- training logs, metrics CSVs (tracked)
    └── results/                <- final tables, plots, figures for the write-up (tracked)
```

## Status

- [ ] Week 1 — data pipeline, splits, baselines
- [ ] Week 2 — Model A (U-Net, L1)
- [ ] Week 3 — Model A GAN ablation
- [ ] Week 4 — Model B (Swin-bottleneck U-Net)
- [ ] Week 5 — Model C (pure ViT)
- [ ] Week 6 — learning curves (E1), uncertainty (E7)
- [ ] Week 7 — domain-shift evaluation (E2), final results
- [ ] Week 8 — write-up

(Update this checklist as you go — it's the fastest way to see progress at a glance.)
