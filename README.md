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

See `docs/project_plan.md` §2 for the full data pipeline, §3–6 for models and
experiments, and §7 for the week-by-week schedule.

## Project layout

```
clearsar/
├── README.md                  <- you are here
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
├── notebooks/                  <- Kaggle/exploratory notebooks (keep thin — logic lives in src/)
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
