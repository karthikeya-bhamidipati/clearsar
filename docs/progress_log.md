# Progress Log

A running record of what's been done, decisions made, and why — kept so the
reasoning behind the current state of the repo isn't lost. Update this as you
go; don't rely on git history alone to explain *why*.

---

## 2026-10-03 — Project setup and Week 1 data pipeline kickoff

### Local environment
- Installed Git for Windows (was present via winget but not on PATH for this
  session) and created the Python venv (`.venv`) with `requirements.txt`
  installed.
- Initialized the git repo locally and made the first commit (scaffold: plan,
  architecture doc, scene-finder script, folder structure).
- Fixed local git author identity to match the real GitHub/Kaggle account
  (`karthikeya-bhamidipati`) instead of a wrong default.

### GitHub
- Created **[github.com/karthikeya-bhamidipati/clearsar](https://github.com/karthikeya-bhamidipati/clearsar)**
  (public — made public specifically so Kaggle could `git clone` it without
  needing an access token; see "Kaggle data acquisition" below).
- Pushed the initial scaffold and all subsequent commits there.

### First attempt: streaming scan from TUM (abandoned)
- `find_region_scenes.py scan` streams the SEN12MS-CR archives directly from
  the TUM university FTP server. Tried this both locally and on a Kaggle
  notebook.
- **Finding:** the TUM server itself is the bottleneck (~1.5 MB/s sustained,
  confirmed by direct measurement), not the client — Kaggle wasn't
  meaningfully faster than local for this specific transfer. A full scan
  across all 4 seasons would have taken multiple hours.
- Cancelled this approach rather than wait it out.

### Second attempt: pre-mirrored Kaggle datasets (adopted)
- Searched Kaggle and found 4 community-uploaded mirrors that together
  reconstruct the complete, correct SEN12MS-CR dataset (single cloudy/clear
  pair, *not* the time-series SEN12MS-CR-TS variant):
  - Spring: [rajaryan1726/sen12mscr-spring-triplets](https://www.kaggle.com/datasets/rajaryan1726/sen12mscr-spring-triplets) (CC0, 41 scenes)
  - Summer: [shinrapb/sen12ms-cr-summer](https://www.kaggle.com/datasets/shinrapb/sen12ms-cr-summer) (CC0, 49 scenes)
  - Fall: [shinrapb/sen12ms-cr-fall](https://www.kaggle.com/datasets/shinrapb/sen12ms-cr-fall) (58 scenes)
  - Winter: [bigoone/sen12ms-cr-winter](https://www.kaggle.com/datasets/bigoone/sen12ms-cr-winter)
  - Considered (and rejected) `mahmoud7abib/sen12mscrts`: wrong dataset
    variant (time-series), only 7 scenes, no license, mis-tagged, 4 years
    stale — not a usable substitute.
- Added two new CLI commands to `find_region_scenes.py`:
  `scan-local` / `extract-local` — same logic as the streaming versions, but
  read directly from already-attached local/Kaggle-input files instead of
  streaming from TUM. No project scope change; same dataset, same pipeline,
  just a faster acquisition path. Pushed to GitHub (`00b6b8a`).
- Attached all 4 datasets as Kaggle notebook Inputs. Kaggle materializes
  large attached datasets onto session disk lazily in the background —
  `scan-local` had to be re-run a few times as more of the data became
  available (27 scenes found → 117 → 175 scenes once fully synced).

### Day-1 result: the real Asia scene count
Final `scan-local --continent Asia` result (supersedes the plan's rough
"35–45 scenes" guess):

- **175 scenes found in total across all 4 seasons** (slightly more than the
  official ~169 — some overlap across the 4 mirror uploads is expected).
- **37 scenes in Asia, 26,608 patches** — comfortably more than needed.
  Spans China, Japan, Korea, India (Rajasthan + Maharashtra), Kazakhstan,
  Iran, Turkey, Yemen, Saudi Arabia, Myanmar, Iraq, Taiwan, Lebanon,
  Pakistan, and more.
- Saved to `data/scans/scenes.csv` (tracked in git — small file).

### Extraction — hit Kaggle's disk quota, had to cut it short
- Ran `extract-local --continent Asia --max-scenes 18 --out ../../data/raw`
  on Kaggle to copy the 18 highest-patch-count Asia scenes (all 3 modalities)
  into `data/raw/`.
- **Problem:** `/kaggle/working` is capped at ~19.5 GiB for this session. 18
  scenes × 3 modalities was never going to fit (≈28 GB estimated). The output
  size climbed past 15 GiB mid-copy, so the run was interrupted deliberately
  (`KeyboardInterrupt`) rather than letting it crash on a full disk.
- A diagnostic pass (file counts + min file size per scene/modality folder)
  showed exactly one season had copied cleanly before the interrupt:
  **`ROIs1868_summer` scenes 31, 119, 127, 133 — complete across s1, s2 and
  s2_cloudy**, file counts matching `scenes.csv`'s `n_patches` exactly
  (775 / 782 / 772 / 784 patches respectively, 3,113 patches total).
  `ROIs1970_fall` had only partial, inconsistent coverage (s1-only for most
  scenes, one truncated `s2_cloudy` folder) and was deleted
  (`shutil.rmtree`) rather than kept in a half-usable state.
  `ROIs1158_spring` and `ROIs2017_winter` were never reached.
- **Net result of this session's extraction: 4 complete Asia scenes
  (Summer: China ×2, Iran, India/Maharashtra), 3,113 patches, in
  `data/raw/ROIs1868_summer/`.** Not the full 18-scene target — the
  remaining ~14 scenes still need extracting in a future session.
- `data/raw/` itself is gitignored (large, regeneratable from `scenes.csv` +
  this log) — only the code and `scenes.csv` are committed to git. The
  extracted scenes currently exist only inside the Kaggle session's
  `/kaggle/working` (not yet saved as a persistent Kaggle Dataset/Version —
  do that before the session recycles, or re-extract later).

### Still open / next steps (Week 1, per `project_plan.md` §2)
- [ ] **Extract the remaining ~14 Asia scenes** (Spring, Fall, Winter) in a
      way that respects the ~19.5 GB quota — e.g. extract one season at a
      time and download/persist each batch (as a Kaggle Dataset version)
      before starting the next, rather than one 18-scene extract-local call.
- [ ] Pick 2 deliberately out-of-continent scenes for the domain-shift test
      (E2) — not yet selected.
- [ ] Preprocessing: clip/rescale SAR + optical bands, run `s2cloudless` for
      cloud masks, convert to NumPy shards.
- [ ] Splits: by-scene train/val/test, spatial-block splits, one deliberate
      random-split run to demonstrate leakage.
- [ ] PyTorch `Dataset`/`DataLoader` (random 128×128 crops + augmentation).
- [ ] Identity and tiny-CNN baselines.

### Working notes / gotchas for next time
- Kaggle's "Add Input" can only reliably process one dataset mount at a time
  in this environment — adding several in rapid succession caused
  `Error: Another mount/unmount operation in progress` and silently dropped
  some attachments. Add one, wait for it to fully register in the Input
  panel's dataset list, then add the next.
- The progress toast Kaggle shows while attaching a dataset (e.g.
  `sen12ms-cr-fall: Downloaded X/Y files`) is unreliable as a completion
  signal — it relabels itself between datasets and can sit at 99%+ for a
  long time while still materializing. Don't trust it; just re-run the
  actual check (`scan-local`) and look at whether the scene count changed.
- **`/kaggle/working` has a hard ~19.5 GB quota for this session** (shown as
  the denominator next to "Output" in the UI, e.g. `15.6GiB / 19.5GiB`).
  `extract-local`'s `--max-scenes` should be sized against that, not just
  against the patch counts in `scenes.csv` — a rough rule of thumb from this
  run: ~800 patches × 3 modalities × ~0.7 MB/patch average ≈ 1.7 GB *per
  scene*, so the whole quota is roughly **10–11 scenes**, not 18. Extract in
  smaller batches and persist each one (download, or save as a Kaggle
  Dataset) before starting the next batch.

---

## 2026-10-21 — Pipeline audit after the Kaggle crash; full rewrite of the training path

### The crash
`RuntimeError: weight of size [64, 0, 3, 3], expected input[16, 6, 256, 256] to have 0 channels`.
The notebook called `ModelA_UNet(False)`; the first positional parameter is `in_channels`, so the
U-Net was built with **0** input channels. Models now must be built with keyword arguments and
reject non-positive / boolean channel counts (`tests/test_pipeline.py::test_positional_bool_bug_is_caught`).

### Silent problems found while auditing (each would have invalidated results without crashing)
1. **SAR input was erased.** SEN12MS-CR S1 is already in dB (mostly negative). The code applied
   `10*log10(clip(x, 1e-5))` again, so every negative value became -50 dB → clipped → the constant
   −1. The models were effectively trained without radar. Fixed per plan §2.2 (clip VV [−25, 0],
   VH [−32.5, 0], rescale to [0, 1]); the Lee filter now runs in the linear domain and is off by
   default (ablation only, as the plan says).
2. **Wrong bands.** `read()[:3]` / `[:4]` took B1 (coastal aerosol), B2, B3 — not RGB. Now all 13
   bands in and out (15 → 13 channels, plan §3); RGB for display is B4/B3/B2.
3. **Triplets were mis-paired.** The dictionary keys never matched (`parts[-3]` is `s1`/`s2`/`cloudy`),
   so the "global list-sorting fallback" zipped three independently sorted lists of 122k files —
   any difference between mirrors shifts every pair. Pairing is now by the exact
   `(season, scene, patch)` parsed with the same regex as the scene finder.
4. **Spatial leakage.** Patches were shuffled and split randomly, so near-duplicate neighbouring
   patches sat in train and test; the "OOD" set was just another random 5%. Splits are now by scene
   (Asia: ~60/20/20 of scenes, one Indian scene forced into test), and the out-of-continent set is one
   scene each from Europe, Africa, North and South America.
5. **"FID" was not FID.** It was a Fréchet distance on 8×8-pooled pixels over 16-image batches
   (rank-deficient covariance). Replaced by real Inception FID (torchmetrics) over the whole test set;
   if it can't be loaded the column is NaN rather than a misleading number.
6. **Protocol differed from the plan.** Every model was trained as a GAN; Model C had no positional
   embedding (attention ignores token order); Model B was a global transformer, not Swin. Now: A/B/C
   with L1, the GAN as a separate A ablation, a real shifted-window Swin bottleneck with relative
   position bias, a conv-free ViT (patch 8, dim 192, learned positions), AdamW + warmup/cosine, one
   GradScaler per network, best checkpoint by validation L1.
7. **`find_region_scenes.py` had been replaced** by a stub that assigned countries from a hash of the
   scene id (`CN/JP/IN/FR` by `sum(ord(c)) % 4`). Restored the real reverse-geocoding version and
   `continents.py` from `1b5d4ea`. `data/scans/scenes.csv` was always from the real scan.

### New structure
- `src/data/dataset.py` — index, scene split, preprocessing, one-off float16 memmap cache, crop dataset
- `src/training/trainer.py` — one training loop for every run (resumable)
- `src/evaluation/{metrics,evaluate,plots}.py` — GPU metrics (checked against scikit-image), tiled
  inference, FID, uncertainty calibration (coverage, sparsification/AUSE), figures
- `notebooks/kaggle_pipeline.py` — the Kaggle script; `app.py` — Streamlit frontend
- `tests/` — unit tests and a synthetic SEN12MS-CR generator; the full pipeline was run end-to-end on
  synthetic data on CPU, and a 16-sample overfit test drives U-Net L1 from 0.22 to 0.002.

### Still open
- Real Kaggle run (≈2 h on a T4 for all five runs at 15k steps).
- Cloud bins use a reflectance-difference proxy, not s2cloudless.
- Not yet implemented from the plan: LR range test, second seeds, E1 data fractions, E3c/E5c
  visualisations, MC dropout / ensemble (E7b/c). The run configs and model flags
  (`use_skips`, `shift`, `window_size`, `use_pos_emb`, `patch_size`) are in place for them.

---

## 2026-10-21 (later) — Complete experiment suite in one Kaggle session

`notebooks/kaggle_pipeline.py` now runs **every experiment in the plan** in one 12-hour session:
core A/B/C, tiny-CNN + identity baselines, E1 (25/50/100%), the leakage check (random patch
split), E2 (out-of-continent), E3a/b/c (no skips, L2, effective receptive field), E4 (λ=100, 10),
E5a/b (no shift, window 8/16), E6a/b (no positional embedding, patch 16), E7a/b/c (variance head,
MC dropout, 3-seed ensemble) and second seeds for A/B/C.

How it fits in 12 h:
- **One step budget for every run.** A short GPU benchmark measures each architecture's speed and
  the planner picks the largest budget (15k, 12k, 10k, ... steps) at which all 26 jobs fit, so
  every comparison is still at an identical budget.
- **Both T4s in parallel.** Each job runs in its own process on one GPU.
- **Priority order + deadline guard.** Jobs run in the plan's priority order ("never cut" first,
  the plan's cut order last). A job is never started if it can't finish before the deadline, and a
  running job stops early, keeping its best checkpoint, if it overruns. The session therefore ends
  before Kaggle's limit and the output is always saved.
- **Crash-safe.** Each job writes its results to `jobs/<run>/` as soon as it finishes; a failed job
  is logged and skipped, not fatal.

New outputs: `metrics_main.csv` (mean ± std over seeds), `experiments.csv` (every ablation vs its
reference, with the plan's prediction), `learning_curves.csv`, `leakage_check.csv`,
`erf_stats.csv`, `uncertainty_summary.csv` for all three uncertainty methods. The frontend gained
an **Experiments** tab.

Not run: E5c (attention maps) — first in the plan's cut order. Cloud bins still use the
reflectance-difference proxy rather than s2cloudless.

---

## 2026-10-21 (evening) — Full Kaggle run started; new frontend; expert system

**Kaggle.** Quick tests found two problems, both fixed before the full run:
1. Reloaded checkpoints stayed on the CPU, so evaluation crashed (device mismatch). Training was unaffected.
2. U-Net and Swin models spent most of a short run moving their output from 0.5 to the real reflectance
   level (~0.1), staying below the cloudy-input baseline while the ViT and tiny CNN did not. Every model's
   output bias is now initialised to the mean band reflectance (Karpathy's "init well"). Effect in the 300-step
   check: U-Net 18.0 → 26.8 dB on the Asian test set.
The full run then started: the GPU benchmark (U-Net 12.6 it/s, Swin 11.0, ViT 34.7 on a T4) allowed the full
**15,000 steps for all 26 runs**, predicted at 5.6 h on two T4s.

**Frontend.** Rebuilt as a multipage app (`frontend/`): Home with a drag-to-compare slider, Model comparison,
Experiments with automatic verdicts against the plan's predictions (noise floor = max(0.3 dB, 2 × seed std)),
Explore patches, Advisor, Knowledge base, How it works.

**Expert system (AI-course project).** `src/expert/`: YAML knowledge base (35 rules, 5 groups, thresholds,
conflict resolution), MYCIN-style forward-chaining engine with explanation trace, fact extraction from the
patch / model outputs / experiment results, XAI (occlusion radar-reliance, input×gradient saliency). Design
and evaluation ideas in `docs/expert_system.md`; 8 scenario and consistency tests in `tests/test_expert.py`.
