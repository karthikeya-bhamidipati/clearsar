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
