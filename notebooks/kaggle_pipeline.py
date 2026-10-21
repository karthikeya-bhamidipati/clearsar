"""
ClearSAR - end-to-end Kaggle pipeline (index -> split -> cache -> train -> evaluate -> export)
==============================================================================================

HOW TO RUN ON KAGGLE
  1. Notebook settings: Accelerator = GPU T4 (x1 or x2), Internet = ON.
     (Avoid P100: recent PyTorch builds no longer ship kernels for its sm_60 GPU.)
  2. Add the four SEN12MS-CR inputs (see README.md): spring, summer, fall, winter.
  3. Either paste this whole file into one cell, or use a two-line cell:
         !git clone -q https://github.com/karthikeya-bhamidipati/clearsar.git /kaggle/working/clearsar
         !python /kaggle/working/clearsar/notebooks/kaggle_pipeline.py
  4. "Save Version -> Save & Run All". When it finishes, download
     /kaggle/working/clearsar_output.zip and extract it into  <project>/outputs/
     then run the frontend:  streamlit run app.py

WHAT IT PRODUCES  (/kaggle/working/clearsar_output/)
  results/metrics_summary.csv          one row per model x test set (PSNR, SSIM, SAM, MAE, RMSE, FID)
  results/metrics_by_cloud_bin.csv     the same, split into 0-20% ... 80-100% cloud cover
  results/metrics_by_country.csv       per-country breakdown (incl. the India slice)
  results/per_patch_metrics.csv        every test patch, every model
  results/uncertainty_summary.csv      E7 calibration numbers for the variance-head model
  results/*.png                        comparison grid, training curves, bar charts
  results/samples.npz                  a few test patches + predictions for the frontend
  results/scene_splits.csv             which scene went to which split (map in the frontend)
  checkpoints/best_<run>.pt            best-validation weights for every run
  logs/history_<run>.csv               validation curve of every run
  logs/pipeline_log.txt                full log of this run
  config.json                          every setting used

Every setting below can be overridden with an environment variable (CLEARSAR_*), which is
how the same script is smoke-tested locally on synthetic data (tests/make_synthetic_data.py).
"""
import gc
import json
import os
import shutil
import subprocess
import sys
import time

REPO_URL = "https://github.com/karthikeya-bhamidipati/clearsar.git"
ON_KAGGLE = os.path.isdir("/kaggle/input")


def _env(name, default, cast=str):
    v = os.environ.get(f"CLEARSAR_{name}")
    return default if v in (None, "") else cast(v)


def _caps(s):
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))}


QUICK = _env("QUICK", 0, int) == 1          # CLEARSAR_QUICK=1 -> short smoke run on Kaggle
CONFIG = dict(
    input_roots=_env("INPUT_ROOTS", "/kaggle/input/*").split(";"),
    out_dir=_env("OUT_DIR", "/kaggle/working/clearsar_output"),
    resume_dir=_env("RESUME_DIR", "/kaggle/working/clearsar_resume"),
    cache_dir=_env("CACHE_DIR", "/tmp/clearsar_cache"),
    # Patches per split, sampled evenly across that split's scenes (~3.7 MB each in the cache).
    caps=_caps(_env("CAPS", "train=600,val=100,test_id=100,ood=100" if QUICK
                    else "train=3000,val=400,test_id=600,ood=600")),
    train_region="Asia",
    ood_continents=["Europe", "Africa", "North America", "South America"],
    force_test_countries=["IN"],          # guarantees an India-only slice in the test set
    apply_lee=_env("APPLY_LEE", 0, int) == 1,   # speckle filter: ablation only (plan §2.2)
    crop=128,
    batch_size=_env("BATCH", 16, int),
    eval_batch=8,
    steps=_env("STEPS", 2000 if QUICK else 15000, int),
    val_every=_env("VAL_EVERY", 500 if QUICK else 1000, int),
    warmup=_env("WARMUP", 500, int),
    workers=_env("WORKERS", min(4, os.cpu_count() or 2), int),
    runs=_env("RUNS", "A_unet,B_swin,C_vit,A_gan,A_unc").split(","),
    n_samples_per_split=4,
    seed=42,
)

# The model comparison (plan §3-5). Same data, same loop; only these settings change.
RUN_SPECS = {
    "A_unet": dict(arch="unet", label="Model A - U-Net", lr=3e-4),
    "B_swin": dict(arch="swin_unet", label="Model B - Swin U-Net", lr=3e-4),
    "C_vit": dict(arch="vit", label="Model C - Pure ViT", lr=3e-4),
    "A_gan": dict(arch="unet", label="Model A + PatchGAN", lr=2e-4, use_gan=True, gan_lambda=100.0),
    "A_unc": dict(arch="unet", label="Model A + uncertainty", lr=3e-4, uncertainty=True),
}


# --------------------------------------------------------------------------------------
def setup():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
    repo = None
    if script_dir and os.path.isdir(os.path.join(script_dir, "..", "src")):
        repo = os.path.abspath(os.path.join(script_dir, ".."))       # running from a checkout
    if ON_KAGGLE:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio", "torchmetrics",
                        "torch-fidelity"], check=False)
        if repo is None:
            repo = "/kaggle/working/clearsar"
            shutil.rmtree(repo, ignore_errors=True)
            subprocess.run(["git", "clone", "-q", REPO_URL, repo], check=True)
    if repo is None:
        repo = os.getcwd()
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        print(line, flush=True)
        self.f.write(time.strftime("%H:%M:%S ") + line + "\n")
        self.f.flush()


def fit_caps_to_disk(caps, cache_dir, log, bytes_per_patch=28 * 256 * 256 * 2):
    os.makedirs(cache_dir, exist_ok=True)
    free = shutil.disk_usage(cache_dir).free
    need = sum(caps.values()) * bytes_per_patch
    log(f"[disk] cache needs ~{need / 1e9:.1f} GB, {free / 1e9:.1f} GB free at {cache_dir}")
    budget = free - 3e9
    if need > budget > 0:
        scale = budget / need
        caps = {k: max(8, int(v * scale)) for k, v in caps.items()}
        log(f"[disk] WARNING not enough space - scaling every split by {scale:.2f}: {caps}")
    return caps


def pick_samples(df, n, rng_seed=0):
    """n valid patches spread across the cloud-cover range (for the gallery)."""
    import numpy as np
    df = df[df.valid].sort_values("cloud_frac")
    if df.empty:
        return []
    qs = np.linspace(0.15, 0.95, n)
    picks = []
    for q in qs:
        i = int(round(q * (len(df) - 1)))
        ci = int(df.iloc[i].cache_idx)
        if ci not in picks:
            picks.append(ci)
    return picks


def main():
    repo = setup()
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import (IN_CHANNELS, OUT_CHANNELS, RGB_IDX, CachedPatchDataset,
                                  attach_scene_metadata, build_cache, index_triplets, make_splits)
    from src.evaluation import plots
    from src.evaluation.evaluate import evaluate
    from src.evaluation.metrics import uncertainty_summary
    from src.models.architectures import load_model_from_checkpoint
    from src.training.trainer import RunConfig, train_run

    cfg = CONFIG
    out = cfg["out_dir"]
    for sub in ("results", "checkpoints", "logs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    log = Logger(os.path.join(out, "logs", "pipeline_log.txt"))
    t_start = time.time()
    log("=" * 78)
    log(f"ClearSAR pipeline | repo {repo} | quick={QUICK}")
    log(json.dumps({k: v for k, v in cfg.items()}, default=str))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        log(f"[gpu] {name} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}")
        if cap < (7, 0):
            log("[gpu] WARNING: this GPU is older than sm_70 - if you see 'no kernel image is "
                "available', switch the Kaggle accelerator to T4.")
        torch.backends.cudnn.benchmark = True
    else:
        log("[gpu] WARNING: no GPU - training will be very slow")

    # ---- 1. index + split -------------------------------------------------------------
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    index_path = os.path.join(cfg["cache_dir"], "index.csv")
    if os.path.exists(index_path):
        triplets = pd.read_csv(index_path)
        log(f"[index] re-using {len(triplets)} indexed triplets from {index_path}")
    else:
        triplets = index_triplets(cfg["input_roots"], log=log)
        triplets.to_csv(index_path, index=False)
    triplets = attach_scene_metadata(triplets, os.path.join(repo, "data", "scans", "scenes.csv"), log=log)
    caps = fit_caps_to_disk(dict(cfg["caps"]), cfg["cache_dir"], log)
    patches, scenes = make_splits(triplets, cfg["train_region"], ood_continents=cfg["ood_continents"],
                                  force_test_countries=cfg["force_test_countries"], caps=caps,
                                  seed=cfg["seed"], log=log)
    scenes.to_csv(os.path.join(out, "results", "scene_splits.csv"), index=False)

    # ---- 2. cache ---------------------------------------------------------------------
    caches, split_df = {}, {}
    for s in ("train", "val", "test_id", "ood"):
        t0 = time.time()
        caches[s], split_df[s] = build_cache(patches[patches.split == s], cfg["cache_dir"], s,
                                             apply_lee=cfg["apply_lee"], num_workers=cfg["workers"], log=log)
        log(f"[cache] {s}: {int(split_df[s].valid.sum())} valid patches in {time.time() - t0:.0f}s")
    pd.concat(split_df.values()).drop(columns=["s1", "s2", "s2_cloudy"]).to_csv(
        os.path.join(out, "results", "patch_splits.csv"), index=False)

    def loader(split, train=False):
        d = split_df[split]
        ds = CachedPatchDataset(caches[split], d.loc[d.valid, "cache_idx"].values,
                                crop=cfg["crop"] if train else None, augment=train)
        if train and len(ds) < cfg["batch_size"]:
            raise RuntimeError(f"Only {len(ds)} training patches - need at least one batch")
        return DataLoader(ds, batch_size=cfg["batch_size"] if train else cfg["eval_batch"], shuffle=train,
                          drop_last=train, num_workers=cfg["workers"], pin_memory=device.type == "cuda",
                          persistent_workers=cfg["workers"] > 0)

    train_loader, val_loader = loader("train", True), loader("val")
    test_loaders = {"test_id": loader("test_id"), "ood": loader("ood")}
    sample_ids = {s: pick_samples(split_df[s], cfg["n_samples_per_split"]) for s in test_loaders}

    # ---- 3. evaluate helpers ----------------------------------------------------------
    per_patch, fids, samples_pred, samples_sigma = [], [], {}, {}
    unc_rows, unc_curves, fid_warned = [], {}, []

    def record(run, label, split, res, extras):
        d = pd.DataFrame(res)
        meta = split_df[split][["cache_idx", "season", "scene", "patch", "country", "continent", "cloud_frac"]]
        d = d.merge(meta, on="cache_idx", how="left")
        d.insert(0, "split", split)
        d.insert(0, "label", label)
        d.insert(0, "run", run)
        per_patch.append(d)
        fids.append(dict(run=run, split=split, fid=extras.get("fid", float("nan"))))
        if extras.get("fid_error") and not fid_warned:
            fid_warned.append(True)
            log(f"[fid] unavailable ({extras['fid_error']}) - FID column will be empty")
        for ci, (pred, sigma) in extras.get("samples", {}).items():
            samples_pred[(run, split, ci)] = pred
            if sigma is not None:
                samples_sigma[(split, ci)] = sigma
        log(f"[eval] {label:26s} {split:7s} PSNR {d.psnr.mean():6.2f} | SSIM {d.ssim.mean():.4f} | "
            f"SAM {d.sam.mean():5.2f} | MAE {d.mae.mean():.4f} | FID {extras.get('fid', float('nan')):.1f}")

    # Baseline: output = cloudy input (plan §2.4). A model that can't beat this is broken.
    for split, ld in test_loaders.items():
        res, ex = evaluate(None, ld, device, OUT_CHANNELS, identity=True, fid=True, collect=set(sample_ids[split]))
        record("identity", "Identity (cloudy input)", split, res, ex)

    # ---- 4. train + evaluate every run ------------------------------------------------
    run_infos, labels = [], {"identity": "Identity (cloudy input)"}
    for name in cfg["runs"]:
        spec = dict(RUN_SPECS[name])
        rc = RunConfig(name=name, steps=cfg["steps"], val_every=cfg["val_every"],
                       warmup=min(cfg["warmup"], max(1, cfg["steps"] // 10)), seed=cfg["seed"], **spec)
        labels[name] = rc.label
        log("=" * 78)
        info = train_run(rc, train_loader, val_loader, device, os.path.join(out, "checkpoints"),
                         os.path.join(out, "logs"), IN_CHANNELS, OUT_CHANNELS, resume_dir=cfg["resume_dir"],
                         log=log)
        run_infos.append(info)
        model, _ = load_model_from_checkpoint(os.path.join(out, "checkpoints", f"best_{name}.pt"), device)
        for split, ld in test_loaders.items():
            res, ex = evaluate(model, ld, device, OUT_CHANNELS, uncertainty=rc.uncertainty, fid=True,
                               collect=set(sample_ids[split]))
            record(name, rc.label, split, res, ex)
            if rc.uncertainty and "unc_errors" in ex:
                summ, curve = uncertainty_summary(ex["unc_errors"], ex["unc_sigmas"])
                unc_rows.append(dict(run=name, split=split, **summ))
                unc_curves[split] = curve
                log(f"[unc] {split}: coverage@95 {summ['coverage95']:.3f} | spearman(err, b) "
                    f"{summ['spearman_err_sigma']:.3f} | AUSE {summ['ause']:.4f}")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- 5. tables --------------------------------------------------------------------
    res_dir = os.path.join(out, "results")
    pp = pd.concat(per_patch, ignore_index=True)
    pp.to_csv(os.path.join(res_dir, "per_patch_metrics.csv"), index=False)
    metrics = ["psnr", "ssim", "sam", "mae", "rmse"]
    info_df = pd.DataFrame(run_infos)
    info_df.to_csv(os.path.join(res_dir, "run_info.csv"), index=False)

    summary = pp.groupby(["run", "label", "split"], sort=False)[metrics].mean().reset_index()
    summary["n_patches"] = pp.groupby(["run", "label", "split"], sort=False).size().values
    summary = summary.merge(pd.DataFrame(fids), on=["run", "split"], how="left")
    if len(info_df):
        summary = summary.merge(info_df[["name", "params", "train_minutes", "best_step"]].rename(
            columns={"name": "run"}), on="run", how="left")
    summary.to_csv(os.path.join(res_dir, "metrics_summary.csv"), index=False)

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    pp["cloud_bin"] = pd.cut(pp.cloud_frac, bins, labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
                             include_lowest=True)
    by_bin = pp.groupby(["run", "label", "split", "cloud_bin"], sort=False, observed=True)[metrics].mean()
    by_bin["n_patches"] = pp.groupby(["run", "label", "split", "cloud_bin"], sort=False, observed=True).size()
    by_bin = by_bin.reset_index().sort_values(["split", "run", "cloud_bin"])
    by_bin.to_csv(os.path.join(res_dir, "metrics_by_cloud_bin.csv"), index=False)
    by_country = pp.groupby(["run", "label", "split", "country"], sort=False)[metrics].mean()
    by_country["n_patches"] = pp.groupby(["run", "label", "split", "country"], sort=False).size()
    by_country.reset_index().to_csv(os.path.join(res_dir, "metrics_by_country.csv"), index=False)
    if unc_rows:
        pd.DataFrame(unc_rows).to_csv(os.path.join(res_dir, "uncertainty_summary.csv"), index=False)
        pd.concat([pd.DataFrame(dict(split=s, fraction=c["fractions"], model_mae=c["model"],
                                     oracle_mae=c["oracle"])) for s, c in unc_curves.items()]).to_csv(
            os.path.join(res_dir, "sparsification.csv"), index=False)

    # ---- 6. figures + frontend samples --------------------------------------------------
    histories = {}
    for name in cfg["runs"]:
        hp = os.path.join(out, "logs", f"history_{name}.csv")
        histories[name] = pd.read_csv(hp) if os.path.exists(hp) else None
    plots.training_curves(histories, labels, os.path.join(res_dir, "training_curves.png"))
    plots.metric_bars(summary, os.path.join(res_dir, "metrics_comparison.png"))
    plots.cloud_bin_plot(by_bin, os.path.join(res_dir, "cloud_bin_psnr.png"))
    if unc_curves:
        plots.sparsification_plot(unc_curves, os.path.join(res_dir, "sparsification.png"))

    X = {s: np.load(caches[s]["x"], mmap_mode="r") for s in test_loaders}
    Y = {s: np.load(caches[s]["y"], mmap_mode="r") for s in test_loaders}
    run_order = ["identity"] + cfg["runs"]
    grid, npz, meta = [], {"x": [], "y": [], "sigma": []}, []
    for r in run_order:
        npz[f"pred_{r}"] = []
    for split in test_loaders:
        sd = split_df[split].set_index("cache_idx")
        for ci in sample_ids[split]:
            row = sd.loc[ci]
            scores = {}
            for r in run_order:
                m = pp[(pp.run == r) & (pp.split == split) & (pp.cache_idx == ci)]
                if len(m):
                    scores[r] = {k: float(m.iloc[0][k]) for k in metrics}
            x, y = np.asarray(X[split][ci], np.float32), np.asarray(Y[split][ci], np.float32)
            preds = {r: samples_pred[(r, split, ci)] for r in run_order if (r, split, ci) in samples_pred}
            sigma = samples_sigma.get((split, ci))
            grid.append(dict(x=x, y=y, preds=preds, sigma=sigma,
                             scores={r: f"{v['psnr']:.1f} dB" for r, v in scores.items()},
                             title=f"{plots.SPLIT_LABELS[split].split(' (')[0]}\n{row.country} "
                                   f"{int(row.cloud_frac * 100)}% cloud"))
            npz["x"].append(x.astype(np.float16))
            npz["y"].append(y.astype(np.float16))
            npz["sigma"].append((sigma if sigma is not None else np.zeros(y.shape[1:])).astype(np.float16))
            for r in run_order:
                p = preds.get(r)
                rgb = np.zeros(y.shape[1:] + (3,), np.uint8) if p is None else \
                    (np.clip(np.stack([p[i] for i in RGB_IDX], -1) * 3.0, 0, 1) * 255).astype(np.uint8)
                npz[f"pred_{r}"].append(rgb)
            meta.append(dict(split=split, season=row.season, scene=int(row.scene), patch=int(row.patch),
                             country=row.country, continent=row.continent, cloud_frac=float(row.cloud_frac),
                             scores=scores))
    if grid:
        plots.comparison_grid(grid, {r: labels[r] for r in run_order},
                              os.path.join(res_dir, "master_architecture_comparison.png"))
        np.savez_compressed(os.path.join(res_dir, "samples.npz"),
                            **{k: np.stack(v) for k, v in npz.items()},
                            meta=np.array(json.dumps(dict(samples=meta, runs=run_order, labels=labels))))

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(config=cfg, runs={n: RUN_SPECS[n] for n in cfg["runs"]}, labels=labels,
                       in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
                       preprocessing=dict(s1="dB, VV clip [-25,0], VH clip [-32.5,0], rescale [0,1]",
                                          s2="clip [0,10000] / 10000, 13 bands", lee=cfg["apply_lee"]),
                       cloud_mask="proxy: cloudy-minus-clear visible reflectance > 0.08",
                       minutes_total=(time.time() - t_start) / 60), f, indent=2, default=str)

    # ---- 7. report + zip --------------------------------------------------------------
    log("=" * 78)
    log("FINAL RESULTS (mean over test patches)")
    show = summary[["label", "split", "n_patches"] + metrics + ["fid"]].copy()
    log("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(f"Total time {(time.time() - t_start) / 60:.1f} min")
    log.f.close()
    zip_path = shutil.make_archive(out, "zip", out)
    print(f"\nDONE - download {zip_path} and extract it into <project>/outputs/", flush=True)


if __name__ == "__main__":
    main()
