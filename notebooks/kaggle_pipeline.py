"""
ClearSAR - the COMPLETE project in one Kaggle session (all experiments of docs/project_plan.md)
==============================================================================================

HOW TO RUN ON KAGGLE
  1. Notebook settings: Accelerator = "GPU T4 x2" (both GPUs are used in parallel),
     Internet = ON, Persistence = off. Add the four SEN12MS-CR datasets as inputs.
  2. One cell:
         !git clone -q https://github.com/karthikeya-bhamidipati/clearsar.git /kaggle/working/clearsar
         !python /kaggle/working/clearsar/notebooks/kaggle_pipeline.py
     Smoke test first (~30 min, same code path, tiny budget):
         !CLEARSAR_QUICK=1 python /kaggle/working/clearsar/notebooks/kaggle_pipeline.py
  3. "Save Version -> Save & Run All (Commit)". The run stops itself before Kaggle's 12 h
     limit (default budget CLEARSAR_HOURS=11). Then download clearsar_output.zip from the
     Output tab, extract into <project>/outputs/ and run:  streamlit run app.py

WHAT IT DOES
  index + split by scene -> cache patches once -> identity baseline -> GPU benchmark picks ONE
  step budget for every run so all experiments fit -> trains the jobs below in priority order,
  one per GPU in parallel, never starting a job that cannot finish before the deadline ->
  ensemble + receptive-field analysis -> tables, figures, frontend samples, zip.

  core      A U-Net, B Swin U-Net, C pure ViT (L1)          baseline  tiny CNN, identity
  E1        A/B/C on 25% and 50% of the training data       leakage   A on a random patch split
  E2        every model on the out-of-continent test set    E3        A without skips, A with L2, ERF
  E4        A + PatchGAN, lambda 100 and 10                  E5        B without shift, window 8, 16
  E6        C without positional embedding, patch 16         E7        variance head, MC dropout, ensemble
  seeds     second seed for A/B/C, third seed for A (ensemble)

Every setting can be overridden with CLEARSAR_* environment variables (see CONFIG); the
whole script is smoke-tested locally on synthetic data (tests/make_synthetic_data.py).
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
IS_JOB = "--job" in sys.argv


def _env(name, default, cast=str):
    v = os.environ.get(f"CLEARSAR_{name}")
    return default if v in (None, "") else cast(v)


def _caps(s):
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))}


QUICK = _env("QUICK", 0, int) == 1
CONFIG = dict(
    input_roots=_env("INPUT_ROOTS", "/kaggle/input/*").split(";"),
    out_dir=_env("OUT_DIR", "/kaggle/working/clearsar_output"),
    resume_dir=_env("RESUME_DIR", "/kaggle/working/clearsar_resume"),
    cache_dir=_env("CACHE_DIR", "/tmp/clearsar_cache"),
    caps=_caps(_env("CAPS", "train=400,val=64,test_id=96,ood=96" if QUICK
                    else "train=3000,val=400,test_id=600,ood=600")),
    train_region="Asia",
    ood_continents=["Europe", "Africa", "North America", "South America"],
    force_test_countries=["IN"],
    apply_lee=_env("APPLY_LEE", 0, int) == 1,
    crop=128,
    batch_size=_env("BATCH", 16, int),
    eval_batch=8,
    hours=_env("HOURS", 1.5 if QUICK else 11.0, float),     # total wall-clock budget
    steps=_env("STEPS", 300 if QUICK else 0, int),         # 0 = choose automatically
    step_options=[15000, 12000, 10000, 8000, 6000, 5000, 4000],
    warmup=500,
    final_margin_min=_env("FINAL_MARGIN", 25, float),      # reserved for ensemble/analysis/export
    slots=_env("SLOTS", 0, int),                           # parallel jobs (0 = number of GPUs)
    workers=_env("WORKERS", min(4, os.cpu_count() or 2), int),
    only=[r for r in _env("RUNS", "").split(",") if r],    # restrict to these runs (debugging)
    n_samples_per_split=4,
    seed=42,
)


# --------------------------------------------------------------------------------------
# The experiment list, in PRIORITY order (plan §7: "never cut" first, plan's cut-order last)
# --------------------------------------------------------------------------------------
def R(name, arch, label, experiment, ref=None, question="", group=None, seed=42, fraction=1.0,
      split_mode="scene", lr=3e-4, use_gan=False, gan_lambda=100.0, uncertainty=False, loss="l1",
      model_kwargs=None, mc_passes=0, family=None):
    return dict(name=name, arch=arch, label=label, experiment=experiment, ref=ref, question=question,
                group=group or name, seed=seed, fraction=fraction, split_mode=split_mode, lr=lr,
                use_gan=use_gan, gan_lambda=gan_lambda, uncertainty=uncertainty, loss=loss,
                model_kwargs=model_kwargs or {}, mc_passes=mc_passes,
                family=family or {"unet": "A", "swin_unet": "B", "vit": "C", "tiny_cnn": "tiny"}[arch])


RUNS = [
    R("A_unet", "unet", "Model A - U-Net", "core"),
    R("B_swin", "swin_unet", "Model B - Swin U-Net", "core"),
    R("C_vit", "vit", "Model C - Pure ViT", "core"),
    R("tiny_cnn", "tiny_cnn", "Baseline - tiny CNN", "baseline", ref="A_unet",
      question="Does the U-Net clearly beat a 3-layer CNN? If not, something is wrong."),
    R("A_f25", "unet", "Model A - 25% data", "E1", fraction=0.25, ref="A_unet",
      question="A should be the best model at 25% (strong inductive bias)."),
    R("B_f25", "swin_unet", "Model B - 25% data", "E1", fraction=0.25, ref="B_swin"),
    R("C_f25", "vit", "Model C - 25% data", "E1", fraction=0.25, ref="C_vit",
      question="C should suffer most from little data but improve fastest as data grows."),
    R("A_f50", "unet", "Model A - 50% data", "E1", fraction=0.5, ref="A_unet"),
    R("B_f50", "swin_unet", "Model B - 50% data", "E1", fraction=0.5, ref="B_swin"),
    R("C_f50", "vit", "Model C - 50% data", "E1", fraction=0.5, ref="C_vit"),
    R("A_random", "unet", "Model A - random patch split", "leakage", split_mode="random", ref="A_unet",
      question="A random patch split leaks neighbouring patches into the test set and inflates the score."),
    R("A_noskip", "unet", "Model A - no skip connections", "E3a", model_kwargs={"use_skips": False},
      ref="A_unet", question="Without skips the output should be blurrier and lose fine detail."),
    R("A_gan", "unet", "Model A + PatchGAN (lambda 100)", "E4", lr=2e-4, use_gan=True, ref="A_unet",
      question="Sharper images, LOWER PSNR, better FID (perception-distortion trade-off)."),
    R("A_unc", "unet", "Model A + variance head", "E7a", uncertainty=True, ref="A_unet",
      question="Uncertainty highest under thick cloud; may NOT rise on out-of-region scenes."),
    R("C_nopos", "vit", "Model C - no positional embedding", "E6a", model_kwargs={"use_pos_emb": False},
      ref="C_vit", question="Without positions attention ignores token order: spatial layout is lost."),
    R("B_noshift", "swin_unet", "Model B - no window shift", "E5a", model_kwargs={"shift": False},
      ref="B_swin", question="Without shifting, information cannot cross window borders."),
    R("A_l2", "unet", "Model A - L2 loss", "E3b", loss="l2", ref="A_unet",
      question="L2 averages over plausible outputs: blurrier than L1."),
    R("A_gan10", "unet", "Model A + PatchGAN (lambda 10)", "E4", lr=2e-4, use_gan=True, gan_lambda=10.0,
      ref="A_unet", question="Stronger adversarial weight: even sharper, even lower PSNR."),
    R("A_mcd", "unet", "Model A + MC dropout", "E7b", model_kwargs={"dropout": 0.1}, mc_passes=20,
      ref="A_unet", question="Epistemic uncertainty should RISE on the out-of-region scenes."),
    R("A_unet_s43", "unet", "Model A - U-Net", "seeds", group="A_unet", seed=43),
    R("A_unet_s44", "unet", "Model A - U-Net", "seeds", group="A_unet", seed=44),
    R("B_swin_s43", "swin_unet", "Model B - Swin U-Net", "seeds", group="B_swin", seed=43),
    R("C_vit_s43", "vit", "Model C - Pure ViT", "seeds", group="C_vit", seed=43),
    R("C_p16", "vit", "Model C - patch 16", "E6b", model_kwargs={"patch_size": 16}, ref="C_vit",
      question="Bigger patches: visible blocky artefacts."),
    R("B_win8", "swin_unet", "Model B - window 8", "E5b", model_kwargs={"window_size": 8}, ref="B_swin",
      question="Larger windows: does more context help with large clouds?"),
    R("B_win16", "swin_unet", "Model B - window 16 (global)", "E5b", model_kwargs={"window_size": 16},
      ref="B_swin", question="Window 16 = global attention at the bottleneck."),
]
SPECIAL = {
    "identity": R("identity", "unet", "Identity (cloudy input)", "baseline", family="identity"),
    "A_ens": R("A_ens", "unet", "Model A - 3-seed ensemble", "E7c", ref="A_unet",
               question="Ensemble disagreement should rise on out-of-region scenes; mean usually beats one model."),
}
RUN_INDEX = {r["name"]: r for r in RUNS} | SPECIAL
MAIN_GROUPS = ["identity", "tiny_cnn", "A_unet", "B_swin", "C_vit", "A_gan", "A_unc", "A_mcd", "A_ens"]
GALLERY_RUNS = MAIN_GROUPS
EVAL_SPLITS = ("test_id", "ood")
KEY_OFFSET = {"train": 0, "val": 1_000_000, "test_id": 2_000_000, "ood": 3_000_000}
DEFAULT_SEC_PER_STEP = {"unet": 0.10, "unet_gan": 0.16, "swin": 0.11, "vit": 0.05, "vit16": 0.03, "tiny": 0.02}


def bench_key(r):
    if r["arch"] == "unet":
        return "unet_gan" if r["use_gan"] else "unet"
    if r["arch"] == "vit":
        return "vit16" if r["model_kwargs"].get("patch_size") == 16 else "vit"
    return {"swin_unet": "swin", "tiny_cnn": "tiny"}[r["arch"]]


def eval_seconds(r):
    return 600 if r["mc_passes"] else 150


# --------------------------------------------------------------------------------------
def setup():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
    repo = None
    if script_dir and os.path.isdir(os.path.join(script_dir, "..", "src")):
        repo = os.path.abspath(os.path.join(script_dir, ".."))
    if ON_KAGGLE and not IS_JOB:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio", "torchmetrics",
                        "torch-fidelity"], check=False)
        if repo is None:
            repo = "/kaggle/working/clearsar"
            shutil.rmtree(repo, ignore_errors=True)
            subprocess.run(["git", "clone", "-q", REPO_URL, repo], check=True)
    if repo is None:
        repo = "/kaggle/working/clearsar" if ON_KAGGLE else os.getcwd()
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


def jobs_dir():
    return os.path.join(CONFIG["out_dir"], "jobs")


def job_done(name):
    return os.path.exists(os.path.join(jobs_dir(), name, "DONE"))


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


def pick_samples(df, n):
    import numpy as np
    df = df[df.valid].sort_values("cloud_frac")
    picks = []
    for q in np.linspace(0.15, 0.95, n) if len(df) else []:
        ci = int(df.iloc[int(round(q * (len(df) - 1)))].cache_idx)
        if ci not in picks:
            picks.append(ci)
    return picks


# --------------------------------------------------------------------------------------
# Shared: evaluate one model (or ensemble / identity) and save its results as a job folder
# --------------------------------------------------------------------------------------
def evaluate_and_save(name, model, eval_sets, device, state, log, info=None, identity=False):
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader

    from src.data.dataset import OUT_CHANNELS, RGB_IDX
    from src.evaluation.evaluate import evaluate
    from src.evaluation.metrics import uncertainty_summary

    spec = RUN_INDEX[name]
    jd = os.path.join(jobs_dir(), name)
    os.makedirs(jd, exist_ok=True)
    result = dict(name=name, info=info or {}, fid={}, unc={}, curves={}, unc_kind=None)
    rows, samples = [], {}
    gallery = name in GALLERY_RUNS
    for split, (ds, meta) in eval_sets.items():
        loader = DataLoader(ds, batch_size=state["eval_batch"], shuffle=False,
                            num_workers=state["job_workers"], pin_memory=device.type == "cuda")
        collect = set(state["sample_keys"].get(split, [])) if gallery else None
        res, ex = evaluate(model, loader, device, OUT_CHANNELS, uncertainty=spec["uncertainty"],
                           identity=identity, fid=True, collect=collect, mc_passes=spec["mc_passes"])
        d = pd.DataFrame(res).rename(columns={"cache_idx": "key"}).merge(meta, on="key", how="left")
        d.insert(0, "split", split)
        rows.append(d)
        result["fid"][split] = ex.get("fid", float("nan"))
        if ex.get("fid_error"):
            result["fid_error"] = ex["fid_error"]
        if "unc_errors" in ex:
            summ, curve = uncertainty_summary(ex["unc_errors"], ex["unc_sigmas"], ex["unc_kind"])
            result["unc"][split], result["unc_kind"] = summ, ex["unc_kind"]
            result["curves"][split] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in curve.items()}
        for key, (pred, sigma) in ex.get("samples", {}).items():
            rgb = (np.clip(np.stack([pred[i] for i in RGB_IDX], -1) * 3.0, 0, 1) * 255).astype(np.uint8)
            samples[f"rgb_{split}_{key}"] = rgb
            if sigma is not None:
                samples[f"sigma_{split}_{key}"] = sigma.astype(np.float16)
        log(f"[eval] {spec['label']:34s} {split:11s} PSNR {d.psnr.mean():6.2f} | SSIM {d.ssim.mean():.4f} | "
            f"SAM {d.sam.mean():5.2f} | MAE {d.mae.mean():.4f} | FID {result['fid'][split]:.1f}")
        for split_u, s in result["unc"].items():
            if split_u == split:
                log(f"[unc]  {spec['label']:34s} {split:11s} coverage95 {s['coverage95']:.3f} | "
                    f"mean sigma {s['mean_sigma']:.4f} | spearman {s['spearman_err_sigma']:.3f} | AUSE {s['ause']:.4f}")
    pd.concat(rows).to_csv(os.path.join(jd, "per_patch.csv"), index=False)
    if samples:
        np.savez_compressed(os.path.join(jd, "samples.npz"), **samples)
    with open(os.path.join(jd, "result.json"), "w") as f:
        json.dump(result, f, default=float)
    open(os.path.join(jd, "DONE"), "w").close()


def load_split_frames(state):
    import pandas as pd
    out = {}
    for s, p in state["split_csv"].items():
        d = pd.read_csv(p)
        d["key"] = d.cache_idx + KEY_OFFSET[s]
        out[s] = d
    return out


META_COLS = ["key", "cache_idx", "season", "scene", "patch", "country", "continent", "cloud_frac"]


def scene_eval_sets(state, frames):
    from src.data.dataset import CachedPatchDataset
    sets = {}
    for s in EVAL_SPLITS:
        d = frames[s]
        idx = d.loc[d.valid, "cache_idx"].values
        sets[s] = (CachedPatchDataset(state["caches"][s], idx, key_offset=KEY_OFFSET[s]), d[META_COLS])
    return sets


# --------------------------------------------------------------------------------------
# One training job (runs in its own process, one per GPU)
# --------------------------------------------------------------------------------------
def run_job(name):
    repo = setup()
    import pandas as pd
    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    from src.data.dataset import (IN_CHANNELS, OUT_CHANNELS, CachedPatchDataset, fraction_indices,
                                  random_patch_split)
    from src.models.architectures import load_model_from_checkpoint
    from src.training.trainer import RunConfig, train_run

    with open(os.path.join(jobs_dir(), "state.json")) as f:
        state = json.load(f)
    out = CONFIG["out_dir"]
    log = Logger(os.path.join(out, "logs", f"job_{name}.txt"))
    spec = RUN_INDEX[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    frames = load_split_frames(state)
    caches = state["caches"]
    crop, workers = state["crop"], state["job_workers"]

    def valid_idx(s):
        return frames[s].loc[frames[s].valid, "cache_idx"].values

    if spec["split_mode"] == "scene":
        tr_idx = fraction_indices(frames["train"], spec["fraction"], seed=CONFIG["seed"])
        train_ds = CachedPatchDataset(caches["train"], tr_idx, crop=crop, augment=True)
        val_ds = CachedPatchDataset(caches["val"], valid_idx("val"), key_offset=KEY_OFFSET["val"])
        eval_sets = scene_eval_sets(state, frames)
    else:  # leakage check: pool the Asia patches and split them at random, same split sizes
        pools = {s: valid_idx(s) for s in ("train", "val", "test_id")}
        sizes = {"train": len(pools["train"]), "val": len(pools["val"]), "random_test": len(pools["test_id"])}
        parts = random_patch_split(pools, sizes, seed=CONFIG["seed"])

        def concat(items, **kw):
            dss = []
            for s in ("train", "val", "test_id"):
                idx = [i for (c, i) in items if c == s]
                if idx:
                    dss.append(CachedPatchDataset(caches[s], idx, key_offset=KEY_OFFSET[s], **kw))
            return ConcatDataset(dss)

        train_ds = concat(parts["train"], crop=crop, augment=True)
        val_ds = concat(parts["val"])
        meta = pd.concat([frames[s][META_COLS] for s in ("train", "val", "test_id")])
        eval_sets = {"random_test": (concat(parts["random_test"]), meta)}
    log(f"[{name}] {spec['label']}: {len(train_ds)} training patches ({spec['split_mode']} split, "
        f"fraction {spec['fraction']})")

    train_loader = DataLoader(train_ds, batch_size=state["batch_size"], shuffle=True, drop_last=True,
                              num_workers=workers, pin_memory=device.type == "cuda",
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=state["eval_batch"], shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    rc = RunConfig(name=name, arch=spec["arch"], label=spec["label"], model_kwargs=spec["model_kwargs"],
                   uncertainty=spec["uncertainty"], use_gan=spec["use_gan"], gan_lambda=spec["gan_lambda"],
                   lr=spec["lr"], steps=state["steps"], warmup=state["warmup"], val_every=state["val_every"],
                   seed=spec["seed"], loss=spec["loss"],
                   deadline=state["deadline"] - state["final_margin_s"] - eval_seconds(spec))
    info = train_run(rc, train_loader, val_loader, device, os.path.join(out, "checkpoints"),
                     os.path.join(out, "logs"), IN_CHANNELS, OUT_CHANNELS, resume_dir=CONFIG["resume_dir"], log=log)
    del train_loader, val_loader
    model, _ = load_model_from_checkpoint(os.path.join(out, "checkpoints", f"best_{name}.pt"), device)
    evaluate_and_save(name, model, eval_sets, device, state, log, info=info)
    log(f"[{name}] job finished")


# --------------------------------------------------------------------------------------
# Planning: measure speed on this GPU, then choose one step budget for every run
# --------------------------------------------------------------------------------------
def benchmark(keys, batch, crop, log, n_warm=5, n_time=15):
    import torch
    import torch.nn as nn

    from src.data.dataset import IN_CHANNELS, OUT_CHANNELS
    from src.models.architectures import build_model
    from src.models.discriminator import PatchGANDiscriminator

    dev = torch.device("cuda")
    sps = {}
    for k in keys:
        arch = {"unet": "unet", "unet_gan": "unet", "swin": "swin_unet", "vit": "vit", "vit16": "vit",
                "tiny": "tiny_cnn"}[k]
        G = build_model(arch, IN_CHANNELS, OUT_CHANNELS, **({"patch_size": 16} if k == "vit16" else {})).to(dev)
        D = PatchGANDiscriminator(IN_CHANNELS + OUT_CHANNELS).to(dev) if k == "unet_gan" else None
        params = list(G.parameters()) + (list(D.parameters()) if D is not None else [])
        opt = torch.optim.AdamW(params, 1e-4)
        scaler = torch.amp.GradScaler("cuda")
        x = torch.rand(batch, IN_CHANNELS, crop, crop, device=dev)
        y = torch.rand(batch, OUT_CHANNELS, crop, crop, device=dev)
        for i in range(n_warm + n_time):
            if i == n_warm:
                torch.cuda.synchronize()
                t0 = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                p = G(x)
                loss = nn.functional.l1_loss(p.float(), y)
                if D is not None:  # ~ one D step + the adversarial term
                    loss = loss + D(x, p.detach()).mean() + D(x, y).mean() + D(x, p).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        sps[k] = (time.time() - t0) / n_time
        log(f"[bench] {k:9s} {sps[k] * 1000:6.1f} ms/step  ({1 / sps[k]:.1f} it/s)")
        del G, D, opt
        torch.cuda.empty_cache()
    return sps


def predict_seconds(r, steps, sps):
    return steps * sps[bench_key(r)] * 1.25 + eval_seconds(r) + 60     # data/val overhead + eval + startup


def makespan(durations, slots):
    loads = [0.0] * slots
    for d in durations:
        i = loads.index(min(loads))
        loads[i] += d
    return max(loads)


def choose_steps(runs, sps, slots, budget_s, options, log):
    for s in options:
        m = makespan([predict_seconds(r, s, sps) for r in runs], slots)
        log(f"[plan] {s:6d} steps/run -> predicted {m / 3600:5.2f} h for {len(runs)} runs on {slots} slot(s) "
            f"(budget {budget_s / 3600:.2f} h)")
        if m <= budget_s:
            return s
    log("[plan] even the smallest budget does not fit - lowest-priority runs will be skipped")
    return options[-1]


# --------------------------------------------------------------------------------------
def main():
    if IS_JOB:
        return run_job(sys.argv[sys.argv.index("--job") + 1])
    t_start = time.time()
    repo = setup()
    import numpy as np
    import pandas as pd
    import torch

    from src.data.dataset import attach_scene_metadata, build_cache, index_triplets, make_splits

    cfg = CONFIG
    out = cfg["out_dir"]
    for sub in ("results", "checkpoints", "logs", "jobs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    log = Logger(os.path.join(out, "logs", "pipeline_log.txt"))
    deadline = t_start + cfg["hours"] * 3600
    final_margin_s = cfg["final_margin_min"] * 60 if not QUICK else 8 * 60
    log("=" * 90)
    log(f"ClearSAR complete pipeline | repo {repo} | quick={QUICK} | budget {cfg['hours']} h")

    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpu else "cpu")
    for g in range(n_gpu):
        cap = torch.cuda.get_device_capability(g)
        log(f"[gpu] cuda:{g} {torch.cuda.get_device_name(g)} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}")
        if cap < (7, 0):
            log("[gpu] WARNING: GPU older than sm_70 - if you see 'no kernel image is available', use T4.")
    if not n_gpu:
        log("[gpu] WARNING: no GPU - training will be very slow")
    slots = cfg["slots"] or max(1, n_gpu)
    job_workers = max(1, cfg["workers"] // slots) if cfg["workers"] else 0

    # ---- 1. index + split + cache -------------------------------------------------------
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    index_path = os.path.join(cfg["cache_dir"], "index.csv")
    if os.path.exists(index_path):
        triplets = pd.read_csv(index_path)
        log(f"[index] re-using {len(triplets)} indexed triplets")
    else:
        triplets = index_triplets(cfg["input_roots"], log=log)
        triplets.to_csv(index_path, index=False)
    triplets = attach_scene_metadata(triplets, os.path.join(repo, "data", "scans", "scenes.csv"), log=log)
    caps = fit_caps_to_disk(dict(cfg["caps"]), cfg["cache_dir"], log)
    patches, scenes = make_splits(triplets, cfg["train_region"], ood_continents=cfg["ood_continents"],
                                  force_test_countries=cfg["force_test_countries"], caps=caps,
                                  seed=cfg["seed"], log=log)
    scenes.to_csv(os.path.join(out, "results", "scene_splits.csv"), index=False)
    caches, split_csv, frames = {}, {}, {}
    for s in ("train", "val", "test_id", "ood"):
        t0 = time.time()
        caches[s], d = build_cache(patches[patches.split == s], cfg["cache_dir"], s,
                                   apply_lee=cfg["apply_lee"], num_workers=cfg["workers"], log=log)
        d = d.drop(columns=["s1", "s2", "s2_cloudy"])
        split_csv[s] = os.path.join(cfg["cache_dir"], f"split_{s}.csv")
        d.to_csv(split_csv[s], index=False)
        frames[s] = d
        log(f"[cache] {s}: {int(d.valid.sum())} valid patches ({time.time() - t0:.0f}s)")
    pd.concat(frames.values()).to_csv(os.path.join(out, "results", "patch_splits.csv"), index=False)
    if int(frames["train"].valid.sum()) < cfg["batch_size"]:
        raise RuntimeError("Too few valid training patches")
    sample_keys = {s: [ci + KEY_OFFSET[s] for ci in pick_samples(frames[s], cfg["n_samples_per_split"])]
                   for s in EVAL_SPLITS}

    state = dict(caches=caches, split_csv=split_csv, sample_keys=sample_keys, crop=cfg["crop"],
                 batch_size=cfg["batch_size"], eval_batch=cfg["eval_batch"], job_workers=job_workers,
                 warmup=cfg["warmup"], deadline=deadline, final_margin_s=final_margin_s, steps=0, val_every=0)

    # ---- 2. identity baseline (also downloads the FID Inception weights once) ------------
    if not job_done("identity"):
        evaluate_and_save("identity", None, scene_eval_sets(dict(state), load_split_frames(state)), device,
                          state, log, identity=True)

    # ---- 3. plan the step budget --------------------------------------------------------
    runs = [r for r in RUNS if not cfg["only"] or r["name"] in cfg["only"]]
    sps = dict(DEFAULT_SEC_PER_STEP)
    remaining = deadline - time.time() - final_margin_s
    if cfg["steps"]:
        steps = cfg["steps"]
        log(f"[plan] fixed budget: {steps} steps per run")
    elif device.type == "cuda":
        try:
            sps.update(benchmark(sorted({bench_key(r) for r in runs}), cfg["batch_size"], cfg["crop"], log))
        except Exception as e:  # never let planning kill the run: fall back to conservative defaults
            log(f"[bench] failed ({type(e).__name__}: {e}) - using default speed estimates")
            torch.cuda.empty_cache()
        todo = [r for r in runs if not job_done(r["name"])]
        steps = choose_steps(todo, sps, slots, deadline - time.time() - final_margin_s, cfg["step_options"], log)
    else:
        steps = cfg["step_options"][-1]
    val_every = max(50, int(round(steps / 15 / 50)) * 50)
    state.update(steps=steps, val_every=val_every, warmup=min(cfg["warmup"], max(1, steps // 10)))
    with open(os.path.join(jobs_dir(), "state.json"), "w") as f:
        json.dump(state, f, indent=1)
    log(f"[plan] {steps} steps per run, validation every {val_every}, {slots} parallel slot(s), "
        f"{remaining / 3600:.2f} h available for training")

    # ---- 4. run the jobs ----------------------------------------------------------------
    script = os.path.join(repo, "notebooks", "kaggle_pipeline.py")
    pending = [r for r in runs if not job_done(r["name"])]
    running = {}
    status = {r["name"]: ("done (earlier)" if job_done(r["name"]) else "pending") for r in runs}
    while pending or running:
        for slot in range(slots):
            if slot in running or not pending:
                continue
            while pending:
                r = pending.pop(0)
                pred = predict_seconds(r, steps, sps)
                if time.time() + pred > deadline - final_margin_s:
                    status[r["name"]] = "skipped (no time)"
                    log(f"[sched] SKIP {r['name']}: needs ~{pred / 60:.0f} min, only "
                        f"{(deadline - final_margin_s - time.time()) / 60:.0f} min left")
                    continue
                env = dict(os.environ)
                if n_gpu:
                    env["CUDA_VISIBLE_DEVICES"] = str(slot % n_gpu)
                proc = subprocess.Popen([sys.executable, "-u", script, "--job", r["name"]], env=env, cwd=repo)
                running[slot] = (r, proc, time.time())
                status[r["name"]] = "running"
                log(f"[sched] START {r['name']} on slot {slot} (predicted {pred / 60:.0f} min)")
                break
        time.sleep(5)
        for slot, (r, proc, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[slot]
            mins = (time.time() - t0) / 60
            if proc.returncode == 0 and job_done(r["name"]):
                status[r["name"]] = f"done ({mins:.0f} min)"
                log(f"[sched] DONE  {r['name']} in {mins:.0f} min")
                try:
                    with open(os.path.join(jobs_dir(), r["name"], "result.json")) as f:
                        info = json.load(f)["info"]
                    if info.get("steps_done"):     # learn the real speed for later predictions
                        sps[bench_key(r)] = max(1e-4, (mins * 60 - eval_seconds(r) - 60) / 1.25 / info["steps_done"])
                except Exception:
                    pass
            else:
                status[r["name"]] = f"FAILED (exit {proc.returncode})"
                log(f"[sched] FAILED {r['name']} (exit {proc.returncode}) - see logs/job_{r['name']}.txt; continuing")
        gc.collect()

    # ---- 5. post-hoc: ensemble (E7c) and effective receptive fields (E3c) ---------------
    from src.models.architectures import load_model_from_checkpoint

    ck = lambda n: os.path.join(out, "checkpoints", f"best_{n}.pt")  # noqa: E731
    members = [n for n in ("A_unet", "A_unet_s43", "A_unet_s44") if job_done(n) and os.path.exists(ck(n))]
    if len(members) >= 2 and not job_done("A_ens") and time.time() < deadline - 5 * 60:
        try:
            models = [load_model_from_checkpoint(ck(n), device)[0] for n in members]
            evaluate_and_save("A_ens", models, scene_eval_sets(state, load_split_frames(state)), device, state, log,
                              info=dict(members=members))
            status["A_ens"] = f"done ({len(members)} members)"
            del models
        except Exception as e:
            log(f"[ens] failed: {e}")
    erf_rows, erfs = [], {}
    try:
        from src.evaluation.analysis import effective_receptive_field, erf_stats
        X = np.load(caches["val"]["x"], mmap_mode="r")
        vi = frames["val"].loc[frames["val"].valid, "cache_idx"].values[:16]
        c0 = (256 - cfg["crop"]) // 2
        xb = torch.from_numpy(np.asarray(X[vi][:, :, c0:c0 + cfg["crop"], c0:c0 + cfg["crop"]], np.float32)).to(device)
        for n in ("tiny_cnn", "A_unet", "A_noskip", "B_swin", "B_noshift", "C_vit", "C_nopos"):
            if job_done(n) and os.path.exists(ck(n)):
                m, _ = load_model_from_checkpoint(ck(n), device)
                erfs[n] = effective_receptive_field(m, xb)
                erf_rows.append(dict(run=n, label=RUN_INDEX[n]["label"], **erf_stats(erfs[n])))
                del m
    except Exception as e:
        log(f"[erf] failed: {e}")

    # ---- 6. aggregate + export ----------------------------------------------------------
    try:
        export(state, frames, caches, erfs, erf_rows, status, steps, log)
    except Exception as e:
        import traceback
        log(f"[export] ERROR {e}\n{traceback.format_exc()}")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(config=cfg, steps_per_run=steps, val_every=val_every, slots=slots, sec_per_step=sps,
                       runs={n: RUN_INDEX[n] for n in status if n in RUN_INDEX}, status=status,
                       labels={n: RUN_INDEX[n]["label"] for n in RUN_INDEX},
                       in_channels=15, out_channels=13,
                       preprocessing=dict(s1="dB, VV clip [-25,0], VH clip [-32.5,0], rescale [0,1]",
                                          s2="clip [0,10000] / 10000, 13 bands", lee=cfg["apply_lee"]),
                       cloud_mask="proxy: cloudy-minus-clear visible reflectance > 0.08",
                       minutes_total=(time.time() - t_start) / 60), f, indent=2, default=str)
    log("=" * 90)
    log("RUN STATUS")
    for n, s in status.items():
        log(f"  {n:12s} {s}")
    log(f"Total time {(time.time() - t_start) / 3600:.2f} h")
    log.f.close()
    zip_path = shutil.make_archive(out, "zip", out)
    print(f"\nDONE - download {zip_path} and extract it into <project>/outputs/", flush=True)


def export(state, frames, caches, erfs, erf_rows, status, steps, log):
    import numpy as np
    import pandas as pd

    from src.evaluation import plots

    out = CONFIG["out_dir"]
    res_dir = os.path.join(out, "results")
    metrics = ["psnr", "ssim", "sam", "mae", "rmse"]
    names = [n for n in RUN_INDEX if job_done(n)]
    pp, results = [], {}
    for n in names:
        d = pd.read_csv(os.path.join(jobs_dir(), n, "per_patch.csv"))
        spec = RUN_INDEX[n]
        for k in ("label", "group", "experiment", "seed", "fraction", "family"):
            d[k] = spec[k]
        d.insert(0, "run", n)
        pp.append(d)
        with open(os.path.join(jobs_dir(), n, "result.json")) as f:
            results[n] = json.load(f)
    pp = pd.concat(pp, ignore_index=True)
    pp.to_csv(os.path.join(res_dir, "per_patch_metrics.csv"), index=False)

    # per run x split
    keys = ["run", "label", "group", "experiment", "seed", "fraction", "split"]
    summary = pp.groupby(keys, sort=False)[metrics].mean().reset_index()
    summary["n_patches"] = pp.groupby(keys, sort=False).size().values
    summary["fid"] = [results[r]["fid"].get(s, np.nan) for r, s in zip(summary.run, summary.split)]
    info = pd.DataFrame([dict(run=n, **{k: v for k, v in results[n]["info"].items() if not isinstance(v, (list, dict))})
                         for n in names])
    if "params" in info:
        summary = summary.merge(info[["run", "params", "train_minutes", "steps_done"]], on="run", how="left")
    summary.to_csv(os.path.join(res_dir, "metrics_summary.csv"), index=False)
    info.to_csv(os.path.join(res_dir, "run_info.csv"), index=False)

    # main comparison: mean +- std over seeds
    main_rows = []
    for g in MAIN_GROUPS:
        sub = summary[(summary.group == g) & summary.split.isin(EVAL_SPLITS)]
        if g == "A_unet":
            sub = sub[sub.experiment.isin(["core", "seeds"])]
        for split, s in sub.groupby("split", sort=False):
            row = dict(group=g, label=RUN_INDEX[g]["label"], split=split, n_seeds=len(s))
            for m in metrics + ["fid"]:
                row[m] = s[m].mean()
                row[f"{m}_std"] = s[m].std(ddof=0) if len(s) > 1 else np.nan
            main_rows.append(row)
    main = pd.DataFrame(main_rows)
    main.to_csv(os.path.join(res_dir, "metrics_main.csv"), index=False)

    # every experiment vs its reference (single seed 42 reference run)
    ex_rows = []
    for n in names:
        spec = RUN_INDEX[n]
        if not spec["ref"] or spec["ref"] not in names:
            continue
        for split in EVAL_SPLITS + ("random_test",):
            a = summary[(summary.run == n) & (summary.split == split)]
            ref_split = "test_id" if split == "random_test" else split
            b = summary[(summary.run == spec["ref"]) & (summary.split == ref_split)]
            if len(a) and len(b):
                row = dict(experiment=spec["experiment"], run=n, label=spec["label"], reference=spec["ref"],
                           reference_label=RUN_INDEX[spec["ref"]]["label"], split=split, question=spec["question"])
                for m in metrics + ["fid"]:
                    row[m], row[f"ref_{m}"] = a.iloc[0][m], b.iloc[0][m]
                    row[f"delta_{m}"] = a.iloc[0][m] - b.iloc[0][m]
                ex_rows.append(row)
    experiments = pd.DataFrame(ex_rows)
    experiments.to_csv(os.path.join(res_dir, "experiments.csv"), index=False)
    if len(experiments):
        experiments[experiments.experiment == "leakage"].to_csv(os.path.join(res_dir, "leakage_check.csv"), index=False)

    # E1 learning curves (100% point = seed-42 core run)
    lc = summary[summary.split.isin(EVAL_SPLITS) & (summary.experiment.isin(["E1", "core"]))].copy()
    lc["model"] = lc.run.map(lambda r: RUN_INDEX[RUN_INDEX[r]["ref"] or r]["label"])
    lc.to_csv(os.path.join(res_dir, "learning_curves.csv"), index=False)

    # cloud bins / countries
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    pp["cloud_bin"] = pd.cut(pp.cloud_frac, bins, labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
                             include_lowest=True)
    gk = ["run", "label", "group", "experiment", "split", "cloud_bin"]
    by_bin = pp.groupby(gk, sort=False, observed=True)[metrics].mean()
    by_bin["n_patches"] = pp.groupby(gk, sort=False, observed=True).size()
    by_bin = by_bin.reset_index().sort_values(["split", "run", "cloud_bin"])
    by_bin.to_csv(os.path.join(res_dir, "metrics_by_cloud_bin.csv"), index=False)
    gk = ["run", "label", "group", "experiment", "split", "country"]
    by_c = pp.groupby(gk, sort=False)[metrics].mean()
    by_c["n_patches"] = pp.groupby(gk, sort=False).size()
    by_c.reset_index().to_csv(os.path.join(res_dir, "metrics_by_country.csv"), index=False)

    # uncertainty: variance head vs MC dropout vs ensemble
    method = {"A_unc": "Variance head (aleatoric)", "A_mcd": "MC dropout (epistemic)", "A_ens": "Deep ensemble (epistemic)"}
    u_rows, curves, sp_rows = [], {}, []
    for n, mlabel in method.items():
        if n not in results:
            continue
        for split, s in results[n]["unc"].items():
            u_rows.append(dict(run=n, method=mlabel, split=split, kind=results[n]["unc_kind"], **s))
            c = results[n]["curves"][split]
            curves[(mlabel, split)] = {k: np.asarray(v) if isinstance(v, list) else v for k, v in c.items()}
            sp_rows.append(pd.DataFrame(dict(run=n, method=mlabel, split=split, fraction=c["fractions"],
                                             model_mae=c["model"], oracle_mae=c["oracle"])))
    if u_rows:
        pd.DataFrame(u_rows).to_csv(os.path.join(res_dir, "uncertainty_summary.csv"), index=False)
        pd.concat(sp_rows).to_csv(os.path.join(res_dir, "sparsification.csv"), index=False)
    if erf_rows:
        pd.DataFrame(erf_rows).to_csv(os.path.join(res_dir, "erf_stats.csv"), index=False)

    # figures (each guarded so one failure never loses the rest)
    labels = {n: RUN_INDEX[n]["label"] for n in RUN_INDEX}

    def safe(fn, *a):
        try:
            fn(*a)
        except Exception as e:
            log(f"[plot] {fn.__name__} failed: {e}")

    hist = {}
    for n in [g for g in MAIN_GROUPS if g in names]:
        hp = os.path.join(out, "logs", f"history_{n}.csv")
        hist[n] = pd.read_csv(hp) if os.path.exists(hp) else None
    safe(plots.training_curves, hist, labels, os.path.join(res_dir, "training_curves.png"))
    if len(main):
        safe(plots.metric_bars, main, os.path.join(res_dir, "metrics_comparison.png"))
    safe(plots.cloud_bin_plot, by_bin[by_bin.run.isin(MAIN_GROUPS)], os.path.join(res_dir, "cloud_bin_psnr.png"))
    if (lc.experiment == "E1").any():
        safe(plots.learning_curve_plot, lc, os.path.join(res_dir, "learning_curves.png"))
    if erfs:
        safe(plots.erf_grid, erfs, labels, {r["run"]: r for r in erf_rows}, os.path.join(res_dir, "erf.png"))
    if curves:
        safe(plots.uncertainty_methods_plot, curves, os.path.join(res_dir, "uncertainty_methods.png"))

    # gallery samples for the frontend
    gal = [n for n in GALLERY_RUNS if n in names]
    npz, meta, grid = {"x": [], "y": []}, [], []
    for n in gal:
        npz[f"pred_{n}"] = []
        npz[f"sigma_{n}"] = []
    samp = {n: dict(np.load(os.path.join(jobs_dir(), n, "samples.npz")))
            for n in gal if os.path.exists(os.path.join(jobs_dir(), n, "samples.npz"))}
    for split in EVAL_SPLITS:
        X = np.load(caches[split]["x"], mmap_mode="r")
        Y = np.load(caches[split]["y"], mmap_mode="r")
        fr = frames[split].set_index("cache_idx")
        for key in state["sample_keys"][split]:
            ci = key - KEY_OFFSET[split]
            row = fr.loc[ci]
            x, y = np.asarray(X[ci], np.float32), np.asarray(Y[ci], np.float32)
            scores = {}
            for n in gal:
                m = pp[(pp.run == n) & (pp.split == split) & (pp.key == key)]
                if len(m):
                    scores[n] = {k: float(m.iloc[0][k]) for k in metrics}
                rgb = samp.get(n, {}).get(f"rgb_{split}_{key}")
                sig = samp.get(n, {}).get(f"sigma_{split}_{key}")
                npz[f"pred_{n}"].append(rgb if rgb is not None else np.zeros(y.shape[1:] + (3,), np.uint8))
                npz[f"sigma_{n}"].append(sig if sig is not None else np.zeros(y.shape[1:], np.float16))
            npz["x"].append(x.astype(np.float16))
            npz["y"].append(y.astype(np.float16))
            meta.append(dict(split=split, season=row.season, scene=int(row.scene), patch=int(row.patch),
                             country=row.country, continent=row.continent, cloud_frac=float(row.cloud_frac),
                             scores=scores))
            grid.append(dict(x=x, y=y, title=f"{plots.SPLIT_LABELS[split].split(' (')[0]}\n{row.country} "
                                              f"{int(row.cloud_frac * 100)}% cloud",
                             preds_rgb={n: npz[f"pred_{n}"][-1] for n in gal},
                             sigma=npz["sigma_A_unc"][-1].astype(np.float32) if "A_unc" in gal else None,
                             scores={n: f"{v['psnr']:.1f} dB" for n, v in scores.items()}))
    if meta:
        np.savez_compressed(os.path.join(res_dir, "samples.npz"), **{k: np.stack(v) for k, v in npz.items()},
                            meta=np.array(json.dumps(dict(samples=meta, runs=gal, labels=labels))))
        safe(plots.comparison_grid, grid[:2] + grid[len(grid) // 2:len(grid) // 2 + 2], {n: labels[n] for n in gal},
             os.path.join(res_dir, "master_architecture_comparison.png"))

    log("=" * 90)
    log(f"MAIN RESULTS (mean over seeds, {steps} steps per run)")
    if len(main):
        log("\n" + main[["label", "split", "n_seeds"] + metrics + ["fid"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
    if len(experiments):
        log("\nEXPERIMENTS (delta vs reference)")
        log("\n" + experiments[["experiment", "label", "split", "delta_psnr", "delta_ssim", "delta_sam", "delta_fid"]]
            .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
