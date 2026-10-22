"""
ClearSAR plan v4 - build ONE model, stage by stage, in one Kaggle session (docs/project_plan_v4.md)
=================================================================================================

HOW TO RUN ON KAGGLE  (same notebook settings as Phase 1: GPU T4 x2, Internet on, 4 SEN12MS-CR inputs)
    !rm -rf /kaggle/working/*
    !git clone -q https://github.com/karthikeya-bhamidipati/clearsar.git /kaggle/working/clearsar
    !python /kaggle/working/clearsar/notebooks/kaggle_pipeline_v4.py
  smoke test first (~30 min):   !CLEARSAR_QUICK=1 python /kaggle/working/clearsar/notebooks/kaggle_pipeline_v4.py
  Then "Save Version -> Save & Run All". Download clearsar_v4_output.zip and extract it into
  <project>/outputs/v4/  (so the app finds outputs/v4/results/...).

WHAT IT DOES
  data      worldwide split by scene: ~132 train / 18 val / 25 test scenes; all 11 Phase-1 test scenes are in
            the test set, so Phase-1 and v4 models are compared on identical scenes
  stage 0   data study: how different are cloudy and clear images even on visible ground? (oracle ceilings)
  stage 1   diagnostics (overfit one batch, LR range test) + plain U-Net on the new data      s1_direct
  stage 2   keep what the satellite saw: residual / gated output                     s2_residual, s2_gated
            -> decision "mode"   = lowest validation L1 of {s1_direct, s2_residual, s2_gated}
  stage 3   scene diversity at fixed size: 3 000 patches from 33 / 66 / 132 scenes     s3_sc33/66/132
  stage 4   losses: + SSIM, + perceptual, + small adversarial                      s4_ssim, s4_perc, s4_adv
            -> decision "loss"   = lowest balanced score  val_L1 * (1 + |1 - sharpness|)
  stage 5   context: global attention bottleneck vs a wider U-Net of equal size       s5_attn, s5_wide
            -> decision "context"
  stage 6   the final model: all winners, full 256 x 256 training, 2x steps, EMA, 2 seeds + uncertainty head
  stage 7   every model (incl. Phase 1) evaluated ONCE on the test scenes; decisions above use validation only
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
FAST_FINAL = _env("FAST_FINAL", 0, int) == 1 or "--final" in sys.argv
CONFIG = dict(
    input_roots=_env("INPUT_ROOTS", "/kaggle/input/*").split(";"),
    out_dir=_env("OUT_DIR", "/kaggle/working/clearsar_v4_output"),
    resume_dir=_env("RESUME_DIR", "/kaggle/working/clearsar_v4_resume"),
    cache_dir=_env("CACHE_DIR", "/tmp/clearsar_cache_v4"),
    caps=_caps(_env("CAPS", "train=600,val=96,test=160" if QUICK else "train=12000,val=1200,test=2500")),
    n_val=_env("N_VAL", 18, int), n_test=_env("N_TEST", 25, int),
    stage3_patches=_env("S3_PATCHES", 300 if QUICK else 3000, int),
    stage3_scenes=[33, 66, 132],
    crop=128, batch_size=_env("BATCH", 16, int), full_batch=_env("FULL_BATCH", 4, int), eval_batch=4,
    steps=_env("STEPS", 300 if QUICK else 15000, int),
    final_steps=_env("FINAL_STEPS", 400 if QUICK else (15000 if FAST_FINAL else 30000), int),
    val_every=_env("VAL_EVERY", 100 if QUICK else 1000, int),
    warmup=500,
    hours=_env("HOURS", 1.5 if QUICK else (2.0 if FAST_FINAL else 11.0), float),
    final_margin_min=_env("FINAL_MARGIN", 8 if QUICK else 30, float),
    slots=_env("SLOTS", 0, int),
    workers=_env("WORKERS", min(4, os.cpu_count() or 2), int),
    only=[r for r in _env("RUNS", "").split(",") if r] or (["v4_final", "p1_A_unet", "p1_B_swin", "p1_C_vit", "p1_tiny_cnn"] if FAST_FINAL else []),
    seed=42,
)
NET = "clearsar_net"
LOSSES = {   # stage-4 candidates (weights chosen so each term is of the order of the L1 term)
    "l1": dict(loss="l1"),
    "ssim": dict(loss="combo", w_ssim=0.2),
    "perc": dict(loss="combo", w_ssim=0.2, w_perc=0.05),
    "adv": dict(loss="combo", w_ssim=0.2, w_perc=0.05, use_gan=True, adv_weight=0.003, lr=2e-4),
}
CONTEXT = {"none": {}, "attn": dict(attn_layers=2), "wide": dict(width=80)}


# ---------------------------------------------------------------------------------- the jobs
def J(name, stage, label, build, deps=(), kind="train", minutes=28):
    return dict(name=name, stage=stage, label=label, build=build, deps=list(deps), kind=kind, minutes=minutes)


def _std(mode, extra_model=None, **run):
    return dict(model_kwargs=dict(output_mode=mode, **(extra_model or {})), **run)


JOBS = [
    J("s1_direct", 1, "Stage 1 - plain U-Net (direct output)", lambda d: _std("direct")),
    J("s2_residual", 2, "Stage 2 - residual output", lambda d: _std("residual")),
    J("s2_gated", 2, "Stage 2 - gated output (learned cloud mask)", lambda d: _std("gated")),
    J("s3_sc33", 3, "Stage 3 - 3k patches from 33 scenes", lambda d: _std(d["mode"], scenes=33), deps=["mode"]),
    J("s3_sc66", 3, "Stage 3 - 3k patches from 66 scenes", lambda d: _std(d["mode"], scenes=66), deps=["mode"]),
    J("s3_sc132", 3, "Stage 3 - 3k patches from 132 scenes", lambda d: _std(d["mode"], scenes=132), deps=["mode"]),
    J("s4_ssim", 4, "Stage 4 - L1 + SSIM", lambda d: _std(d["mode"], select="balanced", **LOSSES["ssim"]), deps=["mode"], minutes=30),
    J("s4_perc", 4, "Stage 4 - L1 + SSIM + perceptual", lambda d: _std(d["mode"], select="balanced", **LOSSES["perc"]), deps=["mode"], minutes=38),
    J("s4_adv", 4, "Stage 4 - L1 + SSIM + perceptual + adversarial", lambda d: _std(d["mode"], select="balanced", **LOSSES["adv"]), deps=["mode"], minutes=45),
    J("s5_attn", 5, "Stage 5 - global attention bottleneck",
      lambda d: _std(d["mode"], CONTEXT["attn"], select="balanced", **LOSSES[d["loss"]]), deps=["loss"], minutes=36),
    J("s5_wide", 5, "Stage 5 - wider U-Net (capacity control)",
      lambda d: _std(d["mode"], CONTEXT["wide"], select="balanced", **LOSSES[d["loss"]]), deps=["loss"], minutes=36),
    J("v4_final", 6, "ClearSAR-Net final (seed 42)", lambda d: _final(d, 42), deps=["context"], minutes=75),
    J("v4_final_s43", 6, "ClearSAR-Net final (seed 43)", lambda d: _final(d, 43), deps=["context"], minutes=75),
    J("v4_final_unc", 6, "ClearSAR-Net final + uncertainty head", lambda d: _final(d, 42, unc=True), deps=["context"], minutes=78),
    J("p1_A_unet", 7, "Phase 1 - Model A U-Net", None, kind="eval_p1", minutes=4),
    J("p1_B_swin", 7, "Phase 1 - Model B Swin U-Net", None, kind="eval_p1", minutes=4),
    J("p1_C_vit", 7, "Phase 1 - Model C pure ViT", None, kind="eval_p1", minutes=4),
    J("p1_tiny_cnn", 7, "Phase 1 - tiny CNN", None, kind="eval_p1", minutes=4),
]
JOB_INDEX = {j["name"]: j for j in JOBS}
DECISIONS = {
    "mode": dict(candidates=["s1_direct", "s2_residual", "s2_gated"], rule="l1",
                 value={"s1_direct": "direct", "s2_residual": "residual", "s2_gated": "gated"}, default="gated"),
    "loss": dict(candidates=["@mode", "s4_ssim", "s4_perc", "s4_adv"], rule="balanced",
                 value={"@mode": "l1", "s4_ssim": "ssim", "s4_perc": "perc", "s4_adv": "adv"}, default="perc",
                 needs=["mode"]),
    "context": dict(candidates=["@loss", "s5_attn", "s5_wide"], rule="balanced",
                    value={"@loss": "none", "s5_attn": "attn", "s5_wide": "wide"}, default="attn", needs=["loss"]),
}
DECISION_RUN = {"mode": {"direct": "s1_direct", "residual": "s2_residual", "gated": "s2_gated"},
                "loss": {"l1": "@mode", "ssim": "s4_ssim", "perc": "s4_perc", "adv": "s4_adv"}}


def _final(d, seed, unc=False):
    run = _std(d["mode"], CONTEXT[d["context"]], select="balanced", ema_decay=0.999, full_res=True,
               steps=CONFIG["final_steps"], val_every=2 * CONFIG["val_every"], seed=seed, uncertainty=unc,
               **LOSSES[d["loss"]])
    return run


def run_of(candidate, decisions):
    """'@mode' -> the run that won decision 'mode' (following chains such as @loss -> @mode -> s2_gated)."""
    while candidate.startswith("@"):
        key = candidate[1:]
        candidate = DECISION_RUN[key][decisions.get(key, DECISIONS[key]["default"])]
    return candidate


# ---------------------------------------------------------------------------------- helpers
def setup():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
    repo = None
    if script_dir and os.path.isdir(os.path.join(script_dir, "..", "src")):
        repo = os.path.abspath(os.path.join(script_dir, ".."))
    if ON_KAGGLE and not IS_JOB:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio", "torchmetrics", "torch-fidelity"],
                       check=False)
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


def history_score(name, rule):
    """Decision score of a finished run, from its validation history (lower is better)."""
    import pandas as pd
    p = os.path.join(CONFIG["out_dir"], "logs", f"history_{name}.csv")
    if not os.path.exists(p):
        return float("inf")
    h = pd.read_csv(p)
    if rule == "balanced" and "val_sharpness" in h:
        return float((h.val_l1 * (1 + (1 - h.val_sharpness).abs())).min())
    return float(h.val_l1.min())


def resolve(decisions, key, log):
    """Decide `key` once all its candidate runs are finished (validation data only)."""
    if key in decisions:
        return True
    spec = DECISIONS[key]
    for need in spec.get("needs", []):
        if not resolve(decisions, need, log):
            return False
    cands = []
    for c in spec["candidates"]:
        run = run_of(c, decisions)
        if run in JOB_INDEX and not job_done(run) and not os.path.exists(os.path.join(jobs_dir(), run, "FAILED")):
            return False                                         # still waiting for a candidate
        cands.append((c, run))
    scores = {c: history_score(run, spec["rule"]) for c, run in cands if job_done(run)}
    if not scores or all(v == float("inf") for v in scores.values()):
        decisions[key] = spec["default"]
        log(f"[decide] {key}: no finished candidate - default {spec['default']!r}")
    else:
        best = min(scores, key=scores.get)
        decisions[key] = spec["value"][best]
        log(f"[decide] {key} = {decisions[key]!r}   (" + ", ".join(
            f"{spec['value'][c]}: {v:.5f}" for c, v in scores.items()) + f"; rule {spec['rule']}, validation only)")
    with open(os.path.join(jobs_dir(), "decisions.json"), "w") as f:
        json.dump(decisions, f, indent=1)
    return True


def pick_gallery(frame, n_asia=4, n_other=4):
    import numpy as np
    d = frame[frame.valid]
    picks = []
    for part, n in ((d[d.continent == "Asia"], n_asia), (d[d.continent != "Asia"], n_other)):
        part = part.sort_values("cloud_frac")
        if part.empty:
            continue
        if "IN" in set(part.country) and n:
            r = part[part.country == "IN"].iloc[len(part[part.country == "IN"]) // 2]
            picks.append(int(r.cache_idx))
            n -= 1
        for q in np.linspace(0.2, 0.92, max(n, 1)) if n else []:
            ci = int(part.iloc[int(round(q * (len(part) - 1)))].cache_idx)
            if ci not in picks:
                picks.append(ci)
    return picks


# ---------------------------------------------------------------------------------- one job
def run_job(name):
    repo = setup()
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import (IN_CHANNELS, OUT_CHANNELS, RGB_IDX, CachedPatchDataset, scene_subset_indices)
    from src.evaluation.evaluate import evaluate
    from src.evaluation.metrics import uncertainty_summary
    from src.models.architectures import load_model_from_checkpoint
    from src.training.trainer import RunConfig, train_run

    with open(os.path.join(jobs_dir(), "state.json")) as f:
        state = json.load(f)
    with open(os.path.join(jobs_dir(), "decisions.json")) as f:
        decisions = json.load(f)
    out = CONFIG["out_dir"]
    log = Logger(os.path.join(out, "logs", f"job_{name}.txt"))
    job = JOB_INDEX[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    frames = {s: pd.read_csv(p) for s, p in state["split_csv"].items()}
    caches, workers = state["caches"], state["job_workers"]

    def vidx(s):
        return frames[s].loc[frames[s].valid, "cache_idx"].values

    def loader(ds, bs, shuffle):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle, num_workers=workers,
                          pin_memory=device.type == "cuda", persistent_workers=workers > 0)

    info, tile, model = {}, 256, None
    if job["kind"] == "train":
        spec = job["build"](decisions)
        run = {k: v for k, v in spec.items() if k not in ("scenes", "full_res")}
        if spec.get("scenes"):
            tr_idx, n_sc = scene_subset_indices(frames["train"], spec["scenes"], state["stage3_patches"])
            log(f"[{name}] {len(tr_idx)} patches from {n_sc} training scenes")
        else:
            tr_idx = vidx("train")
        full = spec.get("full_res", False)
        train_ds = CachedPatchDataset(caches["train"], tr_idx, crop=None if full else state["crop"], augment=True)
        val_ds = CachedPatchDataset(caches["val"], vidx("val"))
        kw = dict(warmup=state["warmup"], val_every=state["val_every"], steps=state["steps"], eval_tile=256,
                  deadline=state["deadline"] - state["final_margin_s"] - 300)
        kw.update(run)
        rc = RunConfig(name=name, arch=NET, label=job["label"], **kw)
        log(f"[{name}] decisions so far {decisions} -> {spec}")
        info = train_run(rc, loader(train_ds, state["full_batch"] if full else state["batch_size"], True),
                         loader(val_ds, state["eval_batch"], False), device, os.path.join(out, "checkpoints"),
                         os.path.join(out, "logs"), IN_CHANNELS, OUT_CHANNELS, resume_dir=CONFIG["resume_dir"], log=log)
        model, meta = load_model_from_checkpoint(os.path.join(out, "checkpoints", f"best_{name}.pt"), device)
        unc = meta.get("uncertainty", False)
    else:  # Phase-1 checkpoint evaluated on the v4 test set (trained on 128 crops -> tiled 128)
        ck = os.path.join(state["phase1_ckpt_dir"], f"best_{name[3:]}.pt")
        model, meta = load_model_from_checkpoint(ck, device)
        unc, tile = False, 128
        info = dict(name=name, label=job["label"], params=sum(p.numel() for p in model.parameters()))

    jd = os.path.join(jobs_dir(), name)
    os.makedirs(jd, exist_ok=True)
    keys = set(state["gallery"])
    result = dict(name=name, info=info, fid=float("nan"), unc={})
    test_ds = CachedPatchDataset(caches["test"], vidx("test"))
    res, ex = evaluate(model, loader(test_ds, state["eval_batch"], False), device, OUT_CHANNELS, uncertainty=unc,
                       fid=True, collect=keys, tile=tile)
    d = pd.DataFrame(res).merge(frames["test"][["cache_idx", "season", "scene", "patch", "country", "continent",
                                                "cloud_frac"]], on="cache_idx", how="left")
    d.to_csv(os.path.join(jd, "per_patch.csv"), index=False)
    result["fid"] = ex.get("fid", float("nan"))
    if "unc_errors" in ex:
        result["unc"], _ = uncertainty_summary(ex["unc_errors"], ex["unc_sigmas"], ex["unc_kind"])
    samples = {}
    for key, (pred, sigma) in ex.get("samples", {}).items():
        samples[f"rgb_{key}"] = (np.clip(np.stack([pred[i] for i in RGB_IDX], -1) * 3.0, 0, 1) * 255).astype(np.uint8)
        if sigma is not None:
            samples[f"sigma_{key}"] = sigma.astype(np.float16)
    if getattr(model, "output_mode", None) == "gated":      # where does the model trust the input?
        X = np.load(caches["test"]["x"], mmap_mode="r")
        with torch.no_grad():
            for key in keys:
                x = torch.from_numpy(np.asarray(X[key], np.float32))[None].to(device)
                _, aux = model(x, return_aux=True)
                samples[f"gate_{key}"] = aux["gate"][0, 0].float().cpu().numpy().astype(np.float16)
    if samples:
        np.savez_compressed(os.path.join(jd, "samples.npz"), **samples)
    with open(os.path.join(jd, "result.json"), "w") as f:
        json.dump(result, f, default=float)
    open(os.path.join(jd, "DONE"), "w").close()
    log(f"[{name}] TEST PSNR {d.psnr.mean():.2f} | aligned {d.psnr_aligned.mean():.2f} | SSIM {d.ssim.mean():.4f} | "
        f"sharp {d.sharpness.mean():.2f} | colour {d.colour_err.mean():.4f} | FID {result['fid']:.1f}")


# ---------------------------------------------------------------------------------- main
def main():
    if IS_JOB:
        return run_job(sys.argv[sys.argv.index("--job") + 1])
    t_start = time.time()
    repo = setup()
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import (CachedPatchDataset, attach_scene_metadata, build_cache, index_triplets,
                                  make_global_splits)
    from src.evaluation.data_study import run_data_study
    from src.evaluation.evaluate import evaluate
    from src.training.trainer import lr_range_test, overfit_one_batch

    cfg = CONFIG
    out = cfg["out_dir"]
    for sub in ("results", "checkpoints", "logs", "jobs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    log = Logger(os.path.join(out, "logs", "pipeline_log.txt"))
    deadline = t_start + cfg["hours"] * 3600
    final_margin_s = cfg["final_margin_min"] * 60
    log("=" * 90)
    log(f"ClearSAR plan v4 | repo {repo} | quick={QUICK} | fast_final={FAST_FINAL} | budget {cfg['hours']} h")
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpu else "cpu")
    for g in range(n_gpu):
        log(f"[gpu] cuda:{g} {torch.cuda.get_device_name(g)}, torch {torch.__version__}")
    slots = cfg["slots"] or max(1, n_gpu)

    # ---- data -----------------------------------------------------------------------
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    ip = os.path.join(cfg["cache_dir"], "index.csv")
    triplets = pd.read_csv(ip) if os.path.exists(ip) else index_triplets(cfg["input_roots"], log=log)
    triplets.to_csv(ip, index=False)
    triplets = attach_scene_metadata(triplets, os.path.join(repo, "data", "scans", "scenes.csv"), log=log)
    p1_path = os.path.join(repo, "outputs", "results", "scene_splits.csv")
    p1 = pd.read_csv(p1_path) if os.path.exists(p1_path) else None
    log(f"[split] Phase-1 scene roles {'kept' if p1 is not None else 'not found'}")
    patches, scenes = make_global_splits(triplets, p1, cfg["n_val"], cfg["n_test"], cfg["caps"], cfg["seed"], log=log)
    scenes.to_csv(os.path.join(out, "results", "v4_scene_splits.csv"), index=False)
    caches, split_csv, frames = {}, {}, {}
    for s in ("train", "val", "test"):
        t0 = time.time()
        caches[s], d = build_cache(patches[patches.split == s], cfg["cache_dir"], s, num_workers=cfg["workers"], log=log)
        d = d.drop(columns=["s1", "s2", "s2_cloudy"])
        split_csv[s] = os.path.join(cfg["cache_dir"], f"split_{s}.csv")
        d.to_csv(split_csv[s], index=False)
        frames[s] = d
        log(f"[cache] {s}: {int(d.valid.sum())} valid patches ({time.time() - t0:.0f}s)")

    # ---- stage 0: data study (validation scenes) ------------------------------------
    res_dir = os.path.join(out, "results")
    try:
        run_data_study(np.load(caches["val"]["x"], mmap_mode="r"), np.load(caches["val"]["y"], mmap_mode="r"),
                       frames["val"], res_dir, "validation", log=log)
    except Exception as e:
        log(f"[stage0] failed: {e}")

    # ---- stage 1 diagnostics ----------------------------------------------------------
    try:
        tr = frames["train"]
        ds = CachedPatchDataset(caches["train"], tr.loc[tr.valid, "cache_idx"].values, crop=cfg["crop"], augment=True)
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=0)
        xb, yb, _ = next(iter(dl))
        of = overfit_one_batch(NET, dict(output_mode="direct"), (xb[:8], yb[:8]), device, steps=60 if QUICK else 300)
        pd.DataFrame(of, columns=["step", "l1"]).to_csv(os.path.join(res_dir, "stage1_overfit.csv"), index=False)
        log(f"[stage1] overfit one batch: L1 {of[0][1]:.4f} -> {of[-1][1]:.4f}")
        lr_rows, lr_sug = lr_range_test(NET, dict(output_mode="direct"), dl, device, steps=60 if QUICK else 300)
        pd.DataFrame(lr_rows, columns=["lr", "loss", "smoothed"]).to_csv(os.path.join(res_dir, "stage1_lr_range.csv"),
                                                                       index=False)
        log(f"[stage1] LR range test: loss blows up near {lr_sug * 10:.1e} -> suggested ~{lr_sug:.1e} "
            f"(all runs use 3e-4, GAN 2e-4)")
        del dl, ds
        torch.cuda.empty_cache() if n_gpu else None
    except Exception as e:
        log(f"[stage1] diagnostics failed: {e}")

    # ---- identity baseline + FID weights ----------------------------------------------
    jd = os.path.join(jobs_dir(), "identity")
    if not job_done("identity"):
        te = frames["test"]
        dl = DataLoader(CachedPatchDataset(caches["test"], te.loc[te.valid, "cache_idx"].values), batch_size=4)
        r, ex = evaluate(None, dl, device, 13, identity=True, fid=True)
        os.makedirs(jd, exist_ok=True)
        pd.DataFrame(r).merge(te[["cache_idx", "season", "scene", "patch", "country", "continent", "cloud_frac"]],
                              on="cache_idx").to_csv(os.path.join(jd, "per_patch.csv"), index=False)
        with open(os.path.join(jd, "result.json"), "w") as f:
            json.dump(dict(name="identity", info={}, fid=ex.get("fid", float("nan")), unc={}), f, default=float)
        open(os.path.join(jd, "DONE"), "w").close()
    if n_gpu:  # download the VGG16 weights once, before parallel jobs need them
        try:
            from src.training.loss import PerceptualLoss
            PerceptualLoss()
        except Exception as e:
            log(f"[vgg] weights not available ({e}) - the perceptual run will fail and be skipped")

    gallery = pick_gallery(frames["test"])
    state = dict(caches=caches, split_csv=split_csv, crop=cfg["crop"], batch_size=cfg["batch_size"],
                 full_batch=cfg["full_batch"], eval_batch=cfg["eval_batch"], steps=cfg["steps"],
                 val_every=cfg["val_every"], warmup=min(cfg["warmup"], max(1, cfg["steps"] // 10)),
                 stage3_patches=cfg["stage3_patches"], deadline=deadline, final_margin_s=final_margin_s,
                 job_workers=max(1, cfg["workers"] // slots), gallery=gallery,
                 phase1_ckpt_dir=os.path.join(repo, "outputs", "checkpoints"))
    with open(os.path.join(jobs_dir(), "state.json"), "w") as f:
        json.dump(state, f, indent=1)
    dpath = os.path.join(jobs_dir(), "decisions.json")
    decisions = json.load(open(dpath)) if os.path.exists(dpath) else {}
    with open(dpath, "w") as f:
        json.dump(decisions, f)

    # ---- scheduler with dependencies --------------------------------------------------
    jobs = [j for j in JOBS if not cfg["only"] or j["name"] in cfg["only"]]
    if not os.path.isdir(state["phase1_ckpt_dir"]):
        jobs = [j for j in jobs if j["kind"] != "eval_p1"]
    scale = (cfg["steps"] / 15000) if not QUICK else 0.1
    pending = [j for j in jobs if not job_done(j["name"])]
    running, status = {}, {j["name"]: "done (earlier)" for j in jobs if job_done(j["name"])}
    script = os.path.join(repo, "notebooks", "kaggle_pipeline_v4.py")
    while pending or running:
        for slot in range(slots):
            if slot in running:
                continue
            for j in list(pending):
                if not all(resolve(decisions, k, log) for k in j["deps"]):
                    continue
                pred = j["minutes"] * 60 * max(scale, 0.1) + 120
                if time.time() + pred > deadline - final_margin_s:
                    pending.remove(j)
                    status[j["name"]] = "skipped (no time)"
                    os.makedirs(os.path.join(jobs_dir(), j["name"]), exist_ok=True)
                    open(os.path.join(jobs_dir(), j["name"], "FAILED"), "w").close()
                    log(f"[sched] SKIP {j['name']} - not enough time left")
                    continue
                env = dict(os.environ)
                if n_gpu:
                    env["CUDA_VISIBLE_DEVICES"] = str(slot % n_gpu)
                running[slot] = (j, subprocess.Popen([sys.executable, "-u", script, "--job", j["name"]], env=env,
                                                     cwd=repo), time.time())
                pending.remove(j)
                status[j["name"]] = "running"
                log(f"[sched] START {j['name']} (stage {j['stage']}) on slot {slot}")
                break
        if not running and pending:   # nothing runnable: a dependency can never be met -> decide with defaults
            for j in pending:
                for k in j["deps"]:
                    if k not in decisions:
                        decisions[k] = DECISIONS[k]["default"]
                        log(f"[decide] {k}: forced default {decisions[k]!r}")
            with open(dpath, "w") as f:
                json.dump(decisions, f)
        time.sleep(5)
        for slot, (j, proc, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[slot]
            mins = (time.time() - t0) / 60
            if proc.returncode == 0 and job_done(j["name"]):
                status[j["name"]] = f"done ({mins:.0f} min)"
                log(f"[sched] DONE  {j['name']} in {mins:.0f} min")
            else:
                status[j["name"]] = f"FAILED (exit {proc.returncode})"
                os.makedirs(os.path.join(jobs_dir(), j["name"]), exist_ok=True)
                open(os.path.join(jobs_dir(), j["name"], "FAILED"), "w").close()
                log(f"[sched] FAILED {j['name']} - see logs/job_{j['name']}.txt; continuing")
        gc.collect()
    for k in DECISIONS:
        resolve(decisions, k, log)

    # ---- export -----------------------------------------------------------------------
    try:
        export(state, frames, caches, decisions, status, log)
    except Exception as e:
        import traceback
        log(f"[export] ERROR {e}\n{traceback.format_exc()}")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(config=cfg, decisions=decisions, status=status, losses=LOSSES, context=CONTEXT,
                       labels={j["name"]: j["label"] for j in JOBS} | {"identity": "Identity (cloudy input)"},
                       minutes_total=(time.time() - t_start) / 60), f, indent=2, default=str)
    log("=" * 90)
    log(f"DECISIONS (validation only): {decisions}")
    log("RUN STATUS")
    for n, s in status.items():
        log(f"  {n:14s} {s}")
    log(f"Total time {(time.time() - t_start) / 3600:.2f} h")
    log.f.close()
    zp = shutil.make_archive(out, "zip", out, ".", )
    print(f"\nDONE - download {zp} and extract it into <project>/outputs/v4/", flush=True)


def export(state, frames, caches, decisions, status, log):
    import numpy as np
    import pandas as pd

    from src.evaluation import plots

    out = CONFIG["out_dir"]
    res = os.path.join(out, "results")
    labels = {j["name"]: j["label"] for j in JOBS} | {"identity": "Identity (cloudy input)"}
    stage = {j["name"]: j["stage"] for j in JOBS} | {"identity": 0}
    done = [n for n in ["identity"] + [j["name"] for j in JOBS] if job_done(n)]
    pp, results = [], {}
    for n in done:
        d = pd.read_csv(os.path.join(jobs_dir(), n, "per_patch.csv"))
        d.insert(0, "run", n)
        pp.append(d)
        results[n] = json.load(open(os.path.join(jobs_dir(), n, "result.json")))
    pp = pd.concat(pp, ignore_index=True)
    pp.to_csv(os.path.join(res, "v4_per_patch.csv"), index=False)
    metrics = ["psnr", "psnr_aligned", "ssim", "sam", "mae", "sharpness", "colour_err", "detail_err"]
    summ = pp.groupby("run", sort=False)[metrics].mean().reset_index()
    summ.insert(1, "label", summ.run.map(labels))
    summ.insert(2, "stage", summ.run.map(stage))
    summ["fid"] = summ.run.map(lambda r: results[r].get("fid", np.nan))
    for k in ("params", "best_step", "best_val_l1", "best_val_psnr", "best_val_sharpness", "best_score", "train_minutes"):
        summ[k] = summ.run.map(lambda r: results[r].get("info", {}).get(k, np.nan))
    summ.to_csv(os.path.join(res, "v4_runs.csv"), index=False)

    asia = pp.assign(region=np.where(pp.continent == "Asia", "Asia", "Other continents"))
    by_region = pd.concat([asia.groupby(["run", "region"])[metrics].mean().reset_index(),
                           pp[pp.country == "IN"].groupby("run")[metrics].mean().reset_index().assign(region="India")])
    by_region.to_csv(os.path.join(res, "v4_by_region.csv"), index=False)
    pp["cloud_bin"] = pd.cut(pp.cloud_frac, [0, .2, .4, .6, .8, 1.0001], include_lowest=True,
                             labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
    pp.groupby(["run", "cloud_bin"], observed=True)[metrics].mean().reset_index().to_csv(
        os.path.join(res, "v4_by_cloud_bin.csv"), index=False)
    pp.groupby(["run", "continent"])[metrics].mean().reset_index().to_csv(os.path.join(res, "v4_by_continent.csv"),
                                                                          index=False)
    # one row per stage decision, for the "Building the model" page
    rows = []
    for key, spec in DECISIONS.items():
        for c in spec["candidates"]:
            run = run_of(c, decisions)
            rows.append(dict(decision=key, option=spec["value"][c], run=run, rule=spec["rule"],
                             val_score=history_score(run, spec["rule"]), chosen=decisions.get(key) == spec["value"][c]))
    pd.DataFrame(rows).to_csv(os.path.join(res, "v4_decisions.csv"), index=False)
    unc = [dict(run=n, **r["unc"]) for n, r in results.items() if r.get("unc")]
    if unc:
        pd.DataFrame(unc).to_csv(os.path.join(res, "v4_uncertainty.csv"), index=False)

    # gallery for the frontend
    X = np.load(caches["test"]["x"], mmap_mode="r")
    Y = np.load(caches["test"]["y"], mmap_mode="r")
    fr = frames["test"].set_index("cache_idx")
    keys = [k for k in state["gallery"] if k in fr.index]
    npz = {"x": np.stack([np.asarray(X[k], np.float16) for k in keys]),
           "y": np.stack([np.asarray(Y[k], np.float16) for k in keys])}
    for n in done:
        p = os.path.join(jobs_dir(), n, "samples.npz")
        if n == "identity" or not os.path.exists(p):
            continue
        z = dict(np.load(p))
        if all(f"rgb_{k}" in z for k in keys):
            npz[f"pred_{n}"] = np.stack([z[f"rgb_{k}"] for k in keys])
        if all(f"gate_{k}" in z for k in keys):
            npz[f"gate_{n}"] = np.stack([z[f"gate_{k}"] for k in keys])
        if all(f"sigma_{k}" in z for k in keys):
            npz[f"sigma_{n}"] = np.stack([z[f"sigma_{k}"] for k in keys])
    meta = []
    for k in keys:
        r = fr.loc[k]
        sc = {n: {m: float(v) for m, v in pp[(pp.run == n) & (pp.cache_idx == k)][metrics].iloc[0].items()}
              for n in done if len(pp[(pp.run == n) & (pp.cache_idx == k)])}
        meta.append(dict(cache_idx=int(k), season=r.season, scene=int(r.scene), patch=int(r.patch), country=r.country,
                         continent=r.continent, cloud_frac=float(r.cloud_frac), scores=sc))
    np.savez_compressed(os.path.join(res, "v4_samples.npz"), **npz,
                        meta=np.array(json.dumps(dict(samples=meta, runs=done, labels=labels))))

    # figures
    try:
        hist = {}
        for n in done:
            hp = os.path.join(out, "logs", f"history_{n}.csv")
            if os.path.exists(hp):
                hist[n] = pd.read_csv(hp)
        plots.training_curves(hist, labels, os.path.join(res, "v4_training_curves.png"))
    except Exception as e:
        log(f"[plot] {e}")
    log("=" * 90)
    log("TEST RESULTS (25 held-out scenes, all continents)")
    show = summ[["label", "psnr", "psnr_aligned", "ssim", "sam", "sharpness", "colour_err", "detail_err", "fid"]]
    log("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
