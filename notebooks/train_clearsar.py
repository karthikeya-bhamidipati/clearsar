"""
ClearSAR - the final model. ONE ClearSAR-Net generator + PatchGAN, trained on the WHOLE SEN12MS-CR dataset.
=========================================================================================================

HOW TO RUN ON KAGGLE  (Notebook settings: Accelerator "GPU T4 x2", Internet ON, the SEN12MS-CR dataset attached)
    !rm -rf /kaggle/working/*
    !git clone -q https://github.com/karthikeya-bhamidipati/clearsar.git /kaggle/working/clearsar
    !CLEARSAR_QUICK=1 python /kaggle/working/clearsar/notebooks/train_clearsar.py      # ~15 min smoke test
  If the smoke test ends with "DONE", start the real run with "Save Version -> Save & Run All (Commit)" and
  this cell (no CLEARSAR_QUICK):
    !python /kaggle/working/clearsar/notebooks/train_clearsar.py                      # ~11.6 h
  Download clearsar_final_output.zip (Output tab) and extract it into <project>/outputs/final/
  (so the app finds outputs/final/results/test_summary.csv).

WHAT IT DOES
  data     every complete triplet (122 218 patches, 175 scenes, all continents and seasons), split BY SCENE:
           ~147 train / 10 val / 18 test scenes. The 11 Phase-1 test scenes stay in the test set.
           Training uses every patch of every training scene (no caps); validation 600 patches (10 scenes),
           test up to 300 patches per test scene. Patches are read directly from the GeoTIFFs.
  model    ClearSAR-Net (src/models/clearsar_net.py): U-Net, 3 levels, width 64, 2 global self-attention
           blocks in the bottleneck, bilinear upsampling, "hybrid" output (learned cloud gate: keep and
           correct what the satellite saw, generate the rest). 12.8 M parameters.
  part A   reconstruction training, L1 + 0.05 (1 - SSIM), 55 % of the time      -> clearsar_net_before_gan.pt
  part B   + spectrally normalised PatchGAN (LSGAN), adaptive adversarial weight -> clearsar_net_final.pt
           (src/training/final_trainer.py explains every step)
  test     ONE evaluation on the held-out scenes: cloudy input, Phase-1 U-Net, before-GAN, final model;
           by cloud cover, by continent, by country; uncertainty from test-time augmentation; gallery.

Environment variables (optional): CLEARSAR_QUICK=1, CLEARSAR_HOURS (whole session, default 11.6),
CLEARSAR_INPUT_ROOTS, CLEARSAR_OUT_DIR, CLEARSAR_WORKERS, CLEARSAR_BATCHES (e.g. "8,6,4"), CLEARSAR_ADV_FACTOR,
CLEARSAR_GAN_START, CLEARSAR_TRAIN_MIN (fixed training minutes, overrides the automatic budget),
CLEARSAR_CAPS (e.g. "train=400,val=48,test=96"), CLEARSAR_N_VAL, CLEARSAR_N_TEST.
"""
import json
import os
import shutil
import subprocess
import sys
import time

T_START = time.time()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less memory fragmentation
os.environ.setdefault("NCCL_P2P_DISABLE", "1")      # the two T4s talk through host memory: avoids P2P hangs in VMs
REPO_URL = "https://github.com/karthikeya-bhamidipati/clearsar.git"
ON_KAGGLE = os.path.isdir("/kaggle/input")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
if not os.path.isdir(os.path.join(REPO, "src")):
    REPO = "/kaggle/working/clearsar"
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _env(name, default, cast=str):
    v = os.environ.get(f"CLEARSAR_{name}")
    return default if v in (None, "") else cast(v)


def _caps(s):
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))} if s else {}


QUICK = _env("QUICK", 0, int) == 1
CFG = dict(
    input_roots=_env("INPUT_ROOTS", "/kaggle/input/*").split(";"),
    out_dir=_env("OUT_DIR", "/kaggle/working/clearsar_final_output" if ON_KAGGLE else os.path.join(REPO, "outputs", "final_run")),
    index_cache=_env("INDEX_CACHE", "/kaggle/working/clearsar_index.csv" if ON_KAGGLE else ""),
    hours=_env("HOURS", 0.3 if QUICK else 11.6, float),          # whole session, setup to zip
    train_minutes=_env("TRAIN_MIN", 0.0, float),                 # 0 = automatic (fill the session)
    caps=_caps(_env("CAPS", "train=400,val=48,test=120" if QUICK else "val=600")),
    test_per_scene=_env("TEST_PER_SCENE", 300, int),
    n_val=_env("N_VAL", 10, int), n_test=_env("N_TEST", 18, int),
    seed=42,
    # ---- model
    model=dict(output_mode="hybrid", attn_layers=2, attn_heads=8, attn_mlp_ratio=2.0, width=_env("WIDTH", 64, int),
               upsample="bilinear"),
    d_base=64,
    # ---- optimisation
    batch_candidates=[int(b) for b in _env("BATCHES", "8,6,4").split(",")],
    lr=3e-4, lr_floor=0.05, warmup_steps=1000 if not QUICK else 20, weight_decay=1e-4, grad_clip=1.0,
    ema_decay=0.999 if not QUICK else 0.99,
    w_ssim=0.05,
    # ---- GAN (part B)
    gan_start=_env("GAN_START", 0.55, float),
    lr_gan=5e-5, lr_d=2e-4,
    adv_factor=_env("ADV_FACTOR", 0.2, float),
    d_warmup_steps=_env("D_WARMUP", 10 if QUICK else 300, int), adv_ramp_steps=_env("ADV_RAMP", 20 if QUICK else 1000, int),
    lambda_every=10,
    # ---- data feeding / bookkeeping
    workers=_env("WORKERS", 3, int), pool_size=128, max_echo=4,
    val_every_min=_env("VAL_EVERY_MIN", 2.0 if QUICK else 20.0, float),
    log_every=20 if QUICK else 200, sync_every=10,
    p1_checkpoint=os.path.join(REPO, "outputs", "checkpoints", "best_A_unet.pt"),
)
CLOUD_BINS = [0, 0.1, 0.3, 0.6, 0.9, 1.01]
CLOUD_LABELS = ["0-10%", "10-30%", "30-60%", "60-90%", "90-100%"]
LABELS = {"identity": "Cloudy input (no model)", "p1_unet": "Phase 1 U-Net (3 000 Asian patches)",
          "before_gan": "ClearSAR-Net, before GAN (part A)", "final": "ClearSAR-Net final (with GAN)"}


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        line = time.strftime("%H:%M:%S ") + " ".join(str(m) for m in msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def setup_kaggle():
    if not ON_KAGGLE:
        return
    try:
        import rasterio  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=True)


# --------------------------------------------------------------------------------------
# Test evaluation (main process, after training)
# --------------------------------------------------------------------------------------
def cloud_fraction_t(x, y, threshold=0.08):
    vis = [1, 2, 3]
    d = x[:, 2:][:, vis].mean(1) - y[:, vis].mean(1)
    return (d > threshold).float().mean(dim=(1, 2))


def tta_predict(model, x):
    """Prediction + per-pixel spread over 4 flips (test-time augmentation): where the model is unsure, the
    four flipped versions disagree. Returns (pred of the unflipped input, gate, sigma map [B, H, W])."""
    import torch
    preds = []
    pred, aux = model(x, return_aux=True)
    preds.append(pred.float())
    for dims in ((-1,), (-2,), (-2, -1)):
        preds.append(model(x.flip(dims)).float().flip(dims))
    s = torch.stack(preds).std(0).mean(1)
    gate = aux.get("gate")
    return pred.float(), None if gate is None else gate[:, 0].float(), s


def evaluate_test(cfg, test_df, models, device, log):
    """models: {run: (model, tile)}; tile None = full patch. Returns a long per-patch dataframe."""
    import numpy as np
    import pandas as pd
    import torch
    from src.evaluation.evaluate import predict_tiled
    from src.evaluation.metrics import compute_metrics
    from src.training.final_trainer import make_loader

    records = test_df[["s1", "s2", "s2_cloudy"]].to_dict("records")
    loader = make_loader(records, 8, cfg["workers"] + 1)
    rows, t0, n = [], time.time(), 0
    with torch.no_grad():
        for b in loader:
            if b is None:
                continue
            x, y, idx = b[0].to(device).float(), b[1].to(device).float(), b[2].numpy()
            cf = cloud_fraction_t(x, y).cpu().numpy()
            extra = {}
            for run, (model, tile) in models.items():
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    if model is None:
                        pred = x[:, 2:]
                    elif run == "final":
                        pred, gate, sig = tta_predict(model, x)
                        extra = dict(gate_mean=gate.mean(dim=(1, 2)).cpu().numpy(), tta_std=sig.mean(dim=(1, 2)).cpu().numpy())
                    elif tile:
                        pred = predict_tiled(model, x, tile)[:, :13]
                    else:
                        pred = model(x)
                m = compute_metrics(pred.float()[:, :13], y)
                for j, i in enumerate(idx):
                    r = dict(row=int(i), run=run, cloud_frac=float(cf[j]), **{k: float(v[j]) for k, v in m.items()})
                    if run == "final":
                        r.update({k: float(v[j]) for k, v in extra.items()})
                    rows.append(r)
            n += len(idx)
            if n % 400 < len(idx):
                log(f"[test] {n}/{len(records)} patches ({n / (time.time() - t0):.1f} patch/s)")
    df = pd.DataFrame(rows)
    meta = test_df.reset_index(drop=True)[["season", "scene", "patch", "country", "continent"]]
    return df.merge(meta, left_on="row", right_index=True, how="left")


def pick_gallery(per, n_per_bin=2):
    """Representative (NOT best) test patches: per cloud bin, the patches whose final PSNR is closest to the
    bin median, from different continents; plus one Indian patch if India is in the test set."""
    f = per[per.run == "final"].copy()
    f["bin"] = __import__("pandas").cut(f.cloud_frac, CLOUD_BINS, labels=CLOUD_LABELS, right=False)
    picks = []
    for _, g in f.groupby("bin", observed=True):
        g = g.assign(dist=(g.psnr - g.psnr.median()).abs()).sort_values("dist")
        conts = set()
        for r in g.itertuples():
            if r.continent not in conts:
                picks.append(r.row)
                conts.add(r.continent)
            if len(conts) >= n_per_bin:
                break
    ind = f[f.country == "IN"]
    if len(ind) and not set(ind.row) & set(picks):
        ind = ind.assign(dist=(ind.psnr - ind.psnr.median()).abs()).sort_values("dist")
        picks.append(int(ind.iloc[0].row))
    return picks


def export_gallery(cfg, test_df, per, models, device, out, log):
    import numpy as np
    import torch
    from src.data.dataset import RGB_IDX, read_triplet
    from src.evaluation.evaluate import predict_tiled

    rows = pick_gallery(per)
    xs, ys, rgb, gates, sigmas, bands, meta = [], [], {r: [] for r in models if models[r][0] is not None}, [], [], [], []
    for i in rows:
        rec = test_df.iloc[i]
        x, y = read_triplet(rec[["s1", "s2", "s2_cloudy"]].to_dict())
        xt = torch.from_numpy(x)[None].to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            for run, (model, tile) in models.items():
                if model is None:
                    continue
                if run == "final":
                    p, g, s = tta_predict(model, xt)
                    gates.append(g[0].cpu().numpy().astype(np.float16))
                    sigmas.append(s[0].cpu().numpy().astype(np.float16))
                    bands.append(p[0].cpu().numpy().astype(np.float16))
                else:
                    p = predict_tiled(model, xt, tile)[:, :13] if tile else model(xt)
                p = p[0].float().cpu().numpy()
                rgb[run].append((np.clip(np.stack([p[c] for c in RGB_IDX], -1) * 3.0, 0, 1) * 255).astype(np.uint8))
        xs.append(x.astype(np.float16))
        ys.append(y.astype(np.float16))
        sc = per[per.row == i].set_index("run")
        meta.append(dict(season=rec.season, scene=int(rec.scene), patch=int(rec.patch), country=rec.country,
                         continent=rec.continent, cloud_frac=float(sc.cloud_frac.iloc[0]),
                         scores={r: {k: float(sc.loc[r, k]) for k in ("psnr", "ssim", "sam", "mae", "sharpness", "psnr_aligned")}
                                 for r in sc.index}))
    npz = dict(x=np.stack(xs), y=np.stack(ys), gate_final=np.stack(gates), sigma_final=np.stack(sigmas),
               bands_final=np.stack(bands), **{f"pred_{r}": np.stack(v) for r, v in rgb.items()})
    np.savez_compressed(os.path.join(out, "results", "samples.npz"), **npz,
                        meta=np.array(json.dumps(dict(samples=meta, runs=list(models), labels=LABELS))))
    log(f"[gallery] {len(rows)} representative test patches -> results/samples.npz")


def plot_curves(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    tr = pd.read_csv(os.path.join(out, "logs", "train_history.csv"))
    va = pd.read_csv(os.path.join(out, "logs", "val_history.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(tr.time_h, tr.l1, label="train L1", lw=1)
    ax[0].plot(va.time_h, va.mae, "o-", label="val L1 (EMA)", ms=3)
    ax[1].plot(va.time_h, va.psnr, "o-", label="val PSNR", ms=3)
    ax[1].plot(va.time_h, va.psnr_aligned, "o-", label="val PSNR, date shift removed", ms=3)
    ax2 = ax[1].twinx()
    ax2.plot(va.time_h, va.sharpness, "s--", color="tab:red", ms=3, label="sharpness")
    ax2.set_ylabel("sharpness (1 = real)")
    if "d_loss" in tr:
        b = tr[tr.phase == "B"]
        ax[2].plot(b.time_h, b.d_real, label="D(real)")
        ax[2].plot(b.time_h, b.d_fake, label="D(fake)")
        ax[2].plot(b.time_h, b.d_loss, label="D loss")
    for a, t in zip(ax, ("Loss", "Validation quality", "Discriminator (part B)")):
        a.set_title(t)
        a.set_xlabel("hours")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "results", "training_curves.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    setup_kaggle()
    import numpy as np
    import pandas as pd
    import torch
    import torch.multiprocessing as mp
    from src.data.dataset import attach_scene_metadata, index_triplets, make_global_splits, read_triplet, _sample_evenly
    from src.models.architectures import load_model_from_checkpoint
    from src.training.final_trainer import make_loader, train_worker

    cfg = dict(CFG)
    out = cfg["out_dir"]
    for sub in ("checkpoints", "logs", "results"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    log = Logger(os.path.join(out, "logs", "pipeline_log.txt"))
    deadline = T_START + cfg["hours"] * 3600
    n_gpu = torch.cuda.device_count()
    log("=" * 100)
    log(f"ClearSAR final model | quick={QUICK} | repo={REPO} | GPUs: "
        + (", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu)) or "none (CPU)") + f" | torch {torch.__version__}")

    # ---- 1. index every triplet, attach country / continent, split by scene
    ip = cfg["index_cache"]
    if ip and os.path.exists(ip):
        tri = pd.read_csv(ip)
        log(f"[index] re-using {ip}: {len(tri)} triplets")
    else:
        tri = index_triplets(cfg["input_roots"], log=log)
        if ip:
            tri.to_csv(ip, index=False)
    tri = attach_scene_metadata(tri, os.path.join(REPO, "data", "scans", "scenes.csv"), log=log)
    tri = tri[tri.continent != "Unknown"]
    p1 = os.path.join(REPO, "outputs", "results", "scene_splits.csv")
    p1_scenes = pd.read_csv(p1) if os.path.exists(p1) else None
    caps = dict(cfg["caps"])
    patches, scenes = make_global_splits(tri, p1_scenes, n_val=cfg["n_val"], n_test=cfg["n_test"],
                                         caps={k: v for k, v in caps.items() if k != "test"}, seed=cfg["seed"], log=log)
    rng = np.random.default_rng(cfg["seed"])
    test_df = patches[patches.split == "test"]
    test_cap = caps.get("test", cfg["test_per_scene"] * test_df.groupby(["season", "scene"]).ngroups)
    test_df = _sample_evenly(test_df, test_cap, rng).sort_values(["season", "scene", "patch"]).reset_index(drop=True)
    train_df = patches[patches.split == "train"].reset_index(drop=True)
    val_df = patches[patches.split == "val"].reset_index(drop=True)
    log(f"[data] train {len(train_df)} patches ({train_df.groupby(['season', 'scene']).ngroups} scenes) | "
        f"val {len(val_df)} | test {len(test_df)} ({test_df.groupby(['season', 'scene']).ngroups} scenes)")
    scenes.to_csv(os.path.join(out, "results", "scene_splits.csv"), index=False)
    test_df[["season", "scene", "patch", "country", "continent"]].to_csv(os.path.join(out, "results", "test_patches.csv"), index=False)

    # ---- 2. band means (start the generator's output at the average clear image) + read-speed benchmark
    t0 = time.time()
    sample = train_df.sample(n=min(48, len(train_df)), random_state=0)[["s1", "s2", "s2_cloudy"]].to_dict("records")
    n_read, ys = 0, []
    for b in make_loader(sample, 8, cfg["workers"] + 1):
        if b is not None:
            ys.append(b[1].float().mean(dim=(0, 2, 3)) * len(b[1]))
            n_read += len(b[1])
    band_means = (torch.stack(ys).sum(0) / n_read).tolist()
    read_rate = n_read / (time.time() - t0)
    log(f"[bench] reading GeoTIFFs: {read_rate:.1f} patches/s (one process, {cfg['workers'] + 1} workers)")
    log(f"[bench] mean clear reflectance per band: {np.round(band_means, 3).tolist()}")

    # ---- 3. training budget: fill the session, keep time for the test pass and the export
    test_reserve = min(150 * 60, max(10 * 60, 1.5 * len(test_df) / max(read_rate, 1.0) + 8 * 60))
    if QUICK:
        test_reserve = min(test_reserve, 5 * 60)
    train_seconds = cfg["train_minutes"] * 60 if cfg["train_minutes"] else deadline - time.time() - test_reserve - 120
    cfg["train_seconds"] = max(60.0, train_seconds)
    log(f"[plan] training {cfg['train_seconds'] / 3600:.2f} h, reserve {test_reserve / 60:.0f} min for test + export")
    json.dump(dict(cfg=cfg, n_train=len(train_df), n_val=len(val_df), n_test=len(test_df), read_rate=read_rate,
                   band_means=band_means, labels=LABELS), open(os.path.join(out, "config.json"), "w"), indent=2)

    # ---- 4. train (one process per GPU)
    train_rec = train_df[["s1", "s2", "s2_cloudy"]].to_dict("records")
    val_rec = val_df[["s1", "s2", "s2_cloudy"]].to_dict("records")
    world = max(1, n_gpu)
    log(f"[train] starting {world} process(es)")
    if world > 1:
        mp.spawn(train_worker, args=(world, cfg, train_rec, val_rec, band_means), nprocs=world, join=True)
    else:
        train_worker(0, 1, cfg, train_rec, val_rec, band_means)

    # ---- 5. the one honest test
    device = torch.device("cuda:0" if n_gpu else "cpu")
    ck = os.path.join(out, "checkpoints")
    models = {"identity": (None, None)}
    if os.path.exists(cfg["p1_checkpoint"]):
        models["p1_unet"] = (load_model_from_checkpoint(cfg["p1_checkpoint"], map_location=str(device))[0], 128)
    for run, f in (("before_gan", "clearsar_net_before_gan.pt"), ("final", "clearsar_net_final.pt")):
        models[run] = (load_model_from_checkpoint(os.path.join(ck, f), map_location=str(device))[0], None)
    log(f"[test] evaluating {list(models)} on {len(test_df)} test patches")
    per = evaluate_test(cfg, test_df, models, device, log)
    res = os.path.join(out, "results")
    per.to_csv(os.path.join(res, "test_per_patch.csv"), index=False)
    metrics = ["psnr", "ssim", "sam", "mae", "rmse", "psnr_aligned", "sharpness", "colour_err", "detail_err"]
    summ = per.groupby("run")[metrics].mean().reindex(list(models)).reset_index()
    summ.insert(1, "label", summ.run.map(LABELS))
    summ.insert(2, "n", per.groupby("run").size().reindex(list(models)).values)
    summ.to_csv(os.path.join(res, "test_summary.csv"), index=False)
    per["cloud_bin"] = pd.cut(per.cloud_frac, CLOUD_BINS, labels=CLOUD_LABELS, right=False)
    per.groupby(["cloud_bin", "run"], observed=True)[metrics].mean().reset_index() \
        .merge(per[per.run == "final"].groupby("cloud_bin", observed=True).size().rename("n").reset_index()) \
        .to_csv(os.path.join(res, "test_by_cloud.csv"), index=False)
    per.groupby(["continent", "run"])[metrics].mean().reset_index() \
        .merge(per[per.run == "final"].groupby("continent").size().rename("n").reset_index()) \
        .to_csv(os.path.join(res, "test_by_continent.csv"), index=False)
    per.groupby(["country", "continent", "run"])[metrics].mean().reset_index() \
        .merge(per[per.run == "final"].groupby("country").size().rename("n").reset_index()) \
        .to_csv(os.path.join(res, "test_by_country.csv"), index=False)
    fin = per[per.run == "final"]
    unc = dict(spearman_tta_std_vs_mae=float(fin.tta_std.corr(fin.mae, method="spearman")),
               mean_gate=float(fin.gate_mean.mean()))
    json.dump(unc, open(os.path.join(res, "uncertainty.json"), "w"), indent=2)
    for r in summ.itertuples():
        log(f"[result] {r.label:40s} PSNR {r.psnr:6.2f}  SSIM {r.ssim:.3f}  SAM {r.sam:5.2f}  "
            f"aligned {r.psnr_aligned:6.2f}  sharpness {r.sharpness:.2f}")
    log(f"[result] uncertainty: Spearman(TTA spread, error) = {unc['spearman_tta_std_vs_mae']:.2f}")

    # ---- 6. gallery, curves, zip
    try:
        export_gallery(cfg, test_df, per, models, device, out, log)
    except Exception as e:                      # never lose the run because of a picture
        log(f"[gallery] WARNING failed: {e!r}")
    try:
        plot_curves(out)
    except Exception as e:
        log(f"[plot] WARNING failed: {e!r}")
    for f in ("clearsar_net_last.pt",):         # keep the zip small: final + before-GAN weights only
        p = os.path.join(ck, f)
        if os.path.exists(p) and not QUICK:
            os.remove(p)
    log(f"DONE in {(time.time() - T_START) / 3600:.2f} h - download {os.path.basename(out)}.zip")
    shutil.make_archive(out, "zip", out)


if __name__ == "__main__":
    main()
