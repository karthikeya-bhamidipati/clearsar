"""
One training loop for every run (docs/project_plan.md §4 "fixed protocol"): only the
RunConfig changes between Model A / B / C and the ablations.

    AdamW (wd 0.01; beta1 = 0.5 for GAN runs), linear warmup -> cosine decay
    AMP (autocast + one GradScaler per network), gradient clipping
    validation every `val_every` steps on full 256x256 val patches (tiled)
    best checkpoint = lowest validation L1;  resumable from last_<name>.pt

Plan v4 additions (all optional, defaults keep the Phase-1 behaviour):
    loss="combo"       w_l1 * L1 + w_ssim * (1 - SSIM) + w_perc * perceptual  (+ adv_weight * adversarial)
    ema_decay > 0      keep an exponential moving average of the weights; validate and save the EMA model
    select="balanced"  best checkpoint = lowest  val_L1 * (1 + |1 - val_sharpness|)  - an L1-only choice
                       favours blurry models (Phase 1: the GAN's best L1 checkpoint was at step 1 000)
    eval_tile          tile size for validation (128 = Phase 1; 256 = whole patches)
"""
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.evaluate import evaluate
from src.models.architectures import build_model, count_parameters, set_output_bias
from src.models.discriminator import PatchGANDiscriminator
from src.training.loss import PerceptualLoss, SSIMLoss, UncertaintyL1Loss


@dataclass
class RunConfig:
    name: str                       # e.g. "A_unet"
    arch: str                       # "unet" | "swin_unet" | "vit"
    label: str = ""                 # human-readable name for tables / the app
    model_kwargs: dict = field(default_factory=dict)
    uncertainty: bool = False       # Laplace variance head + NLL loss (E7)
    use_gan: bool = False           # PatchGAN + lambda * L1 (E4)
    gan_lambda: float = 100.0
    lr: float = 3e-4
    weight_decay: float = 0.01
    steps: int = 15000
    warmup: int = 500
    val_every: int = 1000
    log_every: int = 200
    grad_clip: float = 1.0
    seed: int = 42
    loss: str = "l1"                # "l1" | "l2" (E3b) | "combo" (plan v4)
    deadline: float = 0.0           # unix time; training stops early (keeping the best checkpoint) after it
    w_l1: float = 1.0               # combo loss weights
    w_ssim: float = 0.0
    w_perc: float = 0.0
    adv_weight: float = 0.0         # combo + use_gan: weight of the adversarial term
    ema_decay: float = 0.0          # e.g. 0.999
    select: str = "l1"              # "l1" | "balanced"
    eval_tile: int = 128


def _lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return f


def _forever(loader):
    while True:
        for batch in loader:
            yield batch


def _model_meta(cfg, in_ch, out_ch):
    return dict(arch=cfg.arch, model_kwargs=cfg.model_kwargs, in_channels=in_ch,
                out_channels=out_ch, uncertainty=cfg.uncertainty, run=asdict(cfg))


def train_run(cfg, train_loader, val_loader, device, ckpt_dir, log_dir, in_ch=15, out_ch=13,
              resume_dir=None, log=print):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    resume_dir = resume_dir or ckpt_dir
    os.makedirs(resume_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, f"best_{cfg.name}.pt")
    last_path = os.path.join(resume_dir, f"last_{cfg.name}.pt")
    hist_path = os.path.join(log_dir, f"history_{cfg.name}.csv")
    amp = device.type == "cuda"

    torch.manual_seed(cfg.seed)
    G = build_model(cfg.arch, in_ch, out_ch, cfg.uncertainty, **cfg.model_kwargs).to(device)
    n_params = count_parameters(G)
    betas = (0.5, 0.999) if cfg.use_gan else (0.9, 0.999)
    optG = torch.optim.AdamW(G.parameters(), lr=cfg.lr, betas=betas, weight_decay=cfg.weight_decay)
    schedG = torch.optim.lr_scheduler.LambdaLR(optG, _lr_lambda(cfg.warmup, cfg.steps))
    scalerG = torch.amp.GradScaler("cuda", enabled=amp)
    D = optD = schedD = scalerD = None
    if cfg.use_gan:
        D = PatchGANDiscriminator(in_channels=in_ch + out_ch).to(device)
        optD = torch.optim.AdamW(D.parameters(), lr=cfg.lr, betas=betas, weight_decay=cfg.weight_decay)
        schedD = torch.optim.lr_scheduler.LambdaLR(optD, _lr_lambda(cfg.warmup, cfg.steps))
        scalerD = torch.amp.GradScaler("cuda", enabled=amp)
    recon = UncertaintyL1Loss(out_ch) if cfg.uncertainty else (nn.MSELoss() if cfg.loss == "l2" else nn.L1Loss())
    bce = nn.BCEWithLogitsLoss()
    ssim_loss = SSIMLoss() if cfg.loss == "combo" and cfg.w_ssim > 0 else None
    perc_loss = PerceptualLoss().to(device) if cfg.loss == "combo" and cfg.w_perc > 0 else None
    ema = None
    if cfg.ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(G, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay), use_buffers=True)

    def selection_score(row):
        if cfg.select == "balanced":
            return row["val_l1"] * (1.0 + abs(1.0 - row.get("val_sharpness", 1.0)))
        return row["val_l1"]

    step, best, history, elapsed = 0, dict(val_l1=float("inf"), score=float("inf")), [], 0.0
    if os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("finished"):
            log(f"[{cfg.name}] already finished - skipping training (delete {last_path} to retrain)")
            return state["info"]
        G.load_state_dict(state["model"])
        optG.load_state_dict(state["optG"])
        schedG.load_state_dict(state["schedG"])
        scalerG.load_state_dict(state["scalerG"])
        if D is not None:
            D.load_state_dict(state["D"])
            optD.load_state_dict(state["optD"])
            schedD.load_state_dict(state["schedD"])
            scalerD.load_state_dict(state["scalerD"])
        if ema is not None and "ema" in state:
            ema.load_state_dict(state["ema"])
        step, best, history, elapsed = state["step"], state["best"], state["history"], state["elapsed"]
        best.setdefault("score", best.get("val_l1", float("inf")))
        log(f"[{cfg.name}] resumed at step {step}")
    else:
        # Start every model predicting the mean reflectance of each band (same for all runs).
        sums, n = torch.zeros(out_ch), 0
        for i, (_, yb, _) in enumerate(train_loader):
            sums += yb.float().mean(dim=(2, 3)).sum(0)
            n += yb.shape[0]
            if i >= 9:
                break
        band_means = sums / max(1, n)
        set_output_bias(G, band_means)
        if ema is not None:
            ema.module.load_state_dict(G.state_dict())
        log(f"[{cfg.name}] output bias initialised to mean band reflectance "
            f"(range {band_means.min():.3f}-{band_means.max():.3f})")

    log(f"[{cfg.name}] {cfg.label or cfg.arch}: {n_params / 1e6:.2f}M params, {cfg.steps} steps, "
        f"lr {cfg.lr}, gan={cfg.use_gan}, uncertainty={cfg.uncertainty}, loss={cfg.loss}"
        + (f" (l1 {cfg.w_l1}, ssim {cfg.w_ssim}, perc {cfg.w_perc}, adv {cfg.adv_weight})" if cfg.loss == "combo" else "")
        + (f", ema {cfg.ema_decay}" if ema is not None else "") + f", select={cfg.select}")
    if amp:
        torch.cuda.reset_peak_memory_stats(device)

    G.train()
    if D is not None:
        D.train()
    it = _forever(train_loader)
    run = dict(loss=0.0, adv=0.0, d=0.0, gnorm=0.0, n=0)
    t0 = time.time()
    t_log = time.time()
    stopped_early = False
    while step < cfg.steps:
        if cfg.deadline and time.time() > cfg.deadline and step > 0:
            stopped_early = True
            log(f"[{cfg.name}] time budget reached at step {step} - stopping early")
        x, y, _ = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            out = G(x)
        pred = out[:, :out_ch]

        if D is not None:
            optD.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                d_real = D(x, y)
                d_fake = D(x, pred.detach())
                loss_D = 0.5 * (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake)))
            scalerD.scale(loss_D).backward()
            scalerD.step(optD)
            scalerD.update()
            schedD.step()
            run["d"] += loss_D.item()

        optG.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss_rec = recon(out, y) if cfg.uncertainty else recon(pred.float(), y)
            if cfg.loss == "combo":
                loss_rec = cfg.w_l1 * loss_rec
                if ssim_loss is not None:
                    loss_rec = loss_rec + cfg.w_ssim * ssim_loss(pred, y)
                if perc_loss is not None:
                    loss_rec = loss_rec + cfg.w_perc * perc_loss(pred, y)
            if D is not None:
                d_out = D(x, pred)
                loss_adv = bce(d_out, torch.ones_like(d_out))
                loss = (loss_rec + cfg.adv_weight * loss_adv) if cfg.loss == "combo" else (loss_adv + cfg.gan_lambda * loss_rec)
            else:
                loss = loss_rec
        scalerG.scale(loss).backward()
        scalerG.unscale_(optG)
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
        scalerG.step(optG)
        scalerG.update()
        schedG.step()
        if ema is not None:
            ema.update_parameters(G)
        step += 1

        run["loss"] += loss_rec.item()
        run["adv"] += loss_adv.item() if D is not None else 0.0
        run["gnorm"] += float(gnorm) if torch.isfinite(gnorm) else 0.0
        run["n"] += 1

        if step % cfg.log_every == 0:
            n = max(1, run["n"])
            ips = cfg.log_every / max(1e-6, time.time() - t_log)
            msg = (f"[{cfg.name}] step {step:6d}/{cfg.steps} | rec {run['loss'] / n:.4f} "
                   f"| gnorm {run['gnorm'] / n:.2f} | lr {schedG.get_last_lr()[0]:.2e} | {ips:.1f} it/s")
            if D is not None:
                msg += f" | adv {run['adv'] / n:.3f} | D {run['d'] / n:.3f}"
            log(msg)
            t_log = time.time()

        if step % cfg.val_every == 0 or step == cfg.steps or stopped_early:
            n = max(1, run["n"])
            evm = ema.module if ema is not None else G
            res, _ = evaluate(evm, val_loader, device, out_channels=out_ch, uncertainty=cfg.uncertainty,
                              tile=cfg.eval_tile)
            G.train()
            row = dict(step=step, train_rec=run["loss"] / n, train_adv=run["adv"] / n, train_d=run["d"] / n,
                       grad_norm=run["gnorm"] / n, lr=schedG.get_last_lr()[0],
                       val_l1=float(res["mae"].mean()), val_psnr=float(res["psnr"].mean()),
                       val_ssim=float(res["ssim"].mean()), val_sam=float(res["sam"].mean()),
                       val_sharpness=float(res["sharpness"].mean()), val_colour=float(res["colour_err"].mean()),
                       val_detail=float(res["detail_err"].mean()), minutes=(elapsed + time.time() - t0) / 60)
            row["score"] = selection_score(row)
            history.append(row)
            pd.DataFrame(history).to_csv(hist_path, index=False)
            improved = row["score"] < best["score"]
            if improved:
                best = dict(step=step, score=row["score"], val_l1=row["val_l1"], val_psnr=row["val_psnr"],
                            val_ssim=row["val_ssim"], val_sharpness=row["val_sharpness"],
                            val_colour=row["val_colour"], val_detail=row["val_detail"])
                torch.save(dict(model=evm.state_dict(), step=step, val=best, **_model_meta(cfg, in_ch, out_ch)),
                           best_path)
            log(f"[{cfg.name}] VAL step {step}: L1 {row['val_l1']:.4f} | PSNR {row['val_psnr']:.2f} dB "
                f"| SSIM {row['val_ssim']:.4f} | SAM {row['val_sam']:.2f} deg | sharp {row['val_sharpness']:.2f} "
                f"| colour {row['val_colour']:.4f} | detail {row['val_detail']:.4f}" + ("  * best" if improved else ""))
            run = dict(loss=0.0, adv=0.0, d=0.0, gnorm=0.0, n=0)
            state = dict(model=G.state_dict(), optG=optG.state_dict(), schedG=schedG.state_dict(),
                         scalerG=scalerG.state_dict(), step=step, best=best, history=history,
                         elapsed=elapsed + time.time() - t0)
            if D is not None:
                state.update(D=D.state_dict(), optD=optD.state_dict(), schedD=schedD.state_dict(),
                             scalerD=scalerD.state_dict())
            if ema is not None:
                state["ema"] = ema.state_dict()
            torch.save(state, last_path)
            if stopped_early:
                break

    info = dict(name=cfg.name, label=cfg.label, arch=cfg.arch, params=n_params, steps=cfg.steps,
                best_step=best.get("step"), best_val_l1=best["val_l1"], best_val_psnr=best.get("val_psnr"),
                best_val_ssim=best.get("val_ssim"), best_val_sharpness=best.get("val_sharpness"),
                best_val_colour=best.get("val_colour"), best_val_detail=best.get("val_detail"),
                best_score=best.get("score"), steps_done=step, stopped_early=stopped_early,
                train_minutes=(elapsed + time.time() - t0) / 60,
                peak_gpu_gb=(torch.cuda.max_memory_allocated(device) / 1e9) if amp else 0.0)
    # Replace the large resume state with a small "finished" marker.
    torch.save(dict(finished=True, info=info), last_path)
    with open(os.path.join(log_dir, f"run_{cfg.name}.json"), "w") as f:
        json.dump(dict(info=info, config=asdict(cfg)), f, indent=2)
    log(f"[{cfg.name}] done in {info['train_minutes']:.1f} min, best val L1 {best['val_l1']:.4f} "
        f"at step {best.get('step')}, peak GPU {info['peak_gpu_gb']:.2f} GB")
    return info


# --------------------------------------------------------------------------------------
# Plan v4, stage 1: two diagnostics every practitioner should run before a long training
# --------------------------------------------------------------------------------------
def overfit_one_batch(arch, model_kwargs, batch, device, steps=300, lr=1e-3, in_ch=15, out_ch=13):
    """Train on ONE fixed batch until it is memorised. If the loss does not go towards zero, the
    pipeline is broken (wrong targets, a bug in the model, a learning rate that cannot work).
    Returns a list of (step, L1)."""
    torch.manual_seed(0)
    G = build_model(arch, in_ch, out_ch, **model_kwargs).to(device).train()
    opt = torch.optim.AdamW(G.parameters(), lr=lr)
    x, y = batch[0].to(device), batch[1].to(device)
    out = []
    for s in range(1, steps + 1):
        loss = nn.functional.l1_loss(G(x)[:, :out_ch].float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s == 1 or s % 10 == 0:
            out.append((s, loss.item()))
    return out


def lr_range_test(arch, model_kwargs, loader, device, steps=300, lr_min=1e-6, lr_max=1e-1, in_ch=15, out_ch=13):
    """Leslie Smith's LR range test: raise the learning rate exponentially from lr_min to lr_max over
    `steps` steps and record the loss. The loss falls while the LR is useful and explodes when it is too
    large; a good LR is about 10x below the point where it starts to explode.
    Returns a list of (lr, loss, smoothed_loss) and the suggested LR."""
    torch.manual_seed(0)
    G = build_model(arch, in_ch, out_ch, **model_kwargs).to(device).train()
    opt = torch.optim.AdamW(G.parameters(), lr=lr_min)
    gamma = (lr_max / lr_min) ** (1 / steps)
    it, out, smooth, best = _forever(loader), [], None, float("inf")
    for s in range(steps):
        lr = lr_min * gamma ** s
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, _ = next(it)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = nn.functional.l1_loss(G(x)[:, :out_ch].float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        v = float(loss.detach())
        smooth = v if smooth is None else 0.95 * smooth + 0.05 * v
        out.append((lr, v, smooth))
        best = min(best, smooth)
        if not math.isfinite(v) or smooth > 4 * best:
            break
    sm = [o[2] for o in out]
    blow = next((i for i in range(len(sm)) if sm[i] > 2 * min(sm[: i + 1])), len(sm) - 1)
    return out, out[blow][0] / 10
