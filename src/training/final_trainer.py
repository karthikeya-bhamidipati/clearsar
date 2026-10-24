"""
Training engine of the final model: ClearSAR-Net + PatchGAN on the whole SEN12MS-CR dataset, both GPUs.

    notebooks/train_clearsar.py   builds the data split and calls train_worker() once per GPU

What happens, in order (all times are fractions of the training budget, so the run always fits):

  part A  0 .. gan_start      reconstruction only:  loss = L1 + w_ssim * (1 - SSIM)
                              AdamW, warm-up then cosine decay. Best EMA weights by validation L1
                              -> checkpoints/clearsar_net_before_gan.pt
  part B  gan_start .. 1      the PatchGAN discriminator joins:
                                D:  LSGAN   0.5 * [(D(x, y) - 1)^2 + D(x, G(x))^2]
                                G:  L1 + w_ssim * (1 - SSIM) + lambda * 0.5 * (D(x, G(x)) - 1)^2
                              lambda is ADAPTIVE (Esser et al. 2021, VQGAN): it is set so that the adversarial
                              gradient at the generator's last shared layer is adv_factor (e.g. 0.2) times the
                              reconstruction gradient - "the GAN may push 20 % as hard as the pixel loss",
                              whatever the raw loss scales are. It ramps in after a short D-only warm-up.
                              Best EMA weights by the balanced score L1 * (1 + |1 - sharpness|), counted only
                              once the GAN is fully on -> checkpoints/clearsar_net_final.pt

Efficiency
  * DistributedDataParallel over both T4s (one process per GPU), mixed precision (fp16 + GradScaler)
  * patches are read straight from the GeoTIFFs (no 450 GB cache) by DataLoader workers in a background
    thread. When reading is slower than the GPU, recent patches are re-used with a new random flip/rotation
    instead of letting the GPU wait ("data echoing", Choi et al. 2019); at least 1 in (max_echo + 1)
    batches is always fresh. The log shows the fresh fraction.
  * EMA of the generator weights (decay 0.999) is what gets validated and saved.
"""
import json
import math
import os
import queue
import threading
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import IN_CHANNELS, N_S1, N_S2, OUT_CHANNELS, PATCH_SIZE, preprocess_s1, preprocess_s2
from src.evaluation.metrics import compute_metrics, ssim_per_image


# --------------------------------------------------------------------------------------
# Data: read triplets straight from the GeoTIFFs
# --------------------------------------------------------------------------------------
_POOL = None


def _read_bands(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read()


def read_triplet_parallel(rec):
    """read_triplet() with the three GeoTIFFs of a patch read at the same time by three threads. Reading on
    Kaggle is limited by storage latency, not CPU (quick test: ~15 patches/s cold), so overlapping the waits
    is what makes it faster. Each thread opens its own file handle (GDAL is thread-safe that way)."""
    global _POOL
    if _POOL is None:                                  # one small pool per DataLoader worker process
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=3)
    s1, s2c, s2 = _POOL.map(_read_bands, (rec["s1"], rec["s2_cloudy"], rec["s2"]))
    if s1.shape[0] < N_S1 or s2c.shape[0] < N_S2 or s2.shape[0] < N_S2:
        raise ValueError(f"Unexpected band counts S1={s1.shape[0]} S2c={s2c.shape[0]} S2={s2.shape[0]}")
    x = np.concatenate([preprocess_s1(s1[:N_S1]), preprocess_s2(s2c[:N_S2])], axis=0)
    return x, preprocess_s2(s2[:N_S2])


class TripletStream(Dataset):
    """records: list of dicts with s1 / s2 / s2_cloudy paths. Returns (x fp16 [15,256,256], y fp16 [13,256,256],
    record index, ok). Unreadable or mis-shaped files return ok=False and are dropped by the collate."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        try:
            x, y = read_triplet_parallel(self.records[i])
            if x.shape != (IN_CHANNELS, PATCH_SIZE, PATCH_SIZE) or y.shape != (OUT_CHANNELS, PATCH_SIZE, PATCH_SIZE):
                raise ValueError(f"shape {x.shape} / {y.shape}")
            return torch.from_numpy(x.astype(np.float16)), torch.from_numpy(y.astype(np.float16)), i, True
        except Exception:
            return None, None, i, False


def stream_collate(batch):
    ok = [b for b in batch if b[3]]
    if not ok:
        return None
    return torch.stack([b[0] for b in ok]), torch.stack([b[1] for b in ok]), torch.tensor([b[2] for b in ok])


def make_loader(records, batch, workers, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(TripletStream(records), batch_size=batch, shuffle=shuffle if sampler is None else False,
                      sampler=sampler, num_workers=workers, collate_fn=stream_collate, pin_memory=torch.cuda.is_available(),
                      drop_last=drop_last, persistent_workers=workers > 0 and sampler is not None,
                      prefetch_factor=4 if workers > 0 else None)


def dihedral(x, y):
    """Random flip / 90-degree rotation per sample (8 possibilities), applied on the GPU."""
    xs, ys = [], []
    for a, b in zip(x, y):
        k = int(torch.randint(0, 8, (1,)))
        if k & 4:
            a, b = a.flip(-1), b.flip(-1)
        if k & 3:
            a, b = torch.rot90(a, k & 3, (-2, -1)), torch.rot90(b, k & 3, (-2, -1))
        xs.append(a)
        ys.append(b)
    return torch.stack(xs), torch.stack(ys)


class EchoFeeder:
    """Background thread pulls batches from the DataLoader (endless epochs) into a small queue. next()
    returns a fresh batch if one is ready; otherwise a batch re-drawn from a pool of recent patches
    (with a new random dihedral transform), at most max_echo times in a row; then it waits for fresh data."""

    def __init__(self, loader, sampler, device, pool_size=128, max_echo=4):
        self.loader, self.sampler, self.device = loader, sampler, device
        self.q = queue.Queue(maxsize=3)
        self.stop = threading.Event()
        self.pool_x = self.pool_y = None
        self.pool_n, self.pool_ptr, self.pool_size = 0, 0, pool_size
        self.max_echo, self.echo_run = max_echo, 0
        self.n_fresh = self.n_echo = 0
        self.epoch = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            for b in self.loader:
                if self.stop.is_set():
                    return
                while b is not None and not self.stop.is_set():
                    try:
                        self.q.put(b, timeout=1.0)
                        break
                    except queue.Full:
                        continue
            self.epoch += 1

    def _add_to_pool(self, x, y):
        if self.pool_x is None:
            self.pool_x = torch.empty((self.pool_size,) + tuple(x.shape[1:]), dtype=torch.float16, device=self.device)
            self.pool_y = torch.empty((self.pool_size,) + tuple(y.shape[1:]), dtype=torch.float16, device=self.device)
        for i in range(x.shape[0]):
            self.pool_x[self.pool_ptr], self.pool_y[self.pool_ptr] = x[i], y[i]
            self.pool_ptr = (self.pool_ptr + 1) % self.pool_size
            self.pool_n = min(self.pool_n + 1, self.pool_size)

    def next(self, batch):
        b = None
        if self.echo_run >= self.max_echo or self.pool_n < batch:
            b = self.q.get()
        else:
            try:
                b = self.q.get_nowait()
            except queue.Empty:
                pass
        if b is not None:
            x, y = b[0].to(self.device, non_blocking=True), b[1].to(self.device, non_blocking=True)
            self._add_to_pool(x, y)
            self.echo_run, fresh = 0, True
            self.n_fresh += 1
        else:
            pick = torch.randint(0, self.pool_n, (batch,), device=self.device)
            x, y = self.pool_x[pick], self.pool_y[pick]
            self.echo_run, fresh = self.echo_run + 1, False
            self.n_echo += 1
        x, y = dihedral(x, y)
        return x.float(), y.float(), fresh

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------
def reconstruction_loss(pred, y, w_ssim):
    l1 = (pred - y).abs().mean()
    ssim_l = 1.0 - ssim_per_image(pred.clamp(0, 1), y).mean()
    return l1 + w_ssim * ssim_l, l1, ssim_l


def lsgan(logits, target):
    return 0.5 * (logits.float() - target).pow(2).mean()


def adaptive_lambda(rec, adv, layer_weight, factor, scale=4096.0):
    """lambda such that |d(lambda*adv)/dw| = factor * |d rec/dw| at the last shared generator layer w."""
    g_rec, = torch.autograd.grad(rec * scale, layer_weight, retain_graph=True)
    g_adv, = torch.autograd.grad(adv * scale, layer_weight, retain_graph=True)
    n_rec, n_adv = g_rec.float().norm(), g_adv.float().norm()
    if not (torch.isfinite(n_rec) and torch.isfinite(n_adv)) or n_adv < 1e-12:
        return None
    return float((factor * n_rec / n_adv).clamp(0, 1e4))


# --------------------------------------------------------------------------------------
# Schedules (functions of the fraction of the training budget that has elapsed)
# --------------------------------------------------------------------------------------
def cosine(f, lo=0.0):
    f = min(max(f, 0.0), 1.0)
    return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * f))


def lr_g(cfg, frac, step):
    a = cfg["gan_start"]
    if frac < a:
        warm = min(1.0, (step + 1) / cfg["warmup_steps"])
        return cfg["lr"] * warm * cosine(frac / a, cfg["lr_floor"])
    return cfg["lr_gan"] * cosine((frac - a) / (1 - a), 0.01)


def lr_d(cfg, frac):
    a = cfg["gan_start"]
    return cfg["lr_d"] * cosine((frac - a) / (1 - a), 0.01)


# --------------------------------------------------------------------------------------
# Checkpoints (the same format as Phase 1, so src.models.architectures.load_model_from_checkpoint works)
# --------------------------------------------------------------------------------------
def save_generator(path, model, cfg, extra=None):
    tmp = path + ".tmp"
    torch.save(dict(arch="clearsar_net", in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, uncertainty=False,
                    model_kwargs=cfg["model"], model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    **(extra or {})), tmp)
    os.replace(tmp, path)


def unwrap(m):
    return m.module if hasattr(m, "module") else m


# --------------------------------------------------------------------------------------
# Validation (EMA generator, patches held in RAM)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def validate(model, vx, vy, device, batch=8):
    model.eval()
    acc = {}
    for i in range(0, len(vx), batch):
        x = vx[i:i + batch].to(device).float()
        y = vy[i:i + batch].to(device).float()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = model(x)
        for k, v in compute_metrics(p.float(), y).items():
            acc.setdefault(k, []).append(v)
    m = {k: float(np.nanmean(np.concatenate(v))) for k, v in acc.items()}
    m["score"] = m["mae"] * (1 + abs(1 - m["sharpness"]))
    return m


# --------------------------------------------------------------------------------------
# Batch-size probe
# --------------------------------------------------------------------------------------
def probe_batch(G, D, device, candidates, w_ssim, budget=0.85):
    """Largest batch whose full part-B step (G and D forward + backward) fits in budget * GPU memory
    (the rest is headroom for the echo pool, ~0.5 GB, and the adaptive-weight gradients)."""
    if device.type != "cuda":
        return candidates[-1]
    import copy
    keep_g, keep_d = copy.deepcopy(G.state_dict()), copy.deepcopy(D.state_dict())
    try:
        return _probe(G, D, device, candidates, w_ssim, budget)
    finally:                                # the probe ran BN / spectral-norm updates on random data: undo them
        G.load_state_dict(keep_g)
        D.load_state_dict(keep_d)
        G.zero_grad(set_to_none=True)
        D.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def _probe(G, D, device, candidates, w_ssim, budget):
    total = torch.cuda.get_device_properties(device).total_memory
    for b in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.rand(b, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE, device=device)
            y = torch.rand(b, OUT_CHANNELS, PATCH_SIZE, PATCH_SIZE, device=device)
            with torch.autocast("cuda"):
                p = G(x).float()
                d = D(torch.cat([x, x]), torch.cat([y, p.detach()]))
            lsgan(d, 1.0).backward()
            with torch.autocast("cuda"):
                adv = lsgan(D(x, p), 1.0)
            (reconstruction_loss(p, y, w_ssim)[0] + adv).backward()
            G.zero_grad(set_to_none=True)
            D.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device)
            del x, y, p, d, adv
            if peak < budget * total:
                torch.cuda.empty_cache()
                return b
        except torch.OutOfMemoryError:
            pass
        G.zero_grad(set_to_none=True)
        D.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return candidates[-1]


# --------------------------------------------------------------------------------------
# The training process (one per GPU)
# --------------------------------------------------------------------------------------
def _mean(values):
    """Mean over the finite values (skipped non-finite steps are counted separately in 'nonfinite')."""
    return float(torch.stack([torch.as_tensor(v, dtype=torch.float32).cpu() for v in values]).nanmean())


def _cpu_copy(obj):
    """Deep copy of a (nested) state dict with every tensor moved to the CPU."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_copy(v) for v in obj)
    return obj


def _finite_state(module):
    return all(bool(torch.isfinite(v).all()) for v in module.state_dict().values() if v.is_floating_point())


class RankLog:
    def __init__(self, path, rank):
        self.rank = rank
        self.f = open(path, "a", encoding="utf-8") if rank == 0 else None

    def __call__(self, *msg):
        if self.rank != 0:
            return
        line = time.strftime("%H:%M:%S ") + " ".join(str(m) for m in msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def train_worker(rank, world, cfg, train_records, val_records, band_means):
    from src.models.architectures import build_model
    from src.models.discriminator import SNPatchDiscriminator

    out = cfg["out_dir"]
    log = RankLog(os.path.join(out, "logs", "train_log.txt"), rank)
    ddp = world > 1
    if ddp:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(cfg.get("port", 29533)))
        try:
            dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=60),
                                    device_id=torch.device(f"cuda:{rank}"))
        except TypeError:                              # older torch without device_id
            dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=60))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg["seed"])                      # same initial weights on every rank
    np.random.seed(cfg["seed"] + rank)

    G = build_model("clearsar_net", **cfg["model"]).to(device)
    G.init_output_bias(band_means)
    D = SNPatchDiscriminator(IN_CHANNELS + OUT_CHANNELS, cfg["d_base"]).to(device)
    torch.manual_seed(cfg["seed"] + 1000 + rank)        # different augmentation / echo draws per rank
    n_g = sum(p.numel() for p in G.parameters())
    n_d = sum(p.numel() for p in D.parameters())

    batch = probe_batch(G, D, device, cfg["batch_candidates"], cfg["w_ssim"])
    if ddp:
        t = torch.tensor([batch], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        batch = int(t.item())
    log(f"[train] generator {n_g / 1e6:.2f} M params, discriminator {n_d / 1e6:.2f} M | world {world} | "
        f"batch {batch} per GPU (global {batch * world})")

    last_shared = G.dec1.conv[3].weight                 # for the adaptive GAN weight
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        G = DDP(G, device_ids=[rank])
        G.register_comm_hook(None, default_hooks.fp16_compress_hook)    # send gradients as fp16: half the traffic
        D = DDP(D, device_ids=[rank])
    decay = [p for n, p in unwrap(G).named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in unwrap(G).named_parameters() if p.ndim <= 1]
    opt_g = torch.optim.AdamW([dict(params=decay, weight_decay=cfg["weight_decay"]),
                               dict(params=no_decay, weight_decay=0.0)], lr=cfg["lr"], betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    amp = device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=amp)
    ema = None
    if rank == 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(unwrap(G), multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]), use_buffers=True)

    sampler = None
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(range(len(train_records)), num_replicas=world, rank=rank, shuffle=True,
                                     seed=cfg["seed"], drop_last=True)
    loader = make_loader(train_records, batch, cfg["workers"], shuffle=sampler is None, sampler=sampler, drop_last=True)
    feeder = EchoFeeder(loader, sampler, device, cfg["pool_size"], cfg["max_echo"])

    vx = vy = None
    if rank == 0:                                       # validation patches, read once into RAM
        xs, ys = [], []
        for b in make_loader(val_records, 16, cfg["workers"]):
            if b is not None:
                xs.append(b[0])
                ys.append(b[1])
        vx, vy = torch.cat(xs), torch.cat(ys)
        log(f"[val] {len(vx)} validation patches in RAM")
    if ddp:
        dist.barrier()

    hist_p = os.path.join(out, "logs", "train_history.csv")
    val_p = os.path.join(out, "logs", "val_history.csv")
    ck = os.path.join(out, "checkpoints")
    budget = cfg["train_seconds"]
    t0 = time.time()
    step, last_val, rows, vrows = 0, t0, [], []
    best_a, best_b = float("inf"), float("inf")
    lam_s, b_step = None, 0
    acc = {}
    t_log, fresh_log = time.time(), (0, 0)
    phase_logged = False
    log(f"[train] budget {budget / 3600:.2f} h | part A (reconstruction) until {cfg['gan_start']:.0%}, "
        f"then part B (+ PatchGAN, adaptive weight factor {cfg['adv_factor']})")

    # ---- NaN guard. GradScaler already skips single steps whose gradients are inf / NaN. If MOST steps keep
    # failing (the weights or the batch-norm statistics have gone bad), every rank rolls back to the last good
    # snapshot (weights, optimiser, EMA), halves the learning rate and carries on. Never more than max_restores.
    def take_snapshot():
        return dict(step=step, g=_cpu_copy(unwrap(G).state_dict()), opt_g=_cpu_copy(opt_g.state_dict()),
                    d=_cpu_copy(unwrap(D).state_dict()), opt_d=_cpu_copy(opt_d.state_dict()),
                    ema=_cpu_copy(ema.state_dict()) if ema is not None else None)

    snap = take_snapshot()
    bad_t = torch.zeros((), device=device)
    bad_run, restores, lr_mult, need_restore, guard_events = 0, 0, 1.0, False, []
    in_b = stop = False
    while True:
        # ---- events (validation, part B, stop) are decided by rank 0 from its clock every sync_every steps
        # and broadcast, so both GPUs always do the same thing; the learning rate uses each rank's own clock
        frac = (time.time() - t0) / budget
        do_val = False
        if step % cfg["sync_every"] == 0:
            flags = torch.tensor([frac, float((time.time() - last_val) >= cfg["val_every_min"] * 60), float(frac >= 1.0),
                                  float(need_restore)], device=device, dtype=torch.float64)
            if ddp:
                dist.broadcast(flags, 0)
                dist.all_reduce(bad_t, op=dist.ReduceOp.MAX)
            frac, do_val, stop, val_bad = float(flags[0]), bool(flags[1]), bool(flags[2]), bool(flags[3])
            in_b = frac >= cfg["gan_start"]
            n_bad = int(bad_t.item())
            bad_t.zero_()
            bad_run = bad_run + 1 if n_bad > cfg["sync_every"] // 2 else 0
            if (bad_run >= cfg["nan_patience"] or val_bad) and not stop:
                restores += 1
                why = "validation is NaN" if val_bad else f"{bad_run * cfg['sync_every']} steps mostly non-finite"
                if restores > cfg["max_restores"]:
                    log(f"[guard] step {step}: {why} again - stopping training early, keeping the best checkpoints")
                    guard_events.append(dict(step=step, action="stop", why=why))
                    stop = True
                else:
                    unwrap(G).load_state_dict(snap["g"])
                    opt_g.load_state_dict(snap["opt_g"])
                    unwrap(D).load_state_dict(snap["d"])
                    opt_d.load_state_dict(snap["opt_d"])
                    if ema is not None:
                        ema.load_state_dict(snap["ema"])
                    scaler_g = torch.amp.GradScaler("cuda", enabled=amp)
                    scaler_d = torch.amp.GradScaler("cuda", enabled=amp)
                    lr_mult *= 0.5
                    log(f"[guard] step {step}: {why} -> rolled back to step {snap['step']}, learning rate x{lr_mult:g}")
                    guard_events.append(dict(step=step, action="rollback", to_step=snap["step"], lr_mult=lr_mult, why=why))
                bad_run, need_restore = 0, False
            elif n_bad == 0 and step - snap["step"] >= cfg["snapshot_every"] and _finite_state(unwrap(G)):
                snap = take_snapshot()
        frac = max(frac, cfg["gan_start"]) if in_b else min(frac, cfg["gan_start"] - 1e-9)
        if in_b and not phase_logged:
            phase_logged = True
            log(f"[train] ===== part B: the discriminator joins at step {step} =====")
            if rank == 0:                                       # close part A with a validation
                m = validate(ema.module, vx, vy, device)
                if m["mae"] < best_a:
                    best_a = m["mae"]
                    save_generator(os.path.join(ck, "clearsar_net_before_gan.pt"), ema.module, cfg,
                                   dict(step=step, val=m))
                vrows.append(dict(step=step, time_h=(time.time() - t0) / 3600, phase="A_end", **m))
                log(f"[val] end of part A: L1 {m['mae']:.4f} PSNR {m['psnr']:.2f} sharp {m['sharpness']:.2f}")
                pd.DataFrame(vrows).to_csv(val_p, index=False)
            last_val = time.time()
            if ddp:
                dist.barrier()
        if stop:
            break

        x, y, fresh = feeder.next(batch)
        g_lr = lr_g(cfg, frac, step) * lr_mult
        for gr in opt_g.param_groups:
            gr["lr"] = g_lr
        rec_parts = {}

        with torch.autocast(device_type=device.type, enabled=amp):
            pred = G(x).float()
        rec, l1, ssim_l = reconstruction_loss(pred, y, cfg["w_ssim"])
        loss = rec
        if in_b:
            # ---- discriminator step: real pairs (x, y) -> 1, generated pairs (x, G(x)) -> 0
            d_lr = lr_d(cfg, frac) * lr_mult
            for gr in opt_d.param_groups:
                gr["lr"] = d_lr
            unwrap(D).train()
            for p in D.parameters():
                p.requires_grad_(True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = D(torch.cat([x, x]), torch.cat([y, pred.detach()])).float()
            d_real, d_fake = logits.chunk(2)
            d_loss = lsgan(d_real, 1.0) + lsgan(d_fake, 0.0)
            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            # ---- generator's adversarial term (D frozen, called without its DDP wrapper)
            ramp = 0.0 if b_step < cfg["d_warmup_steps"] else min(1.0, (b_step - cfg["d_warmup_steps"]) / cfg["adv_ramp_steps"])
            if ramp > 0:
                for p in D.parameters():
                    p.requires_grad_(False)
                unwrap(D).eval()
                with torch.autocast(device_type=device.type, enabled=amp):
                    adv = lsgan(unwrap(D)(x, pred), 1.0)
                if lam_s is None or b_step % cfg["lambda_every"] == 0:
                    lam = adaptive_lambda(rec, adv, last_shared, cfg["adv_factor"])
                    if lam is not None:
                        lam_s = lam if lam_s is None else 0.9 * lam_s + 0.1 * lam
                if lam_s is not None:
                    loss = rec + ramp * lam_s * adv
                rec_parts.update(adv=adv.detach(), lam=(lam_s or 0.0) * ramp)
            rec_parts.update(d_loss=d_loss.detach(), d_real=d_real.mean().detach(), d_fake=d_fake.mean().detach(), lr_d=d_lr)
            b_step += 1

        bad_t += (~torch.isfinite(loss.detach())).float()
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(loss).backward()
        scaler_g.unscale_(opt_g)
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), cfg["grad_clip"])
        scaler_g.step(opt_g)
        scaler_g.update()
        if ema is not None:
            ema.update_parameters(unwrap(G))
        step += 1
        if cfg.get("inject_nan_step") and step == cfg["inject_nan_step"]:     # test hook: break the model on purpose
            with torch.no_grad():
                unwrap(G).dec1.conv[0].weight.fill_(float("inf"))

        # ---- logging
        for k, v in dict(loss=loss.detach(), l1=l1.detach(), ssim_loss=ssim_l.detach(), gnorm=gnorm.detach(),
                         nonfinite=(~torch.isfinite(loss.detach())).float(), **rec_parts).items():
            acc.setdefault(k, []).append(v)        # tensors stay on the GPU until the next log line (no stall)
        if step % cfg["log_every"] == 0:
            dt = time.time() - t_log
            nf, ne = feeder.n_fresh - fresh_log[0], feeder.n_echo - fresh_log[1]
            row = dict(step=step, time_h=(time.time() - t0) / 3600, phase="B" if in_b else "A", lr_g=g_lr,
                       it_s=cfg["log_every"] / dt, fresh=nf / max(1, nf + ne), amp_scale=scaler_g.get_scale() if amp else 1.0,
                       patches_read_s=nf * batch * world / dt, epoch=feeder.epoch,
                       **{k: _mean(v) for k, v in acc.items()})
            rows.append(row)
            acc, t_log, fresh_log = {}, time.time(), (feeder.n_fresh, feeder.n_echo)
            if rank == 0:
                pd.DataFrame(rows).to_csv(hist_p, index=False)
                mem = torch.cuda.max_memory_allocated(device) / 1e9 if amp else 0
                gan = (f" | D {row['d_loss']:.3f} (real {row['d_real']:.2f} fake {row['d_fake']:.2f})"
                       f" adv {row.get('adv', float('nan')):.3f} lambda {row.get('lam', 0):.3g}") if in_b else ""
                skip = f" | SKIPPED {row['nonfinite']:.0%} (non-finite)" if row["nonfinite"] > 0 else ""
                log(f"[{row['phase']}] step {step:6d} {frac:5.1%} | L1 {row['l1']:.4f} 1-SSIM {row['ssim_loss']:.4f}{skip}"
                    f"{gan} | lr {g_lr:.2e} | {row['it_s']:.2f} it/s, fresh {row['fresh']:.0%}, "
                    f"read {row['patches_read_s']:.0f} patch/s, epoch {feeder.epoch} | {mem:.1f} GB | scale {row['amp_scale']:.0f}")

        # ---- validation + checkpoint selection (rank 0)
        if do_val:
            if rank == 0:
                m = validate(ema.module, vx, vy, device)
                tag = ""
                if not in_b and m["mae"] < best_a:
                    best_a, tag = m["mae"], " * best before-GAN"
                    save_generator(os.path.join(ck, "clearsar_net_before_gan.pt"), ema.module, cfg, dict(step=step, val=m))
                gan_on = in_b and b_step >= cfg["d_warmup_steps"] + cfg["adv_ramp_steps"]
                if gan_on and m["score"] < best_b:
                    best_b, tag = m["score"], " * best final"
                    save_generator(os.path.join(ck, "clearsar_net_final.pt"), ema.module, cfg, dict(step=step, val=m))
                vrows.append(dict(step=step, time_h=(time.time() - t0) / 3600, phase="B" if in_b else "A",
                                  gan_on=gan_on, **m))
                pd.DataFrame(vrows).to_csv(val_p, index=False)
                if not np.isfinite(m["mae"]):
                    need_restore = True
                    log("[guard] WARNING validation is NaN - the next sync rolls back to the last good snapshot")
                log(f"[val] step {step}: L1 {m['mae']:.4f} PSNR {m['psnr']:.2f} SSIM {m['ssim']:.3f} "
                    f"aligned {m['psnr_aligned']:.2f} sharp {m['sharpness']:.2f} score {m['score']:.4f}{tag}")
            last_val = time.time()
            if ddp:
                dist.barrier()

    feeder.close()
    if rank == 0:
        m = validate(ema.module, vx, vy, device)
        save_generator(os.path.join(ck, "clearsar_net_last.pt"), ema.module, cfg, dict(step=step, val=m))
        if not os.path.exists(os.path.join(ck, "clearsar_net_final.pt")):
            log("[train] no validation after the GAN was fully on - the last weights become the final model")
            save_generator(os.path.join(ck, "clearsar_net_final.pt"), ema.module, cfg, dict(step=step, val=m))
        if not os.path.exists(os.path.join(ck, "clearsar_net_before_gan.pt")):
            save_generator(os.path.join(ck, "clearsar_net_before_gan.pt"), ema.module, cfg, dict(step=step, val=m))
        vrows.append(dict(step=step, time_h=(time.time() - t0) / 3600, phase="end", **m))
        pd.DataFrame(vrows).to_csv(val_p, index=False)
        with open(os.path.join(out, "logs", "train_summary.json"), "w") as f:
            json.dump(dict(steps=step, batch_per_gpu=batch, world=world, generator_params=n_g, discriminator_params=n_d,
                           train_hours=(time.time() - t0) / 3600, best_val_l1_before_gan=best_a,
                           best_val_score_final=best_b, epochs_started=feeder.epoch,
                           fresh_fraction=feeder.n_fresh / max(1, feeder.n_fresh + feeder.n_echo),
                           nan_guard_events=guard_events), f, indent=2)
        log(f"[train] done: {step} steps in {(time.time() - t0) / 3600:.2f} h, {feeder.epoch} epoch(s) started")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
