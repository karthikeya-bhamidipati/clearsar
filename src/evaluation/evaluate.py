"""
Tiled inference and evaluation.

Every model is trained on 128x128 crops and evaluated by cutting each 256x256 patch into
four 128x128 tiles, predicting each tile, and stitching the result back together - the
same input size for every architecture (docs/project_plan.md §3).
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import N_S1, RGB_IDX, RGB_GAIN
from src.evaluation.metrics import FIDScorer, compute_metrics


@torch.no_grad()
def predict_tiled(model, x, tile=128, amp=True):
    """x: [B, C, H, W] -> model output [B, C_out, H, W], predicted tile by tile.
    Inputs whose size is not a multiple of `tile` are reflect-padded and cropped back."""
    B, C, H, W = x.shape
    ph, pw = (-H) % tile, (-W) % tile
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    Hp, Wp = x.shape[-2:]
    nh, nw = Hp // tile, Wp // tile
    tiles = x.reshape(B, C, nh, tile, nw, tile).permute(0, 2, 4, 1, 3, 5).reshape(B * nh * nw, C, tile, tile)
    use_amp = amp and x.is_cuda
    with torch.autocast(device_type=x.device.type, enabled=use_amp):
        out = model(tiles)
    out = out.float()
    Co = out.shape[1]
    out = out.reshape(B, nh, nw, Co, tile, tile).permute(0, 3, 1, 4, 2, 5).reshape(B, Co, Hp, Wp)
    return out[..., :H, :W]


def rgb_tensor(bands, gain=RGB_GAIN):
    """[B, 13, H, W] reflectance -> [B, 3, H, W] display RGB in [0, 1]."""
    return (bands[:, list(RGB_IDX)] * gain).clamp(0, 1)


@torch.no_grad()
def evaluate(model, loader, device, out_channels=13, uncertainty=False, identity=False,
             fid=False, unc_pixels_per_image=2000, collect=None):
    """Run a model (or the identity baseline) over a loader of full patches.

    Returns (per-image metric dict of numpy arrays incl. 'cache_idx', extras) where extras
    holds 'fid' and, for uncertainty models, sampled (abs_error, sigma) pixel pairs.
    `collect` (set of cache indices) -> extras['samples'][idx] = (pred, sigma-map) in numpy."""
    if model is not None:
        model.eval()
    fid_scorer = FIDScorer(device) if fid else None
    rows = {k: [] for k in ("psnr", "ssim", "sam", "mae", "rmse", "cache_idx")}
    errs, sigs, samples = [], [], {}
    gen = torch.Generator().manual_seed(0)
    for x, y, idx in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        sigma = None
        if identity:
            pred = x[:, N_S1:N_S1 + out_channels]
        else:
            out = predict_tiled(model, x)
            pred = out[:, :out_channels]
            if uncertainty:
                sigma = torch.exp(out[:, out_channels:].clamp(-10, 10))
        m = compute_metrics(pred, y)
        for k, v in m.items():
            rows[k].append(v)
        rows["cache_idx"].append(idx.numpy())
        if fid_scorer is not None:
            fid_scorer.update(rgb_tensor(pred), rgb_tensor(y))
        if sigma is not None:
            e = (pred - y).abs().flatten(1).cpu()
            s = sigma.flatten(1).cpu()
            pick = torch.randint(0, e.shape[1], (e.shape[0], unc_pixels_per_image), generator=gen)
            errs.append(torch.gather(e, 1, pick).flatten().numpy())
            sigs.append(torch.gather(s, 1, pick).flatten().numpy())
        if collect:
            for b, i in enumerate(idx.tolist()):
                if i in collect:
                    samples[i] = (pred[b].cpu().numpy(),
                                  None if sigma is None else sigma[b].mean(0).cpu().numpy())
    result = {k: np.concatenate(v) for k, v in rows.items()}
    extras = {"fid": fid_scorer.compute() if fid_scorer else float("nan"), "samples": samples}
    if fid_scorer is not None and not fid_scorer.available:
        extras["fid_error"] = fid_scorer.error
    if errs:
        extras["unc_errors"], extras["unc_sigmas"] = np.concatenate(errs), np.concatenate(sigs)
    return result, extras
