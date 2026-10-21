"""Post-hoc analyses run after training (no extra training needed)."""
import numpy as np
import torch


def effective_receptive_field(model, x):
    """E3c (Luo et al. 2016): |d centre-output-pixel / d input|, summed over channels and the
    batch, normalised to sum 1. x: [B, C, H, W] on the model's device. Returns [H, W] numpy."""
    model.eval()
    x = x.detach().clone().float().requires_grad_(True)
    out = model(x)
    C = getattr(model, "out_channels", out.shape[1])
    H, W = out.shape[-2:]
    out[:, :C, H // 2, W // 2].sum().backward()
    g = x.grad.abs().sum(dim=(0, 1))
    return (g / g.sum().clamp_min(1e-12)).cpu().numpy()


def erf_radius(erf, mass):
    """Smallest radius (pixels, around the centre) that contains `mass` of the ERF."""
    H, W = erf.shape
    yy, xx = np.mgrid[:H, :W]
    d = np.hypot(yy - H // 2, xx - W // 2).ravel()
    order = np.argsort(d)
    cum = np.cumsum(erf.ravel()[order])
    return float(d[order][np.searchsorted(cum, mass * cum[-1])])


def erf_stats(erf):
    return dict(radius50=erf_radius(erf, 0.5), radius90=erf_radius(erf, 0.9))
