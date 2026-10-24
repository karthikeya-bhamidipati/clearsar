"""
Explainability for the deep-learning models (model-agnostic, works for U-Net, Swin and ViT).

occlusion()        Which input did the model use? Replace the radar (or the cloudy optical image)
                   by its spatial mean - i.e. remove its spatial information - and measure how much
                   the prediction changes. The ratio gives a per-pixel "radar reliance" map.
channel_saliency() Which bands mattered? |input x gradient| of the predicted RGB over a region,
                   summed per input channel (VV, VH, B1 ... B12).
"""
import numpy as np
import torch

from src.data.dataset import N_S1, RGB_IDX, S2_BANDS

CHANNEL_NAMES = ["VV", "VH"] + [f"cloudy {b}" for b in S2_BANDS]


def _forward_tiled(model, x, tile=128):
    """Differentiable tiled forward pass. x: [1, C, H, W] (H, W multiples of tile). tile=None: whole patch
    (the final ClearSAR-Net is trained on full 256 x 256 patches)."""
    if not tile:
        return model(x)
    B, C, H, W = x.shape
    nh, nw = H // tile, W // tile
    t = x.reshape(B, C, nh, tile, nw, tile).permute(0, 2, 4, 1, 3, 5).reshape(B * nh * nw, C, tile, tile)
    out = model(t)
    Co = out.shape[1]
    return out.reshape(B, nh, nw, Co, tile, tile).permute(0, 3, 1, 4, 2, 5).reshape(B, Co, H, W)


@torch.no_grad()
def occlusion(model, x, tile=128):
    """x: [15, H, W] numpy in [0, 1]. Returns dict of [H, W] maps: delta_sar, delta_opt, reliance."""
    model.eval()
    oc = model.out_channels
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
    base = _forward_tiled(model, xt, tile)[:, :oc]
    no_sar = xt.clone()
    no_sar[:, :N_S1] = xt[:, :N_S1].mean(dim=(2, 3), keepdim=True)
    no_opt = xt.clone()
    no_opt[:, N_S1:] = xt[:, N_S1:].mean(dim=(2, 3), keepdim=True)
    d_sar = (_forward_tiled(model, no_sar, tile)[:, :oc] - base).abs().mean(1)[0].numpy()
    d_opt = (_forward_tiled(model, no_opt, tile)[:, :oc] - base).abs().mean(1)[0].numpy()
    return dict(delta_sar=d_sar, delta_opt=d_opt, reliance=d_sar / np.maximum(d_sar + d_opt, 1e-6))


def sar_reliance(occ, region):
    """Share of the prediction change caused by removing the radar, over a boolean region."""
    if region is None or region.mean() < 0.02:
        region = np.ones_like(occ["delta_sar"], dtype=bool)
    s, o = occ["delta_sar"][region].sum(), occ["delta_opt"][region].sum()
    return float(s / max(s + o, 1e-9))


def channel_saliency(model, x, region=None, tile=128):
    """Normalised importance of each of the 15 input channels for the predicted RGB in `region`."""
    model.eval()
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].requires_grad_(True)
    out = _forward_tiled(model, xt, tile)[:, list(RGB_IDX)]
    if region is not None and region.mean() > 0.02:
        m = torch.from_numpy(region.astype(np.float32))[None, None]
        target = (out * m).sum() / m.sum()
    else:
        target = out.mean()
    target.backward()
    imp = (xt.grad * xt).abs().sum(dim=(0, 2, 3)).detach().numpy()
    return imp / max(imp.sum(), 1e-12)
