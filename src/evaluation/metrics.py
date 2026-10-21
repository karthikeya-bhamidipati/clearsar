"""
Full-reference image metrics for [B, C, H, W] tensors in [0, 1] (all 13 bands).

    psnr  : dB, data range 1
    ssim  : Gaussian window 11, sigma 1.5 (Wang et al. 2004), averaged over bands
    sam   : spectral angle mapper in degrees, averaged over pixels
    mae   : mean absolute error
    rmse  : root mean squared error

All functions return one value per image ([B] tensor) so results can be grouped later by
scene / country / cloud-cover bin. Validated against scikit-image in tests/test_pipeline.py.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_window(size=11, sigma=1.5, device=None, dtype=torch.float32):
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim_per_image(pred, target, data_range=1.0, win_size=11, sigma=1.5):
    C = pred.shape[1]
    w = _gaussian_window(win_size, sigma, pred.device, pred.dtype).expand(C, 1, win_size, win_size)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    def filt(z):
        return F.conv2d(z, w, groups=C)            # 'valid' convolution, like skimage

    mu_x, mu_y = filt(pred), filt(target)
    sxx = filt(pred * pred) - mu_x ** 2
    syy = filt(target * target) - mu_y ** 2
    sxy = filt(pred * target) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    # skimage crops (win_size - 1) // 2 from each border before averaging; 'valid' conv does that.
    return s.mean(dim=(1, 2, 3))


def psnr_per_image(pred, target, data_range=1.0):
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10.0 * torch.log10(data_range ** 2 / mse)


def sam_per_image(pred, target, eps=1e-8):
    dot = (pred * target).sum(dim=1)
    denom = pred.norm(dim=1) * target.norm(dim=1)
    cos = (dot / denom.clamp_min(eps)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).mean(dim=(1, 2))


@torch.no_grad()
def compute_metrics(pred, target):
    """Returns a dict of [B] numpy arrays: psnr, ssim, sam, mae, rmse."""
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    diff = pred - target
    return {
        "psnr": psnr_per_image(pred, target).cpu().numpy(),
        "ssim": ssim_per_image(pred, target).cpu().numpy(),
        "sam": sam_per_image(pred, target).cpu().numpy(),
        "mae": diff.abs().mean(dim=(1, 2, 3)).cpu().numpy(),
        "rmse": diff.pow(2).mean(dim=(1, 2, 3)).sqrt().cpu().numpy(),
    }


class FIDScorer:
    """Real FID (Inception-v3 pool features) via torchmetrics, on display-RGB images.
    Needs `torchmetrics` + `torch-fidelity` and internet for the first weight download.
    If anything is missing, .available is False and compute() returns NaN, so the
    pipeline never crashes because of FID."""

    def __init__(self, device):
        self.available = False
        self.error = ""
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            self.available = True
        except Exception as e:  # pragma: no cover - depends on environment
            self.error = f"{type(e).__name__}: {e}"
        self.n = 0

    def update(self, pred_rgb, target_rgb):
        """[B, 3, H, W] float in [0, 1]."""
        if not self.available:
            return
        self.metric.update(target_rgb.float().clamp(0, 1), real=True)
        self.metric.update(pred_rgb.float().clamp(0, 1), real=False)
        self.n += pred_rgb.shape[0]

    def compute(self):
        if not self.available or self.n < 2:
            return float("nan")
        try:
            return float(self.metric.compute())
        except Exception as e:  # pragma: no cover
            self.error = f"{type(e).__name__}: {e}"
            return float("nan")


# --------------------------------------------------------------------------------------
# Uncertainty (E7): predicted Laplace scale b = exp(s) vs actual absolute error
# --------------------------------------------------------------------------------------
LAPLACE_Z95 = math.log(20.0)   # P(|e| <= b * ln 20) = 0.95 for a Laplace distribution


def sparsification(errors, sigmas, fractions=np.linspace(0, 0.95, 20)):
    """Sparsification curve: MAE of the pixels that remain after removing the given
    fraction of most-uncertain pixels (model) vs most-erroneous pixels (oracle).
    AUSE = area between the two curves (lower is better; 0 = perfect ranking)."""
    errors, sigmas = np.asarray(errors), np.asarray(sigmas)
    by_unc = errors[np.argsort(-sigmas)]
    by_err = np.sort(errors)[::-1]
    n = len(errors)
    model, oracle = [], []
    for f in fractions:
        k = int(round(f * n))
        model.append(by_unc[k:].mean())
        oracle.append(by_err[k:].mean())
    model, oracle = np.array(model), np.array(oracle)
    ause = float(np.trapezoid(model - oracle, fractions)) if hasattr(np, "trapezoid") \
        else float(np.trapz(model - oracle, fractions))
    return dict(fractions=np.asarray(fractions), model=model, oracle=oracle, ause=ause)


GAUSS_Z95 = 1.959964           # P(|e| <= 1.96 sigma) = 0.95 for a Gaussian


def uncertainty_summary(errors, sigmas, kind="laplace"):
    """kind='laplace' for the variance head (sigma = Laplace scale b),
    'gaussian' for MC dropout / ensembles (sigma = standard deviation)."""
    errors, sigmas = np.asarray(errors), np.asarray(sigmas)
    z = LAPLACE_Z95 if kind == "laplace" else GAUSS_Z95
    from scipy.stats import spearmanr
    rho = spearmanr(errors, sigmas).correlation if len(errors) > 2 else float("nan")
    sp = sparsification(errors, sigmas)
    return dict(
        coverage95=float((errors <= z * sigmas).mean()),
        mean_sigma=float(sigmas.mean()),
        median_sigma=float(np.median(sigmas)),
        mean_abs_error=float(errors.mean()),
        spearman_err_sigma=float(rho),
        ause=sp["ause"],
    ), sp
