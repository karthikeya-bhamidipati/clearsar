"""
Plan v4, stage 0 - understand the data and what is achievable, before training anything.

For every patch the study measures, using only cheap statistics (no deep learning):

  clear_px_psnr        PSNR of the cloudy image against the clear target ON CLOUD-FREE PIXELS.
                       Even where the ground is visible the two images differ, because they were taken on
                       different dates (vegetation, soil moisture, sun angle, atmosphere). This is roughly
                       the best a model can do on visible ground without learning those date differences.
  oracle_shift_psnr    the same after a per-band linear fit  y ≈ a·x + b  on the clear pixels. The fit uses
                       the TARGET, so this is an oracle (not achievable): it shows how much of the visible-
                       ground difference is just a global brightness/colour shift between the two dates.
                       (Real data: ~25 dB -> 31-40 dB - most of it is the date shift.)
  oracle_fill_psnr     whole-patch PSNR when visible pixels are shift-corrected and cloudy pixels are filled
                       with the corrected mean of each band - an oracle "no deep learning" reconstruction.

Cloud-free pixels are found from the cloudy image alone (the expert system's brightness/whiteness test), so
the masks are realistic. Results are grouped by the same cloud-cover bins used everywhere else.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.dataset import N_S1, to_rgb  # noqa: E402
from src.expert.facts import cloud_masks  # noqa: E402

BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
BIN_NAMES = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def _psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def patch_stats(x, y, min_clear=0.05):
    """x: [15, H, W] input, y: [13, H, W] target (both in [0, 1])."""
    cloudy = x[N_S1:]
    cloud, _ = cloud_masks(x)
    clear = ~cloud
    out = dict(cloud_est=float(cloud.mean()), identity_psnr=_psnr(cloudy, y))
    if clear.mean() >= min_clear:
        cx, cy = cloudy[:, clear], y[:, clear]                     # [13, n]
        out["clear_px_psnr"] = _psnr(cx, cy)
        corrected, fill = np.empty_like(cy), np.empty_like(y)
        for b in range(cx.shape[0]):
            a, c = np.polyfit(cx[b], cy[b], 1) if cx[b].std() > 1e-6 else (0.0, float(cy[b].mean()))
            corrected[b] = np.clip(a * cx[b] + c, 0, 1)
            fill[b] = np.where(clear, np.clip(a * cloudy[b] + c, 0, 1), float(corrected[b].mean()))
        out["oracle_shift_psnr"] = _psnr(corrected, cy)
        out["oracle_fill_psnr"] = _psnr(fill, y)
    else:
        out.update(clear_px_psnr=np.nan, oracle_shift_psnr=np.nan, oracle_fill_psnr=np.nan)
    return out


def run_data_study(X, Y, frame, out_dir, split_name="val", log=print):
    """X, Y: memmaps of one cached split; frame: its split dataframe (valid, cache_idx, cloud_frac,
    continent, country). Writes stage0_* files into out_dir and returns the per-bin table."""
    import os
    rows = []
    for r in frame[frame.valid].itertuples():
        x, y = np.asarray(X[r.cache_idx], np.float32), np.asarray(Y[r.cache_idx], np.float32)
        rows.append(dict(cache_idx=r.cache_idx, continent=r.continent, country=r.country,
                         cloud_frac=r.cloud_frac, **patch_stats(x, y)))
    d = pd.DataFrame(rows)
    d["cloud_bin"] = pd.cut(d.cloud_frac, BINS, labels=BIN_NAMES, include_lowest=True)
    d.to_csv(os.path.join(out_dir, "stage0_patch_stats.csv"), index=False)
    t = d.groupby("cloud_bin", observed=True).agg(
        n=("cache_idx", "size"), identity_psnr=("identity_psnr", "mean"), clear_px_psnr=("clear_px_psnr", "mean"),
        oracle_shift_psnr=("oracle_shift_psnr", "mean"), oracle_fill_psnr=("oracle_fill_psnr", "mean")).reset_index()
    t.to_csv(os.path.join(out_dir, "stage0_ceiling.csv"), index=False)
    log(f"[stage0] {split_name}: data ceiling per cloud bin\n" + t.round(2).to_string(index=False))

    # cloud-cover distribution per continent
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for cont, g in d.groupby("continent"):
        ax.hist(g.cloud_frac, bins=20, range=(0, 1), histtype="step", linewidth=2, label=f"{cont} ({len(g)})")
    ax.set_xlabel("cloud fraction of the patch")
    ax.set_ylabel("patches")
    ax.set_title(f"Cloud cover - {split_name} scenes")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage0_cloud_hist.png"), dpi=130)
    plt.close(fig)

    # triplet gallery: 12 patches spread over cloud levels and continents
    pick = d.sort_values("cloud_frac").iloc[np.linspace(0, len(d) - 1, min(12, len(d))).astype(int)]
    fig, axes = plt.subplots(len(pick), 3, figsize=(7.5, 2.5 * len(pick)), squeeze=False)
    for ax_row, r in zip(axes, pick.itertuples()):
        x = np.asarray(X[r.cache_idx], np.float32)
        y = np.asarray(Y[r.cache_idx], np.float32)
        for ax, img, title in zip(ax_row, [x[0], to_rgb(x[N_S1:]), to_rgb(y)],
                                  ["radar VV", f"cloudy · {int(r.cloud_frac * 100)}% cloud", f"clear · {r.country}"]):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage0_triplets.png"), dpi=110)
    plt.close(fig)
    return t
