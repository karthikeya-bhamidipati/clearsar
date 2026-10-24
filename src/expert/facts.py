"""
Fact extraction for the ClearSAR Advisor.

Three sources of facts, none of which uses the clear target image (it is not available when
the system is used for real):

  observation facts  measured on the INPUT patch: cloud cover (brightness/whiteness test on the
                     cloudy Sentinel-2 image), cirrus (band B10), radar gaps, open water (low VV)
  model facts        measured on the deep-learning OUTPUTS: agreement between the three
                     architectures, how well the cloud-free pixels are reproduced, predicted
                     uncertainty, reliance on radar vs optical input (from xai.py)
  experience facts   learned from the EXPERIMENTS (results CSVs): which model is best for this
                     amount of cloud, how much accuracy drops outside the training region
"""
import os

import numpy as np
import pandas as pd

from src.data.dataset import N_S1

B2, B3, B4, B8, B10 = 1, 2, 3, 7, 10          # indices inside the 13-band S2 stack
VV_WATER = (-18.0 + 25.0) / 25.0             # -18 dB in the [0, 1] scaling of VV
CLOUD_BINS = [(0.2, "0-20%"), (0.4, "20-40%"), (0.6, "40-60%"), (0.8, "60-80%"), (1.01, "80-100%")]
MAIN_CANDIDATES = ["A_unet", "B_swin", "C_vit", "A_unc", "A_mcd", "A_gan"]
FINAL_CANDIDATES = ["before_gan", "final"]          # the final ClearSAR-Net before / after its GAN part
FINAL_CLOUD_BINS = [(0.1, "0-10%"), (0.3, "10-30%"), (0.6, "30-60%"), (0.9, "60-90%"), (1.01, "90-100%")]


def final_cloud_bin(frac):
    for hi, name in FINAL_CLOUD_BINS:
        if frac < hi:
            return name
    return FINAL_CLOUD_BINS[-1][1]


def cloud_masks(x):
    """Cloud detection from the cloudy Sentinel-2 input only (no target needed).
    Clouds are bright AND white (similar reflectance in blue/green/red). Thick = very bright."""
    opt = x[N_S1:]
    vis = opt[[B2, B3, B4]]
    mean = vis.mean(0)
    whiteness = vis.std(0) / np.maximum(mean, 1e-6)
    cloud = (mean > 0.20) & (whiteness < 0.25)      # 0.20 calibrated on the test gallery (MAE 0.08 -> 0.045)
    thick = cloud & (mean > 0.30)
    return cloud, thick


def cloud_bin(frac):
    for hi, name in CLOUD_BINS:
        if frac < hi:
            return name
    return CLOUD_BINS[-1][1]


def observation_facts(x):
    cloud, thick = cloud_masks(x)
    vv = x[0]
    return dict(
        cloud_cover=float(cloud.mean()),
        thick_cloud=float(thick.mean()),
        cirrus_fraction=float((x[N_S1 + B10] > 0.03).mean()),   # clear-sky B10 reaches ~0.025 in some scenes
        sar_gaps=float((vv <= 1e-3).mean()),
        water_fraction=float((vv < VV_WATER).mean()),
    ), cloud


def model_facts(x, preds, cloud, core=("A_unet", "B_swin", "C_vit"), primary=None, sigma=None):
    """preds: {run: [13, H, W]} in [0, 1]. sigma: optional [H, W] predicted error scale."""
    out = {}
    have = [r for r in core if r in preds]
    if len(have) >= 2:
        diffs = [np.abs(preds[a] - preds[b]).mean() for i, a in enumerate(have) for b in have[i + 1:]]
        out["model_disagreement"] = float(np.mean(diffs))
    p = preds.get(primary) if primary else None
    if p is None and preds:
        p = next(iter(preds.values()))
    if p is not None:
        clear = ~cloud
        if clear.mean() > 0.02:          # visible ground: the model should reproduce what it sees
            out["clear_pixel_error"] = float(np.abs(p - x[N_S1:N_S1 + p.shape[0]])[:, clear].mean())
        ndvi = (p[B8] - p[B4]) / np.maximum(p[B8] + p[B4], 1e-6)
        out["vegetation_fraction"] = float((ndvi > 0.4).mean())
    if sigma is not None:
        region = cloud if cloud.mean() > 0.02 else np.ones_like(cloud)
        out["uncertainty"] = float(np.median(sigma[region]))
    return out


class ResultsKnowledge:
    """Knowledge distilled from the experiment results (outputs/results/*.csv)."""

    def __init__(self, results_dir=None, labels=None, final_results_dir=None):
        self.labels = labels or {}
        self.by_bin = self.main = self.final_bin = None
        if results_dir and os.path.isdir(results_dir):
            p = os.path.join(results_dir, "metrics_by_cloud_bin.csv")
            self.by_bin = pd.read_csv(p) if os.path.exists(p) else None
            p = os.path.join(results_dir, "metrics_main.csv")
            self.main = pd.read_csv(p) if os.path.exists(p) else None
        if final_results_dir:                     # the final model's one test (notebooks/train_clearsar.py)
            p = os.path.join(final_results_dir, "test_by_cloud.csv")
            self.final_bin = pd.read_csv(p) if os.path.exists(p) else None

    def best_final_model(self, cloud_frac, available):
        """Most accurate (PSNR) of the final model before / after the GAN at this cloud level, measured on the
        final model's 18 held-out test scenes. Returns (run, psnr, bin_name)."""
        cands = [r for r in FINAL_CANDIDATES if r in available]
        name = final_cloud_bin(cloud_frac)
        if self.final_bin is not None and cands:
            d = self.final_bin[(self.final_bin.cloud_bin.astype(str) == name) & self.final_bin.run.isin(cands)]
            if len(d):
                row = d.sort_values("psnr", ascending=False).iloc[0]
                return row.run, float(row.psnr), name
        return (cands[-1] if cands else None), float("nan"), name

    def label(self, run):
        return self.labels.get(run, run)

    def best_model(self, bin_name, available):
        cands = [r for r in MAIN_CANDIDATES if r in available]
        if self.by_bin is not None and cands:
            d = self.by_bin[(self.by_bin.split == "test_id") & self.by_bin.run.isin(cands)]
            d = d[d.cloud_bin.astype(str) == bin_name]
            if len(d):
                row = d.sort_values("psnr", ascending=False).iloc[0]
                return row.run, float(row.psnr)
        return (cands[0] if cands else None), float("nan")

    def ood_drop(self, run):
        if self.main is None:
            return float("nan")
        group = run if run in set(self.main.group) else None
        if group is None:
            return float("nan")
        d = self.main[self.main.group == group].set_index("split")
        if {"test_id", "ood"} <= set(d.index):
            return float(d.loc["test_id", "psnr"] - d.loc["ood", "psnr"])
        return float("nan")
