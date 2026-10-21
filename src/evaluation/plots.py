"""Figures written by the Kaggle pipeline into <out>/results/ (matplotlib, headless)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.dataset import N_S1, to_rgb  # noqa: E402

SPLIT_LABELS = {"test_id": "In-domain test (Asia)", "ood": "Out-of-continent test"}


def comparison_grid(samples, run_labels, path, dpi=150):
    """samples: list of dicts with x [15,H,W], y [13,H,W], preds {run: [13,H,W]},
    sigma (optional [H,W]), title. One row per sample."""
    runs = list(run_labels)
    has_sigma = any(s.get("sigma") is not None for s in samples)
    cols = 3 + len(runs) + (1 if has_sigma else 0)
    fig, axes = plt.subplots(len(samples), cols, figsize=(2.6 * cols, 2.7 * len(samples)), squeeze=False)
    for r, s in enumerate(samples):
        panels = [(s["x"][0], "SAR VV", "gray"), (to_rgb(s["x"][N_S1:]), "Cloudy input", None),
                  (to_rgb(s["y"]), "Target (clear)", None)]
        for run in runs:
            p = s["preds"].get(run)
            panels.append((to_rgb(p) if p is not None else np.zeros_like(panels[1][0]),
                           f"{run_labels[run]}\n{s.get('scores', {}).get(run, '')}", None))
        if has_sigma:
            panels.append((s.get("sigma") if s.get("sigma") is not None else np.zeros(s["y"].shape[1:]),
                           "Uncertainty (b)", "magma"))
        for c, (img, title, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        axes[r, 0].text(-0.08, 0.5, s.get("title", ""), transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close(fig)


def training_curves(histories, run_labels, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for run, h in histories.items():
        if h is None or h.empty:
            continue
        lab = run_labels.get(run, run)
        axes[0].plot(h.step, h.train_rec, label=lab)
        axes[1].plot(h.step, h.val_l1, label=lab)
        axes[2].plot(h.step, h.val_psnr, label=lab)
    for ax, t in zip(axes, ["Train reconstruction loss", "Validation L1", "Validation PSNR (dB)"]):
        ax.set_title(t)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def metric_bars(summary, path, metrics=("psnr", "ssim", "sam", "mae")):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    labels = summary["label"].unique().tolist()
    x = np.arange(len(labels))
    for ax, m in zip(axes, metrics):
        for k, split in enumerate(["test_id", "ood"]):
            sub = summary[summary.split == split].set_index("label").reindex(labels)
            ax.bar(x + (k - 0.5) * 0.38, sub[m], 0.38, label=SPLIT_LABELS[split])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(m.upper() + (" (lower is better)" if m in ("sam", "mae", "rmse") else ""))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def cloud_bin_plot(by_bin, path, metric="psnr"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, ["test_id", "ood"]):
        sub = by_bin[by_bin.split == split]
        for lab, g in sub.groupby("label", sort=False):
            ax.plot(g.cloud_bin.astype(str), g[metric], marker="o", label=lab)
        ax.set_title(f"{SPLIT_LABELS[split]}: {metric.upper()} by cloud cover")
        ax.set_xlabel("cloud cover bin")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def sparsification_plot(curves, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for split, sp in curves.items():
        ax.plot(sp["fractions"], sp["model"], label=f"{SPLIT_LABELS.get(split, split)} (model)")
        ax.plot(sp["fractions"], sp["oracle"], "--", label=f"{SPLIT_LABELS.get(split, split)} (oracle)")
    ax.set_xlabel("fraction of most-uncertain pixels removed")
    ax.set_ylabel("MAE of remaining pixels")
    ax.set_title("Sparsification curve (Model A + variance head)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
