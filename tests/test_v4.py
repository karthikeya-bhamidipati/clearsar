"""Tests for plan v4 (ClearSAR-Net, new metrics, losses, trainer options, global split):
python -m pytest tests/test_v4.py -q"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ["CLEARSAR_PERC_PRETRAINED"] = "0"          # no 500 MB VGG download in tests

from src.data.dataset import make_global_splits, scene_subset_indices  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.architectures import build_model, load_model_from_checkpoint  # noqa: E402
from src.models.clearsar_net import GlobalAttentionBlock, sinusoidal_2d  # noqa: E402


@pytest.mark.parametrize("mode", ["direct", "residual", "gated"])
@pytest.mark.parametrize("attn", [0, 2])
def test_clearsar_net_modes(mode, attn):
    m = build_model("clearsar_net", output_mode=mode, attn_layers=attn).eval()
    x = torch.rand(2, 15, 128, 128)
    with torch.no_grad():
        y, aux = m(x, return_aux=True)
        assert y.shape == (2, 13, 128, 128) and 0 <= y.min() and y.max() <= 1
        assert m(torch.rand(1, 15, 256, 256)).shape[-1] == 256          # trained on crops, runs on patches
    if mode == "residual":                                              # identity at initialisation
        assert (y - x[:, 2:].clamp(1e-3, 1 - 1e-3)).abs().max() < 1e-4
    if mode == "gated":
        assert abs(aux["gate"].mean().item() - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-3
    m.train()
    m(x).mean().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_uncertainty_head_and_bias_init():
    m = build_model("clearsar_net", output_mode="gated", uncertainty=True)
    assert m(torch.rand(1, 15, 128, 128)).shape[1] == 26
    from src.models.architectures import set_output_bias
    means = torch.linspace(0.05, 0.3, 13)
    set_output_bias(m, means)
    torch.testing.assert_close(torch.sigmoid(m.head.bias.detach()), means, atol=1e-5, rtol=0)


def test_handwritten_attention_matches_pytorch():
    blk = GlobalAttentionBlock(64, heads=4).eval()
    t = torch.rand(2, 37, 64)
    with torch.no_grad():
        q, k, v = blk.qkv(t).reshape(2, 37, 3, 4, 16).permute(2, 0, 3, 1, 4)
        ref = blk.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 37, 64))
        torch.testing.assert_close(blk.attention(t), ref, atol=1e-5, rtol=1e-4)
    pe = sinusoidal_2d(4, 5, 16)
    assert pe.shape == (20, 16) and torch.unique(pe, dim=0).shape[0] == 20   # every position is distinct


def test_eye_metrics_and_aligned_psnr():
    t = torch.rand(2, 13, 64, 64) * 0.3
    m = compute_metrics(t, t)
    assert np.allclose(m["sharpness"], 1) and m["colour_err"].max() < 1e-6 and m["detail_err"].max() < 1e-6
    blurred = compute_metrics(F.avg_pool2d(t, 5, 1, 2), t)
    assert blurred["sharpness"].max() < 0.6 and (blurred["detail_err"] > blurred["colour_err"]).all()
    shifted = compute_metrics(t * 0.8 + 0.05, t)                       # a perfect image with a date shift
    assert shifted["psnr_aligned"].min() > 60 and shifted["psnr"].max() < 40


def test_losses_are_differentiable():
    from src.training.loss import PerceptualLoss, SSIMLoss
    t = torch.rand(2, 13, 64, 64)
    p = (t + 0.1 * torch.randn_like(t)).clamp(0, 1).requires_grad_()
    (SSIMLoss()(p, t) + PerceptualLoss()(p, t)).backward()
    assert p.grad.abs().sum() > 0 and SSIMLoss()(t, t).item() < 1e-6


def test_global_split_keeps_phase1_roles():
    sc = pd.read_csv(os.path.join(ROOT, "data", "scans", "scenes.csv"))
    df = pd.DataFrame([dict(season=r.season, scene=int(r.scene), patch=p, country=r.country, continent=r.continent,
                            lon=r.lon, lat=r.lat) for r in sc.itertuples() for p in range(20)])
    p1 = pd.DataFrame(dict(season=sc.season[:6], scene=sc.scene[:6], split=["train", "val", "test_id", "ood", "train", "val"]))
    patches, scenes = make_global_splits(df, p1, n_val=10, n_test=15, caps=dict(train=3000), log=lambda *a: None)
    role = dict(zip(zip(scenes.season, scenes.scene), scenes.split))
    for r, want in zip(p1.itertuples(), ["train", "val", "test", "test", "train", "val"]):
        assert role[(r.season, r.scene)] == want
    assert (scenes.split == "test").sum() == 15 and (scenes.split == "val").sum() == 10
    assert scenes.groupby(["season", "scene"]).split.nunique().max() == 1
    assert set(scenes[scenes.split == "test"].continent) >= {"Asia", "Europe", "Africa"}
    tr = patches[patches.split == "train"].reset_index(drop=True).assign(valid=True)
    tr["cache_idx"] = np.arange(len(tr))
    a, na = scene_subset_indices(tr, 5, 60)
    b, nb = scene_subset_indices(tr, 10, 60)
    sa = set(zip(tr.loc[a, "season"], tr.loc[a, "scene"]))
    sb = set(zip(tr.loc[b, "season"], tr.loc[b, "scene"]))
    assert (na, nb) == (5, 10) and len(a) == len(b) == 60 and sa <= sb


def test_trainer_combo_ema_balanced(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    from src.training.trainer import RunConfig, lr_range_test, overfit_one_batch, train_run
    x, y = torch.rand(8, 15, 128, 128), torch.rand(8, 13, 128, 128)
    dl = DataLoader(TensorDataset(x, y, torch.arange(8)), batch_size=4, shuffle=True)
    cfg = RunConfig(name="t", arch="clearsar_net", model_kwargs=dict(output_mode="gated"), steps=4, val_every=2,
                    warmup=1, loss="combo", w_ssim=0.2, w_perc=0.05, ema_decay=0.9, select="balanced", eval_tile=128)
    info = train_run(cfg, dl, dl, torch.device("cpu"), str(tmp_path / "ck"), str(tmp_path / "lg"), log=lambda *a: None)
    assert info["best_score"] is not None and info["best_val_sharpness"] is not None
    h = pd.read_csv(tmp_path / "lg" / "history_t.csv")
    assert {"val_sharpness", "val_colour", "val_detail", "score"} <= set(h.columns)
    m, meta = load_model_from_checkpoint(str(tmp_path / "ck" / "best_t.pt"))
    assert meta["arch"] == "clearsar_net" and m.output_mode == "gated"
    curve = overfit_one_batch("clearsar_net", dict(output_mode="direct"), (x[:2], y[:2]), torch.device("cpu"), steps=30)
    assert curve[-1][1] < curve[0][1]
    rows, lr = lr_range_test("clearsar_net", dict(output_mode="direct"), dl, torch.device("cpu"), steps=8)
    assert len(rows) >= 2 and lr > 0


def test_data_study_oracle_is_an_upper_bound():
    from src.evaluation.data_study import patch_stats
    rng = np.random.default_rng(0)
    y = rng.random((13, 64, 64)).astype(np.float32) * 0.3
    x = np.concatenate([rng.random((2, 64, 64)), y * 0.7 + 0.04], 0).astype(np.float32)   # date shift, no cloud
    st = patch_stats(x, y)
    assert st["oracle_shift_psnr"] > st["clear_px_psnr"] + 10
