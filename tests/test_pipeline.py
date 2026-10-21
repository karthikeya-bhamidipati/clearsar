"""Unit tests: python -m pytest tests -q   (CPU only, ~1 minute)."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.dataset import (IN_CHANNELS, OUT_CHANNELS, attach_scene_metadata, index_triplets,  # noqa: E402
                              make_splits, preprocess_s1, preprocess_s2)
from src.evaluation.evaluate import predict_tiled  # noqa: E402
from src.evaluation.metrics import compute_metrics, sparsification  # noqa: E402
from src.models.architectures import (ModelA_UNet, build_model, count_parameters,  # noqa: E402
                                      window_partition, window_reverse)
from src.models.discriminator import PatchGANDiscriminator  # noqa: E402
from src.training.loss import UncertaintyL1Loss  # noqa: E402


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("arch", ["unet", "swin_unet", "vit", "tiny_cnn"])
@pytest.mark.parametrize("unc", [False, True])
def test_model_shapes_and_range(arch, unc):
    m = build_model(arch, IN_CHANNELS, OUT_CHANNELS, uncertainty=unc).eval()
    x = torch.rand(2, IN_CHANNELS, 128, 128)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, OUT_CHANNELS * (2 if unc else 1), 128, 128)
    assert 0 <= y[:, :OUT_CHANNELS].min() and y[:, :OUT_CHANNELS].max() <= 1
    print(arch, unc, f"{count_parameters(m) / 1e6:.2f}M")


def test_backward_all_models():
    for arch in ["unet", "swin_unet", "vit"]:
        m = build_model(arch, IN_CHANNELS, OUT_CHANNELS)
        loss = m(torch.rand(2, IN_CHANNELS, 128, 128)).mean()
        loss.backward()
        assert all(p.grad is not None for p in m.parameters() if p.requires_grad), arch


def test_positional_bool_bug_is_caught():
    with pytest.raises(ValueError):
        ModelA_UNet(False)          # the exact call that crashed on Kaggle


def test_vit_other_sizes_and_ablations():
    m = build_model("vit", use_pos_emb=False).eval()
    assert m(torch.rand(1, IN_CHANNELS, 128, 128)).shape[-1] == 128
    m = build_model("vit").eval()
    assert m(torch.rand(1, IN_CHANNELS, 256, 256)).shape[-1] == 256     # interpolated pos-emb
    m = build_model("swin_unet", shift=False, window_size=8).eval()
    assert m(torch.rand(1, IN_CHANNELS, 128, 128)).shape[-1] == 128
    m = build_model("unet", use_skips=False).eval()
    assert m(torch.rand(1, IN_CHANNELS, 128, 128)).shape[-1] == 128
    m = build_model("vit", patch_size=16).eval()
    assert m(torch.rand(1, IN_CHANNELS, 128, 128)).shape[-1] == 128
    m = build_model("swin_unet", window_size=16).eval()
    assert m(torch.rand(1, IN_CHANNELS, 128, 128)).shape[-1] == 128


def test_window_roundtrip():
    x = torch.rand(2, 16, 16, 8)
    assert torch.equal(window_reverse(window_partition(x, 4), 4, 2, 16, 16), x)


def test_discriminator():
    d = PatchGANDiscriminator(in_channels=IN_CHANNELS + OUT_CHANNELS)
    out = d(torch.rand(2, IN_CHANNELS, 128, 128), torch.rand(2, OUT_CHANNELS, 128, 128))
    assert out.shape[:2] == (2, 1)


def test_uncertainty_loss_finite():
    pred = torch.cat([torch.rand(2, 13, 8, 8), torch.full((2, 13, 8, 8), -50.0)], 1)
    assert torch.isfinite(UncertaintyL1Loss(13)(pred, torch.rand(2, 13, 8, 8)))


# ---------------------------------------------------------------- preprocessing
def test_s1_db_input_is_not_destroyed():
    # Regression: the old code applied 10*log10 to values that were already dB, which
    # turned every negative value into a constant -1 and erased the SAR signal.
    db = np.stack([np.random.uniform(-25, 0, (32, 32)), np.random.uniform(-32, 0, (32, 32))]).astype(np.float32)
    out = preprocess_s1(db)
    assert out.min() >= 0 and out.max() <= 1
    assert out.std() > 0.1
    np.testing.assert_allclose(out[0], (db[0] + 25) / 25, atol=1e-5)
    lin = 10 ** (db / 10)                                   # linear input gets converted
    np.testing.assert_allclose(preprocess_s1(lin), out, atol=1e-4)
    assert np.isfinite(preprocess_s1(db, apply_lee=True)).all()


def test_s2_scaling():
    a = np.array([[[-5, 0, 5000, 20000, np.nan]]], dtype=np.float32)
    np.testing.assert_allclose(preprocess_s2(a), [[[0, 0, 0.5, 1, 0]]])


# ---------------------------------------------------------------- metrics
def test_metrics_match_skimage():
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    rng = np.random.default_rng(0)
    t = rng.random((2, 13, 64, 64)).astype(np.float32)
    p = np.clip(t + rng.normal(0, 0.05, t.shape), 0, 1).astype(np.float32)
    m = compute_metrics(torch.from_numpy(p), torch.from_numpy(t))
    for i in range(2):
        ref_ssim = structural_similarity(p[i], t[i], channel_axis=0, data_range=1.0, gaussian_weights=True,
                                         sigma=1.5, use_sample_covariance=False)
        assert abs(m["ssim"][i] - ref_ssim) < 1e-3
        assert abs(m["psnr"][i] - peak_signal_noise_ratio(t[i], p[i], data_range=1.0)) < 1e-3
    same = compute_metrics(torch.from_numpy(t), torch.from_numpy(t))
    assert same["sam"].max() < 0.1 and same["ssim"].min() > 0.999


def test_sparsification_oracle():
    e = np.random.rand(1000)
    assert abs(sparsification(e, e)["ause"]) < 1e-9          # perfect ranking -> AUSE 0


# ---------------------------------------------------------------- tiling
def test_predict_tiled_identity():
    x = torch.rand(2, 5, 256, 256)
    torch.testing.assert_close(predict_tiled(torch.nn.Identity(), x), x)
    x = torch.rand(1, 5, 200, 140)
    torch.testing.assert_close(predict_tiled(torch.nn.Identity(), x), x)


# ---------------------------------------------------------------- indexing / splitting
def test_index_and_split(tmp_path):
    subprocess.run([sys.executable, os.path.join(ROOT, "tests", "make_synthetic_data.py"),
                    "--out", str(tmp_path), "--patches", "3"], check=True)
    df = index_triplets([str(tmp_path)], log=lambda *a: None)
    assert len(df) == 12 * 3
    df = attach_scene_metadata(df, os.path.join(ROOT, "data", "scans", "scenes.csv"), log=lambda *a: None)
    patches, scenes = make_splits(df, log=lambda *a: None)
    # no scene appears in two splits
    assert scenes.groupby(["season", "scene"]).split.nunique().max() == 1
    assert set(scenes.split) == {"train", "val", "test_id", "ood"}
    assert "IN" in set(scenes[scenes.split == "test_id"].country)
    assert set(scenes[scenes.split == "ood"].continent) <= {"Europe", "Africa", "North America", "South America"}
    assert (scenes[scenes.split != "ood"].continent == "Asia").all()


# ---------------------------------------------------------------- experiments support
def test_fraction_subsets_are_nested_and_random_split_disjoint():
    import pandas as pd
    from src.data.dataset import fraction_indices, random_patch_split
    df = pd.DataFrame(dict(season=["ROIs1158_spring"] * 40 + ["ROIs1868_summer"] * 40,
                           scene=[1] * 40 + [2] * 40, cache_idx=range(80), valid=True))
    a, b, c = (set(fraction_indices(df, f)) for f in (0.25, 0.5, 1.0))
    assert a < b < c and len(a) == 20 and len(c) == 80
    parts = random_patch_split({"x": np.arange(30), "y": np.arange(20)}, {"tr": 30, "va": 10, "te": 10})
    flat = [i for v in parts.values() for i in v]
    assert len(flat) == len(set(flat)) == 50


def test_ensemble_mc_dropout_and_erf():
    from src.evaluation.analysis import effective_receptive_field, erf_stats
    from src.evaluation.evaluate import evaluate
    from torch.utils.data import TensorDataset, DataLoader
    x, y = torch.rand(4, IN_CHANNELS, 128, 128), torch.rand(4, OUT_CHANNELS, 128, 128)
    loader = DataLoader(TensorDataset(x, y, torch.arange(4)), batch_size=2)
    models = [build_model("tiny_cnn").eval() for _ in range(2)]
    res, ex = evaluate(models, loader, torch.device("cpu"), collect={1})
    assert ex["unc_kind"] == "gaussian" and len(res["psnr"]) == 4 and 1 in ex["samples"]
    m = build_model("unet", dropout=0.2)
    res, ex = evaluate(m, loader, torch.device("cpu"), mc_passes=3)
    assert ex["unc_sigmas"].mean() > 0 and not m.training
    erf = effective_receptive_field(build_model("tiny_cnn"), x[:2])
    st = erf_stats(erf)
    assert abs(erf.sum() - 1) < 1e-4 and st["radius50"] <= st["radius90"] <= 4.5   # 3 convs -> radius 3
