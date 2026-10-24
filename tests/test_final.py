"""Tests for the final model pipeline (notebooks/train_clearsar.py, src/training/final_trainer.py):
hybrid output, bilinear decoder, spectral-norm PatchGAN, adaptive GAN weight, data echoing, schedules,
checkpoint round trip.   python -m pytest tests/test_final.py -q"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.architectures import build_model, load_model_from_checkpoint  # noqa: E402
from src.models.discriminator import SNConv2d, SNPatchDiscriminator  # noqa: E402
from src.training import final_trainer as ft  # noqa: E402

FINAL = dict(output_mode="hybrid", attn_layers=2, width=16, upsample="bilinear", attn_heads=4)


def test_hybrid_starts_as_soft_copy_and_trains_everything():
    m = build_model("clearsar_net", **FINAL)
    m.init_output_bias([0.1] * 13)
    x = torch.rand(2, 15, 64, 64)
    m.eval()
    out, aux = m(x, return_aux=True)
    g = torch.sigmoid(torch.tensor(2.0))
    expect = g * x[:, 2:].clamp(1e-3, 1 - 1e-3) + (1 - g) * 0.1
    assert out.shape == (2, 13, 64, 64) and aux["gate"].shape == (2, 1, 64, 64)
    assert torch.allclose(out, expect, atol=1e-3)
    m.train()
    out = m(x)
    (out - torch.rand_like(out)).abs().mean().backward()
    m(x).mean().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_spectral_norm_bounds_the_layer():
    c = SNConv2d(8, 16, 3, padding=1)
    for _ in range(50):
        c(torch.rand(1, 8, 8, 8))
    w = c.weight.reshape(16, -1)
    assert abs(float(c._sigma()) - float(torch.linalg.matrix_norm(w, 2))) < 1e-3
    d = SNPatchDiscriminator()
    assert d(torch.rand(2, 15, 256, 256), torch.rand(2, 13, 256, 256)).shape == (2, 1, 30, 30)


def test_adaptive_lambda_balances_gradients():
    m = build_model("clearsar_net", **FINAL)
    d = SNPatchDiscriminator(base=8)
    x, y = torch.rand(2, 15, 64, 64), torch.rand(2, 13, 64, 64)
    p = m(x)
    rec = ft.reconstruction_loss(p, y, 0.05)[0]
    adv = ft.lsgan(d(x, p), 1.0)
    w = m.dec1.conv[3].weight
    lam = ft.adaptive_lambda(rec, adv, w, factor=0.2)
    g_rec, = torch.autograd.grad(rec, w, retain_graph=True)
    g_adv, = torch.autograd.grad(lam * adv, w)
    assert abs(g_adv.norm() / g_rec.norm() - 0.2) < 1e-3


def test_learning_rate_schedule():
    cfg = dict(gan_start=0.5, warmup_steps=10, lr=3e-4, lr_floor=0.05, lr_gan=5e-5, lr_d=2e-4)
    assert ft.lr_g(cfg, 0.0, 0) < 1e-4                         # warm-up
    assert abs(ft.lr_g(cfg, 0.0, 100) - 3e-4) < 1e-9
    assert abs(ft.lr_g(cfg, 0.5 - 1e-9, 100) - 3e-4 * 0.05) < 1e-7
    assert abs(ft.lr_g(cfg, 0.5, 100) - 5e-5) < 1e-9          # part B restarts at the GAN learning rate
    assert ft.lr_g(cfg, 1.0, 100) < 1e-6 and ft.lr_d(cfg, 1.0) < 3e-6


def test_dihedral_keeps_input_and_target_aligned():
    x = torch.rand(6, 15, 16, 16)
    y = x[:, 2:].clone()
    xa, ya = ft.dihedral(x, y)
    assert torch.equal(xa[:, 2:], ya)


class _SlowLoader:
    def __init__(self, n, delay):
        self.n, self.delay = n, delay

    def __iter__(self):
        for _ in range(self.n):
            time.sleep(self.delay)
            yield torch.rand(4, 15, 8, 8).half(), torch.rand(4, 13, 8, 8).half(), torch.arange(4)


def test_echo_feeder_reuses_data_but_never_starves():
    f = ft.EchoFeeder(_SlowLoader(1000, 0.05), None, torch.device("cpu"), pool_size=16, max_echo=3)
    kinds = [f.next(4)[2] for _ in range(40)]
    f.close()
    assert kinds[0] is True and any(not k for k in kinds)
    run = longest = 0
    for k in kinds:
        run = 0 if k else run + 1
        longest = max(longest, run)
    assert longest <= 3


def test_checkpoint_round_trip(tmp_path):
    m = build_model("clearsar_net", **FINAL).eval()
    p = str(tmp_path / "g.pt")
    ft.save_generator(p, m, dict(model=FINAL), dict(step=5))
    m2, ck = load_model_from_checkpoint(p)
    x = torch.rand(1, 15, 32, 32)
    assert ck["step"] == 5 and torch.allclose(m(x), m2(x))


def test_stream_dataset_drops_unreadable_files():
    ds = ft.TripletStream([dict(s1="missing.tif", s2="missing.tif", s2_cloudy="missing.tif")])
    assert ds[0][3] is False and ft.stream_collate([ds[0]]) is None


def test_bottleneck_stays_bounded_when_attention_grows():
    """Root cause of the first full run's NaNs: an unnormalised attention residual stream feeding the decoder
    overflowed fp16. The final LayerNorm must keep the decoder input normalised whatever the stream size."""
    m = build_model("clearsar_net", **FINAL).eval()
    with torch.no_grad():
        for blk in m.attn:
            blk.mlp[2].weight.mul_(1e4)
            blk.mlp[2].bias.fill_(1e3)
    seen = {}
    m.up3.register_forward_hook(lambda mod, inp, out: seen.update(x=inp[0]))
    out = m(torch.rand(2, 15, 64, 64))
    assert torch.isfinite(out).all()
    assert seen["x"].abs().max() < 50 and seen["x"].dtype == torch.float32
