"""
The three generator architectures compared in ClearSAR (see docs/project_plan.md §3).

All three map the same input to the same output:
    input : [B, in_channels, H, W]   (default 15 = VV, VH + 13 cloudy S2 bands, all in [0, 1])
    output: [B, out_channels, H, W]  (default 13 clear S2 bands in [0, 1], via sigmoid)

With `uncertainty=True` the output has 2 * out_channels channels:
    [:, :out_channels]  predicted mean (sigmoid, in [0, 1])
    [:, out_channels:]  predicted log-scale s = log(b) of a Laplace distribution (unbounded)

Training uses 128x128 crops; evaluation tiles 256x256 patches into 128x128 pieces, so every
model always sees the size it was trained on (H, W must be multiples of 8 for A/B and of the
patch size for C).

Always construct models with keyword arguments, e.g. ModelA_UNet(in_channels=15, out_channels=13).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_channels(in_channels, out_channels):
    # Guards against the positional-argument bug ModelA_UNet(False) -> in_channels=0.
    for name, v in (("in_channels", in_channels), ("out_channels", out_channels)):
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError(f"{name} must be a positive int, got {v!r}. "
                             "Construct models with keyword arguments.")


def _split_output(out, out_channels, uncertainty):
    if not uncertainty:
        return torch.sigmoid(out)
    return torch.cat([torch.sigmoid(out[:, :out_channels]), out[:, out_channels:]], dim=1)


# --------------------------------------------------------------------------------------
# Model A - convolutional U-Net
# --------------------------------------------------------------------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ModelA_UNet(nn.Module):
    """3-level U-Net (64-128-256, bottleneck 512) with skip connections.

    use_skips=False replaces each skip tensor with zeros (E3a ablation) so the parameter
    count stays identical and only the information flow changes.
    """

    def __init__(self, in_channels=15, out_channels=13, uncertainty=False, use_skips=True):
        super().__init__()
        _check_channels(in_channels, out_channels)
        self.out_channels = out_channels
        self.uncertainty = uncertainty
        self.use_skips = use_skips
        n_out = out_channels * (2 if uncertainty else 1)

        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = UNetBlock(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        self.final = nn.Conv2d(64, n_out, kernel_size=1)

    def _skip(self, e):
        return e if self.use_skips else torch.zeros_like(e)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        return e1, e2, e3

    def middle(self, e3):
        return self.bottleneck(self.pool(e3))

    def decode(self, b, e1, e2, e3):
        d3 = self.dec3(torch.cat([self.up3(b), self._skip(e3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self._skip(e2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self._skip(e1)], dim=1))
        return self.final(d1)

    def forward(self, x):
        e1, e2, e3 = self.encode(x)
        out = self.decode(self.middle(e3), e1, e2, e3)
        return _split_output(out, self.out_channels, self.uncertainty)


# --------------------------------------------------------------------------------------
# Model B - U-Net with a Swin-Transformer bottleneck
# --------------------------------------------------------------------------------------
def window_partition(x, ws):
    """[B, H, W, C] -> [B * nW, ws * ws, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def window_reverse(windows, ws, B, H, W):
    """[B * nW, ws * ws, C] -> [B, H, W, C]"""
    C = windows.shape[-1]
    x = windows.view(B, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, C)


def shifted_window_mask(H, W, ws, shift, device):
    """Attention mask for shifted windows: tokens that came from different image regions
    (because of the cyclic roll) must not attend to each other. Returns [nW, N, N] with 0 / -100."""
    img = torch.zeros(1, H, W, 1, device=device)
    cnt = 0
    for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
        for wsl in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            img[:, hs, wsl, :] = cnt
            cnt += 1
    win = window_partition(img, ws).squeeze(-1)                # [nW, N]
    mask = win.unsqueeze(1) - win.unsqueeze(2)                 # [nW, N, N]
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


class WindowAttention(nn.Module):
    """Multi-head self-attention inside a window, with a learned relative position bias."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.ws, self.heads = window_size, num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size),
                                            indexing="ij")).flatten(1)          # [2, N]
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (window_size - 1)
        self.register_buffer("rel_index", rel[..., 0] * (2 * window_size - 1) + rel[..., 1],
                             persistent=False)                                    # [N, N]

    def forward(self, x, mask=None):
        Bw, N, C = x.shape
        q, k, v = self.qkv(x).view(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        bias = self.rel_bias[self.rel_index.reshape(-1)].view(N, N, self.heads).permute(2, 0, 1)
        attn_bias = bias.unsqueeze(0)                                             # [1, h, N, N]
        if mask is not None:
            nW = mask.shape[0]
            attn_bias = (attn_bias.unsqueeze(1) + mask.view(1, nW, 1, N, N))     # [1, nW, h, N, N]
            attn_bias = attn_bias.expand(Bw // nW, -1, -1, -1, -1).reshape(Bw, self.heads, N, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.to(q.dtype))
        return self.proj(out.transpose(1, 2).reshape(Bw, N, C))


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift, mlp_ratio=4.0):
        super().__init__()
        self.ws, self.shift = window_size, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):                                   # x: [B, H, W, C]
        B, H, W, C = x.shape
        ws = min(self.ws, H, W)
        shift = self.shift if ws < min(H, W) else 0          # no shift if one window covers all
        if H % ws or W % ws:
            raise ValueError(f"Swin bottleneck {H}x{W} is not divisible by window {ws}")
        h = self.norm1(x)
        mask = None
        if shift:
            h = torch.roll(h, shifts=(-shift, -shift), dims=(1, 2))
            mask = shifted_window_mask(H, W, ws, shift, x.device)
        h = window_reverse(self.attn(window_partition(h, ws), mask), ws, B, H, W)
        if shift:
            h = torch.roll(h, shifts=(shift, shift), dims=(1, 2))
        x = x + h
        return x + self.mlp(self.norm2(x))


class ModelB_SwinUNet(ModelA_UNet):
    """Model A's encoder and decoder; the conv bottleneck is replaced by Swin blocks at
    H/8 x W/8 (16x16 tokens for a 128 crop), followed by a 1x1 projection 256 -> 512 so the
    decoder is unchanged.

    shift=False and window_size are the E5 probes (window 16 = global attention at 16x16).
    """

    def __init__(self, in_channels=15, out_channels=13, uncertainty=False, use_skips=True,
                 depth=4, num_heads=8, window_size=4, shift=True):
        super().__init__(in_channels, out_channels, uncertainty, use_skips)
        del self.bottleneck
        dim = 256
        self.swin = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size, shift=(window_size // 2 if shift and i % 2 else 0))
            for i in range(depth)
        ])
        self.swin_norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Conv2d(dim, 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

    def middle(self, e3):
        t = self.pool(e3).permute(0, 2, 3, 1)               # [B, H/8, W/8, 256]
        for blk in self.swin:
            t = blk(t)
        t = self.swin_norm(t).permute(0, 3, 1, 2).contiguous()
        return self.proj(t)


# Backwards-compatible name used in older notebooks / app labels.
ModelB_HybridUNet = ModelB_SwinUNet


# --------------------------------------------------------------------------------------
# Model C - pure ViT encoder-decoder (no convolutions)
# --------------------------------------------------------------------------------------
class ModelC_PureViT(nn.Module):
    """Patchify (linear) -> + positional embedding -> transformer encoder -> transformer
    decoder layers -> linear unpatchify. No convolutions anywhere.

    use_pos_emb=False is the E6a probe (attention becomes permutation-invariant);
    patch_size is the E6b probe.
    """

    def __init__(self, in_channels=15, out_channels=13, uncertainty=False, img_size=128,
                 patch_size=8, dim=192, depth=8, decoder_depth=2, num_heads=3, mlp_ratio=4.0,
                 use_pos_emb=True):
        super().__init__()
        _check_channels(in_channels, out_channels)
        self.out_channels = out_channels
        self.uncertainty = uncertainty
        self.p = patch_size
        self.n_out = out_channels * (2 if uncertainty else 1)
        self.grid = img_size // patch_size

        self.embed = nn.Linear(in_channels * patch_size * patch_size, dim)
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

        def stack(n):
            layer = nn.TransformerEncoderLayer(dim, num_heads, int(dim * mlp_ratio), dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, n, enable_nested_tensor=False)

        self.encoder = stack(depth)
        self.decoder = stack(decoder_depth)
        self.norm = nn.LayerNorm(dim)
        self.unpatch = nn.Linear(dim, self.n_out * patch_size * patch_size)

    def _pos(self, gh, gw):
        pe = self.pos_emb
        if gh == self.grid and gw == self.grid:
            return pe
        # Different input size (e.g. untiled inference): interpolate the learned grid.
        pe = pe.reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(gh, gw), mode="bicubic", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.p
        if H % p or W % p:
            raise ValueError(f"Input {H}x{W} not divisible by patch size {p}")
        gh, gw = H // p, W // p
        tokens = x.reshape(B, C, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(B, gh * gw, C * p * p)
        t = self.embed(tokens)
        if self.use_pos_emb:
            t = t + self._pos(gh, gw)
        t = self.norm(self.decoder(self.encoder(t)))
        out = self.unpatch(t).reshape(B, gh, gw, self.n_out, p, p)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(B, self.n_out, H, W)
        return _split_output(out, self.out_channels, self.uncertainty)


# --------------------------------------------------------------------------------------
ARCHITECTURES = {
    "unet": ModelA_UNet,
    "swin_unet": ModelB_SwinUNet,
    "vit": ModelC_PureViT,
}


def build_model(arch, in_channels=15, out_channels=13, uncertainty=False, **kwargs):
    if arch not in ARCHITECTURES:
        raise KeyError(f"Unknown arch {arch!r}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](in_channels=in_channels, out_channels=out_channels,
                               uncertainty=uncertainty, **kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_from_checkpoint(path, map_location="cpu"):
    """Rebuild a generator from a checkpoint written by src/training/trainer.py."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ckpt["arch"], ckpt["in_channels"], ckpt["out_channels"],
                        ckpt["uncertainty"], **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
