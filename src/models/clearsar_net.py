"""
ClearSAR-Net - the single model of plan v4 (docs/project_plan_v4.md).

A U-Net (same encoder/decoder as Model A) with three switchable ideas, each tested in its own stage:

  output_mode   how the output is formed from the network's features            (stage 2)
      "direct"    out = sigmoid(head(features))                     - regenerate every pixel
      "residual"  out = sigmoid(logit(cloudy) + head(features))     - predict a correction to the
                  cloudy image; the head starts at zero, so the untrained model returns the input
      "gated"     out = m * cloudy + (1 - m) * sigmoid(head(features)),  m = sigmoid(gate(features))
                  - a learned soft mask m decides, per pixel, whether to keep what the satellite saw
                  or to fill in; m is returned for visualisation (where does the model trust the input?)
      "hybrid"    out = m * sigmoid(logit(cloudy) + corr(features)) + (1 - m) * sigmoid(head(features))
                  - the gate of "gated", but the kept pixels also get a learned correction. The clear target
                    was photographed on a different date, so even cloud-free pixels need a small brightness /
                    colour shift; "gated" can only copy them. corr starts at zero (keep = copy at init).
                  The final model (notebooks/train_clearsar.py) uses this mode.
  upsample      "transpose" (2x2 transposed conv, Model A) or "bilinear" (bilinear resize + 3x3 conv). The
                  bilinear variant cannot produce the 2x2 block / checkerboard artefacts that a GAN
                  discriminator tends to amplify (Odena et al. 2016), so the final model uses it.
  attn_layers   number of global self-attention blocks on the bottleneck            (stage 5)
                  every bottleneck position can look at every other one - context from far away,
                  which Phase 1 showed matters under thick cloud
  uncertainty   adds 13 Laplace log-scale channels (as Model A's variance head)     (stage 6)

The attention is written out by hand (no library call) so it can be read and understood; a unit test
checks it against torch.nn.functional.scaled_dot_product_attention.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures import UNetBlock, _check_channels

N_S1 = 2   # the cloudy optical bands start after the two radar channels


def sinusoidal_2d(h, w, dim, device=None):
    """Fixed 2-D sine/cosine positional encoding [h*w, dim] (Vaswani et al. 2017, extended to 2-D):
    half the channels encode the row, half the column, at geometrically spaced frequencies. Unlike a
    learned embedding it works for any grid size, so the model can be trained on 128 crops and run
    on 256 patches."""
    assert dim % 4 == 0
    d = dim // 4
    freq = torch.exp(-math.log(10000.0) * torch.arange(d, device=device, dtype=torch.float32) / d)
    ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] * freq[None]     # [h, d]
    xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] * freq[None]     # [w, d]
    pe_y = torch.cat([ys.sin(), ys.cos()], -1)[:, None, :].expand(h, w, 2 * d)
    pe_x = torch.cat([xs.sin(), xs.cos()], -1)[None, :, :].expand(h, w, 2 * d)
    return torch.cat([pe_y, pe_x], -1).reshape(h * w, dim)


class GlobalAttentionBlock(nn.Module):
    """Pre-norm transformer block: x <- x + MHSA(LN(x)); x <- x + MLP(LN(x)).

    Multi-head self-attention, step by step:
      1. every token (bottleneck position) is projected to a query q, a key k and a value v
      2. scores = q · kᵀ / sqrt(d)  - how relevant every other token is to this one
      3. weights = softmax(scores)  - each row sums to 1
      4. output = weights · v       - a weighted average of all values
    Several heads do this in parallel on slices of the channels, then a linear layer mixes them."""

    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.d = heads, dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.keep_weights = False
        self.last_weights = None

    def attention(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, self.d).permute(2, 0, 3, 1, 4)   # [B, h, N, d]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)                            # [B, h, N, N]
        weights = scores.float().softmax(dim=-1).to(v.dtype)
        if self.keep_weights:
            self.last_weights = weights.detach()
        out = weights @ v                                                                 # [B, h, N, d]
        return self.proj(out.transpose(1, 2).reshape(B, N, C))

    def fast_attention(self, x):
        """The same computation through PyTorch's fused kernel (never builds the N x N matrix in memory).
        Used for training; tests check that it matches attention() above."""
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, self.d).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))

    def forward(self, x):
        # the residual stream x is kept in float32: under mixed precision only the matrix products run in fp16.
        # (An fp16 residual stream can grow past fp16's limit of 65 504 and turn into inf / NaN.)
        attn = self.attention if self.keep_weights else self.fast_attention
        x = x.float()
        x = x + attn(self.norm1(x)).float()
        return x + self.mlp(self.norm2(x)).float()


class ClearSARNet(nn.Module):
    def __init__(self, in_channels=15, out_channels=13, uncertainty=False, width=64, output_mode="direct",
                 attn_layers=0, attn_heads=8, attn_mlp_ratio=2.0, dropout=0.0, upsample="transpose"):
        super().__init__()
        _check_channels(in_channels, out_channels)
        if output_mode not in ("direct", "residual", "gated", "hybrid"):
            raise ValueError(f"output_mode must be direct, residual, gated or hybrid, got {output_mode!r}")
        if upsample not in ("transpose", "bilinear"):
            raise ValueError(f"upsample must be transpose or bilinear, got {upsample!r}")
        self.out_channels, self.uncertainty, self.output_mode = out_channels, uncertainty, output_mode
        w = width
        self.enc1, self.enc2, self.enc3 = UNetBlock(in_channels, w), UNetBlock(w, 2 * w), UNetBlock(2 * w, 4 * w)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = UNetBlock(4 * w, 8 * w)
        self.attn = nn.ModuleList([GlobalAttentionBlock(8 * w, attn_heads, attn_mlp_ratio) for _ in range(attn_layers)])
        # final LayerNorm (as in every ViT): the decoder always receives normalised features, whatever the size
        # the residual stream grows to. Without it the first decoder convolution can overflow in fp16.
        self.attn_norm = nn.LayerNorm(8 * w) if attn_layers else nn.Identity()
        def up(cin, cout):
            if upsample == "transpose":
                return nn.ConvTranspose2d(cin, cout, 2, stride=2)
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 nn.Conv2d(cin, cout, 3, padding=1))
        self.up3, self.dec3 = up(8 * w, 4 * w), UNetBlock(8 * w, 4 * w)
        self.up2, self.dec2 = up(4 * w, 2 * w), UNetBlock(4 * w, 2 * w)
        self.up1, self.dec1 = up(2 * w, w), UNetBlock(2 * w, w)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(w, out_channels, 1)
        if output_mode == "residual":          # start as the identity: correction = 0
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        if output_mode == "hybrid":            # correction of the kept pixels starts at zero
            self.corr = nn.Conv2d(w, out_channels, 1)
            nn.init.zeros_(self.corr.weight)
            nn.init.zeros_(self.corr.bias)
        if output_mode in ("gated", "hybrid"):  # start by mostly trusting the input (m ~ 0.88)
            self.gate = nn.Conv2d(w, 1, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 2.0)
        if uncertainty:                        # Laplace log-scale, starting at b ~ 0.03
            self.logscale = nn.Conv2d(w, out_channels, 1)
            nn.init.zeros_(self.logscale.weight)
            nn.init.constant_(self.logscale.bias, math.log(0.03))

    @torch.no_grad()
    def init_output_bias(self, band_means):
        """Start the generated image at the mean band reflectance (not for 'residual', which starts
        as the identity)."""
        if self.output_mode == "residual":
            return
        m = torch.as_tensor(band_means, dtype=torch.float32, device=self.head.bias.device).clamp(1e-3, 1 - 1e-3)
        self.head.bias.copy_(torch.log(m / (1 - m)))

    def features(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.drop(self.bottleneck(self.pool(e3)))
        if len(self.attn):
            B, C, H, W = b.shape
            t = b.float().flatten(2).transpose(1, 2) + sinusoidal_2d(H, W, C, b.device)[None]
            for blk in self.attn:
                t = blk(t)
            b = self.attn_norm(t).transpose(1, 2).reshape(B, C, H, W).to(b.dtype)
        d3 = self.drop(self.dec3(torch.cat([self.up3(b), e3], 1)))
        d2 = self.drop(self.dec2(torch.cat([self.up2(d3), e2], 1)))
        return self.dec1(torch.cat([self.up1(d2), e1], 1))

    def forward(self, x, return_aux=False):
        f = self.features(x)
        cloudy = x[:, N_S1:N_S1 + self.out_channels]
        aux = {}
        if self.output_mode == "direct":
            mean = torch.sigmoid(self.head(f))
        elif self.output_mode == "residual":
            c = cloudy.float().clamp(1e-3, 1 - 1e-3)
            mean = torch.sigmoid(torch.log(c / (1 - c)) + self.head(f).float()).to(f.dtype)
        elif self.output_mode == "gated":
            m = torch.sigmoid(self.gate(f))
            mean = m * cloudy + (1 - m) * torch.sigmoid(self.head(f))
            aux["gate"] = m
        else:
            m = torch.sigmoid(self.gate(f).float())
            c = cloudy.float().clamp(1e-3, 1 - 1e-3)
            keep = torch.sigmoid(torch.log(c / (1 - c)) + self.corr(f).float())
            mean = (m * keep + (1 - m) * torch.sigmoid(self.head(f).float())).to(f.dtype)
            aux["gate"] = m
        out = torch.cat([mean, self.logscale(f)], 1) if self.uncertainty else mean
        return (out, aux) if return_aux else out
