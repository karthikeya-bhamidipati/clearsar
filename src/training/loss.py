import torch
import torch.nn as nn


class UncertaintyL1Loss(nn.Module):
    """Laplace negative log-likelihood (up to a constant) for an aleatoric variance head:

        loss = |y - mu| / b + log b,   with   b = exp(s)

    `pred` is [B, 2C, H, W]: C mean channels then C log-scale channels s. Computed in
    float32 for AMP stability; s is clamped so exp(-s) cannot overflow.
    """

    def __init__(self, out_channels=13, s_min=-10.0, s_max=10.0):
        super().__init__()
        self.c, self.s_min, self.s_max = out_channels, s_min, s_max

    def forward(self, pred, target):
        pred, target = pred.float(), target.float()
        mu = pred[:, : self.c]
        s = pred[:, self.c:].clamp(self.s_min, self.s_max)
        return (torch.exp(-s) * (mu - target).abs() + s).mean()


# --------------------------------------------------------------------------------------
# Plan v4, stage 4: losses that reward structure and texture, not only per-pixel accuracy
# --------------------------------------------------------------------------------------
class SSIMLoss(nn.Module):
    """1 - SSIM (Gaussian window 11, sigma 1.5), averaged over all bands. SSIM compares local brightness,
    contrast and structure, so this loss punishes blur that L1 tolerates."""

    def forward(self, pred, target):
        from src.evaluation.metrics import ssim_per_image
        return 1.0 - ssim_per_image(pred.float().clamp(0, 1), target.float()).mean()


class PerceptualLoss(nn.Module):
    """Perceptual (VGG) loss, Johnson et al. 2016: compare images in the feature space of an ImageNet-trained
    VGG16 (layers relu1_2, relu2_2, relu3_3) instead of pixel space. Features respond to edges and textures,
    so matching them rewards realistic detail. Computed on the RGB bands (B4, B3, B2), brightened x2.5 and
    ImageNet-normalised. The VGG is frozen. Set CLEARSAR_PERC_PRETRAINED=0 to skip the weight download
    (tests only)."""

    RGB = (3, 2, 1)
    LAYERS = (3, 8, 15)           # indices of relu1_2, relu2_2, relu3_3 in torchvision's vgg16.features

    def __init__(self, pretrained=None):
        super().__init__()
        import os
        import torchvision
        if pretrained is None:
            pretrained = os.environ.get("CLEARSAR_PERC_PRETRAINED", "1") != "0"
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = torchvision.models.vgg16(weights=weights).features[: max(self.LAYERS) + 1].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _feats(self, x):
        x = (x[:, list(self.RGB)] * 2.5 - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.LAYERS:
                out.append(x)
        return out

    def train(self, mode=True):           # the VGG always stays in eval mode
        super().train(mode)
        self.vgg.eval()
        return self

    def forward(self, pred, target):
        fp, ft = self._feats(pred), self._feats(target)
        return sum((a - b).abs().mean() for a, b in zip(fp, ft)) / len(fp)
