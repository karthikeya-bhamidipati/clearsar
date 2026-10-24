import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN (Pix2Pix). Scores (input, output) pairs patch-by-patch as real/fake.

    in_channels = generator input channels + generator output channels
    (default 15 + 13 = 28). Returns raw logits; use BCEWithLogitsLoss.
    """

    def __init__(self, in_channels=28):
        super().__init__()

        def block(in_c, out_c, normalize=True, stride=2):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


# --------------------------------------------------------------------------------------
# Final model: spectrally normalised PatchGAN (notebooks/train_clearsar.py)
# --------------------------------------------------------------------------------------
class SNConv2d(nn.Conv2d):
    """Conv2d with spectral normalisation (Miyato et al. 2018): the weight is divided by its largest
    singular value sigma, so the layer can never stretch its input by more than a factor 1. A discriminator
    built from such layers changes smoothly, which keeps the generator's gradients well behaved.
    sigma is estimated with one power-iteration step per training forward pass (u is a persistent
    estimate of the top singular vector). The estimate is done in float32 even under mixed precision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        u = torch.randn(self.out_channels)
        self.register_buffer("u", u / u.norm())

    def _sigma(self):
        w = self.weight.float().reshape(self.out_channels, -1)
        with torch.no_grad():
            if self.training:
                v = torch.nn.functional.normalize(w.t() @ self.u, dim=0, eps=1e-12)
                self.u.copy_(torch.nn.functional.normalize(w @ v, dim=0, eps=1e-12))
            v = torch.nn.functional.normalize(w.t() @ self.u, dim=0, eps=1e-12)
        return self.u.clone() @ (w @ v)          # clone: u is updated in place by the next pass

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            w = self.weight / self._sigma()
        return self._conv_forward(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class SNPatchDiscriminator(nn.Module):
    """70x70 PatchGAN (Isola et al. 2017) with spectral normalisation instead of batch norm.

    It is CONDITIONAL: it sees the generator input x (radar + cloudy image, 15 channels) next to a
    candidate clear image y (13 channels), and outputs a 30x30 map (for 256x256 inputs) of raw scores -
    one per overlapping 70x70 window: "is this window of y a plausible clear image for this x?".
    Judging windows instead of the whole image makes it a texture / sharpness critic, which is exactly
    what L1 training lacks. Trained with the least-squares (LSGAN) loss: real -> 1, fake -> 0."""

    def __init__(self, in_channels=28, base=64):
        super().__init__()
        b = base
        self.model = nn.Sequential(
            SNConv2d(in_channels, b, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(b, 2 * b, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(2 * b, 4 * b, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(4 * b, 8 * b, 4, stride=1, padding=1), nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(8 * b, 1, 4, stride=1, padding=1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y.to(x.dtype)], dim=1))
