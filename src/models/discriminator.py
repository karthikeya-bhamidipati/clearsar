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
