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
