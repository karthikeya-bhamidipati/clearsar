import torch
import torch.nn as nn

class UncertaintyL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mu = pred[:, :3, :, :]
        log_var = pred[:, 3:, :, :]
        loss = torch.exp(-log_var) * torch.abs(mu - target) + log_var
        return loss.mean()