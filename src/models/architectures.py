import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

class ModelA_UNet(nn.Module):
    def __init__(self, in_channels=6, uncertainty=False):
        super().__init__()
        self.out_ch = 6 if uncertainty else 3
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
        self.final = nn.Conv2d(64, self.out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.final(d1)
        if self.out_ch == 3: 
            return torch.tanh(out)
        return torch.cat([torch.tanh(out[:, :3]), out[:, 3:]], dim=1)

class ModelB_HybridUNet(ModelA_UNet):
    def __init__(self, in_channels=6, uncertainty=False):
        super().__init__(in_channels, uncertainty)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b_cnn = self.pool(e3) # [B, 256, H/8, W/8]
        B, C, H, W = b_cnn.shape
        b_flat = b_cnn.flatten(2).transpose(1, 2)
        b_trans = self.transformer(b_flat)
        b_out = b_trans.transpose(1, 2).view(B, C, H, W)
        
        b = self.bottleneck(b_out)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.final(d1)
        if self.out_ch == 3: 
            return torch.tanh(out)
        return torch.cat([torch.tanh(out[:, :3]), out[:, 3:]], dim=1)

class ModelC_PureViT(nn.Module):
    def __init__(self, in_channels=6, uncertainty=False):
        super().__init__()
        self.out_ch = 6 if uncertainty else 3
        self.patch_embed = nn.Conv2d(in_channels, 256, kernel_size=16, stride=16)
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=6)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=4), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.out_ch, 3, padding=1)
        )

    def forward(self, x):
        p = self.patch_embed(x)
        B, C, H, W = p.shape
        p_flat = p.flatten(2).transpose(1, 2)
        t_out = self.transformer(p_flat)
        t_img = t_out.transpose(1, 2).view(B, C, H, W)
        out = self.upsample(t_img)
        if self.out_ch == 3: 
            return torch.tanh(out)
        return torch.cat([torch.tanh(out[:, :3]), out[:, 3:]], dim=1)