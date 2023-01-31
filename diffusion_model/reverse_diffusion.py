import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from functools import partial

import torchinfo


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x: torch.Tensor):
        epsilon = 1e-5 if x.dtype == torch.float32 else 1e-3

        mu = einops.reduce(self.weight, "c ... -> c 1 1 1", "mean")
        sigma = einops.reduce(self.weight, "c ... -> c 1 1 1", partial(torch.std, unbiased = False))

        normalized_weight = (self.weight - mu) / (sigma + epsilon)

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, c_mid: int = None, kernel_size: int = 3, stride: int = 1, padding: int = 1, residual: bool = False):
        super().__init__()

        self.residual = residual

        if not c_mid:
            c_mid = c_out

        self.conv_block = nn.Sequential(
            WeightStandardizedConv2d(c_in, c_mid, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=c_mid),
            nn.GELU(),
            WeightStandardizedConv2d(c_mid, c_out, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=c_out)
        )

    def forward(self, x: torch.Tensor):
        if self.residual:
            return F.gelu(x + self.conv_block(x))
        else:
            return self.conv_block(x)


class DownBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(c_in, c_in, residual=True),
            DoubleConv(c_in, c_out)
        )

        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, c_out)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.conv(x)
        emb = self.time_embedding(t)[:, :, None, None]

        return x + emb


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int = 256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = nn.Sequential(
            DoubleConv(c_in, c_in, residual=True),
            DoubleConv(c_in, c_out, c_mid=c_in//2)
        )

        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, c_out)
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor, t: torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv(x)
        emb = self.time_embedding(t)[:, :, None, None]

        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.ln = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.shape

        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x_ln = self.ln(x)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        attn = attn + x
        out = self.feedforward(attn) + attn
        out = einops.rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        return out



class UNet(nn.Module):
    def __init__(self, c_in: int = 3, c_out: int = 3, time_dim: int = 256):
        super().__init__()

        self.time_dim = time_dim

        # initial block
        self.initial = DoubleConv(c_in, 64)

        # down blocks
        self.down1 = DownBlock(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = DownBlock(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = DownBlock(256, 256)
        self.sa3 = SelfAttention(256)

        # bottleneck
        self.bttln1 = DoubleConv(256, 512)
        self.bttln2 = DoubleConv(512, 512)
        self.bttln3 = DoubleConv(512, 256)

        # up blocks
        self.up1 = UpBlock(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = UpBlock(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = UpBlock(128, 64)
        self.sa6 = SelfAttention(64)

        # proj block
        self.out = WeightStandardizedConv2d(64, 3, kernel_size=1)

    def positional_encoding(self, t: torch.Tensor, channels: int):
        t = t.unsqueeze(-1).float()

        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        
        pos_enc_1 = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_2 = torch.cos(t.repeat(1, channels // 2) * inv_freq)

        return torch.cat([pos_enc_1, pos_enc_2], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # shapes
        # x: [batch_size, channels, height, width]
        # t: [batch_size, ]
        t = self.positional_encoding(t, self.time_dim)

        x1 = self.initial(x)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)

        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bttln1(x4)
        x4 = self.bttln2(x4)
        x4 = self.bttln3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)

        x = self.up2(x, x2, t)
        x = self.sa5(x)

        x = self.up3(x, x1, t)
        x = self.sa6(x)

        return self.out(x)


if __name__ == "__main__":
    unet = UNet()

    torchinfo.summary(unet, [(32, 3, 64, 64), (32,)], dtypes=[torch.float, torch.long])