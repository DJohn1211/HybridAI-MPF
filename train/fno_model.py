"""
fno_model.py

FNO2dEnhanced:
- in_channels/out_channels configurable via FNOConfig
- supports time-window inputs: (C = n_phys*T_in + 3 grid channels)
- optional Residual U-Net correction head to capture multiscale local details

Notes:
- FFT is computed in fp32 for stability under AMP.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Spectral conv (2D)
# =========================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        # complex weights stored as (real, imag)
        scale = 1.0 / (in_channels * out_channels)
        self.weights1r = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weights1i = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weights2r = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weights2i = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    @staticmethod
    def compl_mul2d(input_ft: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        w = torch.complex(wr, wi)
        return torch.einsum("bixy,ioxy->boxy", input_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) can be fp16/bf16 under AMP. FFT done in fp32.
        """
        B, C, H, W = x.shape
        orig_dtype = x.dtype
        x_f = x.float()
        x_ft = torch.fft.rfft2(x_f, norm="ortho")

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            device=x.device, dtype=torch.complex64
        )

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights1r[:, :, :m1, :m2],
            self.weights1i[:, :, :m1, :m2],
        )

        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2r[:, :, :m1, :m2],
            self.weights2i[:, :, :m1, :m2],
        )

        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x_out.to(orig_dtype)


# =========================
# Small MLP conv blocks
# =========================

class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Conv2d(width, width, 1)
        self.fc2 = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


# =========================
# Residual U-Net (multi-scale refinement)
# =========================

def _pick_gn_groups(channels: int, desired: int = 8) -> int:
    for g in (desired, 16, 8, 4, 2, 1):
        if g <= channels and channels % g == 0:
            return g
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, gn_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g1 = _pick_gn_groups(out_ch, gn_groups)
        g2 = _pick_gn_groups(out_ch, gn_groups)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g1, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g2, out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, gn_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, gn_groups=gn_groups, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, gn_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        # in_ch is channels of concatenated skip + upsampled
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, gn_groups=gn_groups, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Pad to same spatial size if needed (odd sizes)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffX != 0 or diffY != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResidualUNet(nn.Module):
    """
    Predict a residual correction Δy from (pred_coarse, x_original).
    Depth controls number of downsamplings.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        base_channels: int = 32,
        depth: int = 4,
        gn_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        depth = int(depth)
        if depth < 2:
            raise ValueError("ResidualUNet depth must be >= 2")

        self.inc = DoubleConv(in_channels, base_channels, gn_groups=gn_groups, dropout=dropout)

        # Encoder
        enc: List[nn.Module] = []
        ch = base_channels
        for _ in range(depth - 1):
            enc.append(Down(ch, ch * 2, gn_groups=gn_groups, dropout=dropout))
            ch *= 2
        self.downs = nn.ModuleList(enc)

        # Bottleneck
        self.bottleneck = DoubleConv(ch, ch, gn_groups=gn_groups, dropout=dropout)

        # Decoder
        dec: List[nn.Module] = []
        for _ in range(depth - 1):
            dec.append(Up(ch, ch // 2, gn_groups=gn_groups, dropout=dropout))
            ch //= 2
        self.ups = nn.ModuleList(dec)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        # pop last skip (same as x before bottleneck)
        skips = skips[:-1][::-1]  # reverse excluding deepest
        for up, skip in zip(self.ups, skips):
            x = up(x, skip)

        return self.outc(x)


# =========================
# Config + main model
# =========================

@dataclass
class FNOConfig:
    in_channels: int
    out_channels: int = 1
    modes: int = 12
    width: int = 64
    padding: int = 12

    # residual refinement head
    use_residual: bool = True
    residual_unet_depth: int = 4
    residual_unet_base_channels: int = 32
    residual_unet_gn_groups: int = 8
    residual_unet_dropout: float = 0.1

    # regularization
    noise_std: float = 0.0


class FNO2dEnhanced(nn.Module):
    def __init__(self, cfg: FNOConfig):
        super().__init__()
        self.cfg = cfg

        self.p = nn.Linear(cfg.in_channels, cfg.width)

        self.conv0 = SpectralConv2d(cfg.width, cfg.width, cfg.modes, cfg.modes)
        self.conv1 = SpectralConv2d(cfg.width, cfg.width, cfg.modes, cfg.modes)
        self.conv2 = SpectralConv2d(cfg.width, cfg.width, cfg.modes, cfg.modes)
        self.conv3 = SpectralConv2d(cfg.width, cfg.width, cfg.modes, cfg.modes)

        self.w0 = nn.Conv2d(cfg.width, cfg.width, 1)
        self.w1 = nn.Conv2d(cfg.width, cfg.width, 1)
        self.w2 = nn.Conv2d(cfg.width, cfg.width, 1)
        self.w3 = nn.Conv2d(cfg.width, cfg.width, 1)

        self.mlp0 = MLP(cfg.width)
        self.mlp1 = MLP(cfg.width)
        self.mlp2 = MLP(cfg.width)
        self.mlp3 = MLP(cfg.width)

        self.q = nn.Sequential(
            nn.Conv2d(cfg.width, cfg.width, 1),
            nn.GELU(),
            nn.Conv2d(cfg.width, cfg.out_channels, 1),
        )

        if cfg.use_residual:
            self.residual_net = ResidualUNet(
                in_channels=cfg.out_channels + cfg.in_channels,
                out_channels=cfg.out_channels,
                base_channels=cfg.residual_unet_base_channels,
                depth=cfg.residual_unet_depth,
                gn_groups=cfg.residual_unet_gn_groups,
                dropout=cfg.residual_unet_dropout,
            )
        else:
            self.residual_net = None

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.cfg.noise_std and self.cfg.noise_std > 0:
            return x + self.cfg.noise_std * torch.randn_like(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != self.cfg.in_channels:
            raise ValueError(f"Expected in_channels={self.cfg.in_channels}, got {C}")

        x_original = x
        x = self.add_noise(x)

        x = x.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        x = self.p(x).permute(0, 3, 1, 2).contiguous()  # (B,width,H,W)

        pad = int(self.cfg.padding) if self.cfg.padding else 0
        if pad > 0:
            x = F.pad(x, (0, pad, 0, pad))

        x = F.gelu(self.mlp0(self.conv0(x)) + self.w0(x))
        x = F.gelu(self.mlp1(self.conv1(x)) + self.w1(x))
        x = F.gelu(self.mlp2(self.conv2(x)) + self.w2(x))
        x = self.mlp3(self.conv3(x)) + self.w3(x)

        if pad > 0:
            x = x[:, :, :H, :W].contiguous()

        pred = self.q(x)

        if self.residual_net is not None:
            pred = pred + self.residual_net(torch.cat([pred, x_original], dim=1))

        return pred
