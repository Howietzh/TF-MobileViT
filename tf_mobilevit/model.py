from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MV2Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 2) -> None:
        super().__init__()
        hidden = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out + x if self.use_residual else out


class MobileViTBlock(nn.Module):
    def __init__(self, channels: int, transformer_dim: int | None = None, depth: int = 1, patch_size: int = 2) -> None:
        super().__init__()
        transformer_dim = transformer_dim or channels
        self.patch_size = patch_size
        self.local = ConvBlock(channels, channels)
        self.project_in = nn.Conv2d(channels, transformer_dim, 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=max(1, min(4, transformer_dim // 16)),
            dim_feedforward=transformer_dim * 2,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.project_out = nn.Conv2d(transformer_dim, channels, 1)
        self.fuse = ConvBlock(channels * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        y = self.project_in(local)
        b, c, h, w = y.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h))
        _, _, hp, wp = y.shape
        tokens = y.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        y = tokens.transpose(1, 2).reshape(b, c, hp, wp)
        y = y[:, :, :h, :w]
        y = self.project_out(y)
        return self.fuse(torch.cat([x, y], dim=1))


class TFGAM(nn.Module):
    def __init__(self, trad_channels: int, rgb_channels: int) -> None:
        super().__init__()
        if trad_channels <= 0:
            self.encoder = None
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(trad_channels, rgb_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(rgb_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(rgb_channels, rgb_channels, 1),
                nn.Sigmoid(),
            )

    def forward(self, rgb_features: torch.Tensor, trad_features: torch.Tensor | None) -> torch.Tensor:
        if self.encoder is None or trad_features is None or trad_features.shape[1] == 0:
            return rgb_features
        attention = self.encoder(trad_features)
        return rgb_features * attention + rgb_features


class BuiltInMobileViTEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = ConvBlock(in_channels, c1, stride=2)
        self.stage1 = MV2Block(c1, c2, stride=2)
        self.stage2 = nn.Sequential(MV2Block(c2, c3, stride=2), MobileViTBlock(c3))
        self.stage3 = nn.Sequential(MV2Block(c3, c4, stride=2), MobileViTBlock(c4))
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stem(x)
        f2 = self.stage1(f1)
        f3 = self.stage2(f2)
        f4 = self.stage3(f3)
        return [f1, f2, f3, f4]


class TimmEncoder(nn.Module):
    def __init__(self, name: str, pretrained: bool) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        self.out_channels = list(self.model.feature_info.channels())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TFMobileViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        trad_channels: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        config = config or {}
        channels = list(config.get("encoder_channels", [32, 48, 64, 96]))
        decoder_channels = int(config.get("decoder_channels", 64))
        self.use_tfgam = bool(config.get("use_tfgam", True))
        self.rgb_shallow = ConvBlock(3, channels[0])
        self.tfgam = TFGAM(trad_channels, channels[0]) if self.use_tfgam else TFGAM(0, channels[0])

        self.encoder_kind = "builtin"
        try:
            if config.get("timm_backbone"):
                self.encoder = TimmEncoder(config["timm_backbone"], bool(config.get("pretrained", False)))
                enc_channels = self.encoder.out_channels
                self.encoder_input = nn.Conv2d(channels[0], 3, 1)
                self.encoder_kind = "timm"
            else:
                raise ImportError
        except ModuleNotFoundError:
            warnings.warn(
                "timm is not installed; falling back to the built-in MobileViT-style encoder. "
                "Install requirements.txt to use pretrained timm MobileViT weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.encoder = BuiltInMobileViTEncoder(channels[0], channels)
            enc_channels = self.encoder.out_channels
            self.encoder_input = nn.Identity()
        except ImportError:
            self.encoder = BuiltInMobileViTEncoder(channels[0], channels)
            enc_channels = self.encoder.out_channels
            self.encoder_input = nn.Identity()

        c1, c2, c3, c4 = enc_channels[:4]
        self.dec3 = DecoderBlock(c4, c3, decoder_channels)
        self.dec2 = DecoderBlock(decoder_channels, c2, decoder_channels)
        self.dec1 = DecoderBlock(decoder_channels, c1, decoder_channels)
        self.high_fuse = DecoderBlock(decoder_channels, channels[0], decoder_channels)
        self.head = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, image: torch.Tensor, trad_features: torch.Tensor | None = None) -> torch.Tensor:
        shallow = self.rgb_shallow(image)
        guided = self.tfgam(shallow, trad_features)
        features = self.encoder(self.encoder_input(guided))
        x = self.dec3(features[-1], features[-2])
        x = self.dec2(x, features[-3])
        x = self.dec1(x, features[-4])
        x = self.high_fuse(x, guided)
        logits = self.head(x)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def build_model(config: dict[str, Any], trad_channels: int) -> TFMobileViT:
    num_classes = len(config["dataset"]["classes"])
    return TFMobileViT(num_classes=num_classes, trad_channels=trad_channels, config=config.get("model", {}))
