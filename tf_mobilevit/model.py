from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def load_checkpoint_file(path: str) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("Loading .safetensors weights requires `safetensors`. Install it with requirements.txt.") from exc
        return load_file(str(checkpoint_path))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def convert_timm_feature_key(key: str) -> str:
    for idx in range(8):
        prefix = f"stages.{idx}."
        if key.startswith(prefix):
            return f"stages_{idx}." + key[len(prefix) :]
    return key


def remove_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


def load_local_timm_features_checkpoint(model: nn.Module, path: str) -> None:
    raw_state = load_checkpoint_file(path)
    model_state = model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in raw_state.items():
        new_key = convert_timm_feature_key(remove_prefix(key, "module."))
        if new_key.startswith(("head.", "classifier.", "fc.")):
            skipped += 1
            continue
        if new_key in model_state and model_state[new_key].shape == value.shape:
            converted[new_key] = value
        else:
            skipped += 1
    incompatible = model.load_state_dict(converted, strict=False)
    print(
        f"Loaded local timm checkpoint from {path}: "
        f"{len(converted)} tensors loaded, {skipped} skipped, "
        f"{len(incompatible.missing_keys)} missing, {len(incompatible.unexpected_keys)} unexpected.",
        flush=True,
    )


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
    def __init__(self, trad_channels: int, target_channels: int) -> None:
        super().__init__()
        if trad_channels <= 0:
            self.encoder = None
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(trad_channels, target_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(target_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(target_channels, target_channels, 1),
                nn.Sigmoid(),
            )

    def forward(self, target_features: torch.Tensor, trad_features: torch.Tensor | None) -> torch.Tensor:
        if self.encoder is None or trad_features is None or trad_features.shape[1] == 0:
            return target_features
        attention = self.encoder(trad_features)
        return target_features * attention


class MultiScaleTFGAM(nn.Module):
    def __init__(self, trad_channels: int, feature_channels: list[int], alpha_init: float = 0.1) -> None:
        super().__init__()
        self.enabled = trad_channels > 0 and len(feature_channels) > 0
        if self.enabled:
            self.guidance_blocks = nn.ModuleList([TFGAM(trad_channels, channels) for channels in feature_channels])
            self.alphas = nn.Parameter(torch.full((len(feature_channels),), float(alpha_init)))
        else:
            self.guidance_blocks = nn.ModuleList()
            self.register_parameter("alphas", None)

    def forward(
        self,
        features: list[torch.Tensor],
        trad_features: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if not self.enabled or trad_features is None or trad_features.shape[1] == 0:
            return list(features)

        guided_features: list[torch.Tensor] = []
        for idx, feature in enumerate(features):
            resized_trad = F.interpolate(
                trad_features,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            guided_delta = self.guidance_blocks[idx](feature, resized_trad)
            guided_features.append(feature + self.alphas[idx] * guided_delta)
        return guided_features


class DirectTraditionalFusion(nn.Module):
    def __init__(self, trad_channels: int, feature_channels: list[int], alpha_init: float = 0.1) -> None:
        super().__init__()
        self.enabled = trad_channels > 0 and len(feature_channels) > 0
        if self.enabled:
            self.projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(trad_channels, channels, 3, padding=1, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.SiLU(inplace=True),
                    )
                    for channels in feature_channels
                ]
            )
            self.alphas = nn.Parameter(torch.full((len(feature_channels),), float(alpha_init)))
        else:
            self.projections = nn.ModuleList()
            self.register_parameter("alphas", None)

    def forward(self, features: list[torch.Tensor], trad_features: torch.Tensor | None) -> list[torch.Tensor]:
        if not self.enabled or trad_features is None or trad_features.shape[1] == 0:
            return list(features)
        fused: list[torch.Tensor] = []
        for idx, feature in enumerate(features):
            resized_trad = F.interpolate(trad_features, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            fused.append(feature + self.alphas[idx] * self.projections[idx](resized_trad))
        return fused


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
    def __init__(self, name: str, pretrained: bool, pretrained_path: str | None = None) -> None:
        super().__init__()
        import timm

        checkpoint_path = str(Path(pretrained_path).expanduser()) if pretrained_path else ""
        self.model = timm.create_model(
            name,
            pretrained=pretrained and not checkpoint_path,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        if checkpoint_path:
            load_local_timm_features_checkpoint(self.model, checkpoint_path)
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


class SimpleUNet(nn.Module):
    def __init__(self, num_classes: int, channels: list[int] | None = None) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels or [32, 64, 128, 256]
        self.enc1 = ConvBlock(3, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, image: torch.Tensor, trad_features: torch.Tensor | None = None) -> torch.Tensor:
        f1 = self.enc1(image)
        f2 = self.enc2(self.pool(f1))
        f3 = self.enc3(self.pool(f2))
        f4 = self.enc4(self.pool(f3))
        x = self.dec3(f4, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        return self.head(x)


class ConvPyramidEncoder(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stage1 = ConvBlock(3, c1, stride=2)
        self.stage2 = ConvBlock(c1, c2, stride=2)
        self.stage3 = ConvBlock(c2, c3, stride=2)
        self.stage4 = ConvBlock(c3, c4, stride=2)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: tuple[int, ...] = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1 if rate == 1 else 3, padding=0 if rate == 1 else rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
                for rate in rates
            ]
        )
        self.project = ConvBlock(out_channels * len(rates), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabV3PlusSegmentation(nn.Module):
    def __init__(self, num_classes: int, channels: list[int] | None = None, decoder_channels: int = 128) -> None:
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        self.encoder = ConvPyramidEncoder(channels)
        c1, _, _, c4 = channels
        self.aspp = ASPP(c4, decoder_channels)
        self.low_proj = nn.Sequential(
            nn.Conv2d(c1, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )
        self.decoder = ConvBlock(decoder_channels + 48, decoder_channels)
        self.head = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, image: torch.Tensor, trad_features: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder(image)
        x = self.aspp(features[-1])
        low = self.low_proj(features[0])
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, low], dim=1))
        logits = self.head(x)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


class TimmEncoderDecoderSegmentation(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str,
        decoder_channels: int = 128,
        fallback_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        fallback_channels = fallback_channels or [32, 64, 160, 256]
        try:
            self.encoder = TimmEncoder(backbone, pretrained=False, pretrained_path=None)
        except Exception as exc:
            warnings.warn(
                f"Could not create timm backbone {backbone!r}; using a compact convolutional pyramid instead. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.encoder = ConvPyramidEncoder(fallback_channels)
        channels = self.encoder.out_channels[:4]
        c1, c2, c3, c4 = channels
        self.proj4 = nn.Conv2d(c4, decoder_channels, 1)
        self.proj3 = nn.Conv2d(c3, decoder_channels, 1)
        self.proj2 = nn.Conv2d(c2, decoder_channels, 1)
        self.proj1 = nn.Conv2d(c1, decoder_channels, 1)
        self.fuse = ConvBlock(decoder_channels * 4, decoder_channels)
        self.head = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, image: torch.Tensor, trad_features: torch.Tensor | None = None) -> torch.Tensor:
        f1, f2, f3, f4 = self.encoder(image)[:4]
        target_size = f1.shape[-2:]
        features = [
            self.proj1(f1),
            F.interpolate(self.proj2(f2), size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(self.proj3(f3), size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(self.proj4(f4), size=target_size, mode="bilinear", align_corners=False),
        ]
        x = self.fuse(torch.cat(features, dim=1))
        logits = self.head(x)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


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
        self.use_direct_trad_fusion = bool(config.get("direct_trad_fusion", False))
        alpha_init = float(config.get("tfgam_alpha_init", 0.1))

        self.encoder_kind = "builtin"
        try:
            if config.get("timm_backbone"):
                pretrained_path = config.get("pretrained_path")
                if pretrained_path and not Path(str(pretrained_path)).expanduser().exists():
                    warnings.warn(
                        f"Configured pretrained_path does not exist: {pretrained_path}. "
                        "Falling back to timm pretrained download if enabled.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    pretrained_path = None
                self.encoder = TimmEncoder(
                    config["timm_backbone"],
                    bool(config.get("pretrained", False)),
                    str(pretrained_path) if pretrained_path else None,
                )
                enc_channels = self.encoder.out_channels
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
            self.encoder = BuiltInMobileViTEncoder(3, channels)
            enc_channels = self.encoder.out_channels
        except ImportError:
            self.encoder = BuiltInMobileViTEncoder(3, channels)
            enc_channels = self.encoder.out_channels

        c1, c2, c3, c4 = enc_channels[:4]
        tfgam_channels = enc_channels[:4] if self.use_tfgam else []
        self.tfgam = MultiScaleTFGAM(trad_channels, tfgam_channels, alpha_init=alpha_init)
        direct_fusion_channels = enc_channels[:4] if self.use_direct_trad_fusion and not self.use_tfgam else []
        self.direct_trad_fusion = DirectTraditionalFusion(trad_channels, direct_fusion_channels, alpha_init=alpha_init)
        self.dec3 = DecoderBlock(c4, c3, decoder_channels)
        self.dec2 = DecoderBlock(decoder_channels, c2, decoder_channels)
        self.dec1 = DecoderBlock(decoder_channels, c1, decoder_channels)
        self.head = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, image: torch.Tensor, trad_features: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder(image)
        features = self.tfgam(features, trad_features)
        features = self.direct_trad_fusion(features, trad_features)
        x = self.dec3(features[-1], features[-2])
        x = self.dec2(x, features[-3])
        x = self.dec1(x, features[-4])
        logits = self.head(x)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def build_model(config: dict[str, Any], trad_channels: int) -> nn.Module:
    num_classes = len(config["dataset"]["classes"])
    model_config = config.get("model", {})
    name = str(model_config.get("name", "tf_mobilevit")).lower()
    if name == "unet":
        return SimpleUNet(num_classes=num_classes, channels=list(model_config.get("unet_channels", [32, 64, 128, 256])))
    if name == "deeplabv3plus":
        return DeepLabV3PlusSegmentation(
            num_classes=num_classes,
            channels=list(model_config.get("deeplab_channels", [64, 128, 256, 512])),
            decoder_channels=int(model_config.get("decoder_channels", 128)),
        )
    if name == "segformer_b0":
        return TimmEncoderDecoderSegmentation(
            num_classes=num_classes,
            backbone=str(model_config.get("segformer_backbone", "mit_b0")),
            decoder_channels=int(model_config.get("decoder_channels", 128)),
            fallback_channels=list(model_config.get("segformer_channels", [32, 64, 160, 256])),
        )
    if name in {"tf_mobilevit", "mobilevit_no_trad", "mobilevit_baseline", "gabor", "gabor_lbp", "gabor_lbp_clahe"}:
        return TFMobileViT(num_classes=num_classes, trad_channels=trad_channels, config=model_config)
    raise ValueError(f"Unsupported model.name: {name}")
