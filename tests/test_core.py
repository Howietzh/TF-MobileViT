from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from tf_mobilevit.data import CCMPatchDataset
from tf_mobilevit.features import FeatureConfig, TraditionalFeatureExtractor, compute_traditional_features, feature_channel_count
from tf_mobilevit.model import MultiScaleTFGAM, build_model
from tf_mobilevit.patches import compute_patch_grid, extract_patch, stitch_probabilities


def test_patch_stitch_preserves_size() -> None:
    image = np.zeros((70, 90, 3), dtype=np.uint8)
    coords = compute_patch_grid(70, 90, patch_size=32, overlap=0.5)
    patches = [torch.ones(2, 32, 32) for _ in coords]
    stitched = stitch_probabilities(patches, coords, (70, 90))
    assert stitched.shape == (2, 70, 90)
    assert torch.allclose(stitched, torch.ones_like(stitched))
    assert extract_patch(image, coords[-1], 32).shape == (32, 32, 3)


def test_traditional_features_shape_and_range() -> None:
    image = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    cfg = FeatureConfig(gabor_kernel_size=9)
    features = compute_traditional_features(image, cfg)
    assert features.shape == (feature_channel_count(cfg), 64, 64)
    assert features.min() >= 0
    assert features.max() <= 1


def test_gpu_traditional_features_shape_and_range() -> None:
    cfg = FeatureConfig(
        compute_device="gpu",
        use_clahe=False,
        gabor_kernel_size=9,
        gpu_features={"gabor": True, "lbp": True, "clahe": False},
    )
    extractor = TraditionalFeatureExtractor(cfg)
    image = torch.rand(2, 3, 32, 32)
    features = extractor(image)
    assert features.shape == (2, 7, 32, 32)
    assert torch.isfinite(features).all()
    assert features.min() >= 0
    assert features.max() <= 1


def test_model_forward_shape() -> None:
    config = {
        "dataset": {"classes": ["background", "a", "b"]},
        "model": {
            "timm_backbone": None,
            "encoder_channels": [8, 12, 16, 24],
            "decoder_channels": 16,
            "use_tfgam": True,
            "tfgam_alpha_init": 0.1,
        },
    }
    model = build_model(config, trad_channels=8)
    image = torch.randn(1, 3, 64, 64)
    features = torch.randn(1, 8, 64, 64)
    logits = model(image, features)
    assert logits.shape == (1, 3, 64, 64)


def test_model_forward_with_gpu_features() -> None:
    feature_cfg = FeatureConfig(compute_device="gpu", use_clahe=False, gpu_features={"gabor": True, "lbp": True})
    extractor = TraditionalFeatureExtractor(feature_cfg)
    config = {
        "dataset": {"classes": ["background", "a", "b"]},
        "model": {
            "timm_backbone": None,
            "encoder_channels": [8, 12, 16, 24],
            "decoder_channels": 16,
            "use_tfgam": True,
            "tfgam_alpha_init": 0.1,
        },
    }
    image = torch.rand(1, 3, 64, 64)
    features = extractor(image)
    model = build_model(config, trad_channels=features.shape[1])
    logits = model(image, features)
    assert logits.shape == (1, 3, 64, 64)


def test_baseline_model_names_forward_shape() -> None:
    names = ["unet", "deeplabv3plus", "segformer_b0", "mobilevit_no_trad"]
    image = torch.rand(1, 3, 64, 64)
    trad = torch.rand(1, 7, 64, 64)
    for name in names:
        config = {
            "dataset": {"classes": ["background", "a", "b"]},
            "model": {
                "name": name,
                "timm_backbone": None,
                "pretrained": False,
                "encoder_channels": [8, 12, 16, 24],
                "decoder_channels": 16,
                "unet_channels": [8, 12, 16, 24],
                "deeplab_channels": [8, 12, 16, 24],
                "segformer_channels": [8, 12, 16, 24],
                "segformer_backbone": "mit_b0",
                "use_tfgam": False,
            },
        }
        model = build_model(config, trad_channels=0 if name != "mobilevit_no_trad" else trad.shape[1])
        logits = model(image, trad)
        assert logits.shape == (1, 3, 64, 64)


def test_direct_traditional_fusion_forward_shape() -> None:
    config = {
        "dataset": {"classes": ["background", "a", "b"]},
        "model": {
            "name": "gabor_lbp",
            "timm_backbone": None,
            "pretrained": False,
            "encoder_channels": [8, 12, 16, 24],
            "decoder_channels": 16,
            "use_tfgam": False,
            "direct_trad_fusion": True,
            "tfgam_alpha_init": 0.1,
        },
    }
    model = build_model(config, trad_channels=7)
    image = torch.rand(1, 3, 64, 64)
    trad = torch.rand(1, 7, 64, 64)
    logits = model(image, trad)
    assert logits.shape == (1, 3, 64, 64)


def test_multiscale_tfgam_shapes_and_learnable_alphas() -> None:
    module = MultiScaleTFGAM(trad_channels=7, feature_channels=[8, 12, 16, 24], alpha_init=0.1)
    features = [
        torch.randn(2, 8, 32, 32),
        torch.randn(2, 12, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 24, 4, 4),
    ]
    trad = torch.rand(2, 7, 64, 64)
    guided = module(features, trad)

    assert [item.shape for item in guided] == [item.shape for item in features]
    assert module.alphas is not None
    assert module.alphas.requires_grad
    assert module.alphas.shape == (4,)
    assert any(param is module.alphas for param in module.parameters())


def test_multiscale_tfgam_identity_without_traditional_features() -> None:
    module = MultiScaleTFGAM(trad_channels=7, feature_channels=[8, 12], alpha_init=0.1)
    features = [torch.randn(1, 8, 16, 16), torch.randn(1, 12, 8, 8)]
    guided = module(features, None)

    assert len(guided) == len(features)
    for original, output in zip(features, guided):
        assert torch.equal(original, output)


def test_split_file_dataset_layout(tmp_path) -> None:
    root = tmp_path / "CCM_Defect"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 1
    Image.fromarray(image).save(root / "images" / "11.png")
    Image.fromarray(mask).save(root / "annotations" / "11.png")
    (root / "train_subset_1.txt").write_text("11\n", encoding="utf-8")
    cfg = {
        "dataset": {
            "root": str(root),
            "image_dir": "images",
            "mask_dir": "annotations",
            "train_split_file": "train_subset_1.txt",
            "classes": ["background", "defect"],
            "patch_size": 32,
            "overlap": 0.5,
            "ignore_index": 255,
            "augment": {"enabled": False},
        },
        "features": {"enabled": False},
    }
    sample = CCMPatchDataset(cfg, split="train", training=False)[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].max().item() == 1


def test_dataset_defers_gpu_features(tmp_path) -> None:
    root = tmp_path / "CCM_Defect"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(root / "images" / "11.png")
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(root / "annotations" / "11.png")
    (root / "train_subset_1.txt").write_text("11\n", encoding="utf-8")
    cfg = {
        "dataset": {
            "root": str(root),
            "image_dir": "images",
            "mask_dir": "annotations",
            "train_split_file": "train_subset_1.txt",
            "classes": ["background", "defect"],
            "patch_size": 32,
            "overlap": 0.5,
            "ignore_index": 255,
            "augment": {"enabled": False},
        },
        "features": {"enabled": True, "compute_device": "gpu", "use_gabor": True, "use_lbp": True, "use_clahe": False},
    }
    sample = CCMPatchDataset(cfg, split="train", training=False)[0]
    assert sample["features"].shape == (0, 32, 32)
