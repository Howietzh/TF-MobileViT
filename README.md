# TF-MobileViT

Traditional Feature-Guided MobileViT Network for few-shot multi-class compact camera module defect segmentation.

This repository provides a runnable PyTorch research implementation with:

- overlapping high-resolution patch training and inference
- Gabor, LBP, and CLAHE traditional feature priors
- a multi-scale Traditional Feature-Guided Attention Module (TFGAM)
- a lightweight MobileViT-style encoder-decoder segmentation model
- train, eval, infer, and ablation CLIs

Copy-paste augmentation is intentionally not included.

## Dataset Layout

Masks are expected to store integer class IDs.

```text
data/CCM_Defect/
  images/
    0.png
    1.png
    ...
  annotations/
    0.png
    1.png
    ...
  train_subset_1.txt
  train_subset_2.txt
  ...
  val.txt
  test.txt
```

Default classes:

| ID | Class |
| --- | --- |
| 0 | background |
| 1 | FPC cracks |
| 2 | foreign particles |
| 3 | end face notches |
| 4 | lens scratches |

Each split txt file contains one sample ID or filename per line. For example, `11` resolves to `images/11.png` and `annotations/11.png`.

## Install

```bash
pip install -r requirements.txt
```

The implementation can run without `timm`; when `timm` is unavailable it falls back to the built-in compact MobileViT-style encoder. Installing `timm` is recommended for research experiments.

By default the config expects a local timm MobileViT checkpoint at:

```text
weights/model.safetensors
```

The `weights/` directory is ignored by git. If this file exists, the model loads it locally and does not contact Hugging Face for MobileViT weights. The loader adapts timm classification checkpoint keys for `features_only=True` encoders and ignores classifier-head tensors. To use online timm pretrained download instead, set `model.pretrained_path: null`.

Traditional feature extraction can be moved from DataLoader workers to the training device for faster batch computation of Gabor and LBP:

```yaml
features:
  compute_device: gpu
  use_clahe: false
```

CLAHE remains CPU-only; disable it for the fastest GPU feature path or keep `compute_device: cpu` when strict CLAHE features are required.

The default TF-MobileViT model feeds RGB patches directly into the timm MobileViT encoder. Traditional features guide the encoder outputs through multi-scale TFGAM instead of being injected before MobileViT:

```text
F_s_guided = F_s + alpha_s * (F_s * A_s)
```

Each encoder scale has its own attention block and learnable scalar `alpha_s`, initialized by `model.tfgam_alpha_init`.

## Commands

```bash
python -m tf_mobilevit.cli train --config configs/default.yaml
python -m tf_mobilevit.cli train --config configs/default.yaml --train-split-file train_subset_3.txt
python -m tf_mobilevit.cli eval --config configs/default.yaml --checkpoint runs/tf_mobilevit/best.pt
python -m tf_mobilevit.cli infer --config configs/default.yaml --checkpoint runs/tf_mobilevit/best.pt --input path/to/image.png --output runs/predictions
python -m tf_mobilevit.cli ablate --config configs/default.yaml
```

Batch few-shot training:

```bash
bash scripts/run_few_shot.sh
```

Useful overrides:

```bash
SUBSETS="1 3 5" RUN_ROOT=runs/few_shot_mobilevit bash scripts/run_few_shot.sh
```

The few-shot script uses GPU batch computation for Gabor and LBP by default:

```bash
FEATURE_COMPUTE_DEVICE=gpu USE_CLAHE=false bash scripts/run_few_shot.sh
```

To reproduce the older CPU feature path with CLAHE:

```bash
FEATURE_COMPUTE_DEVICE=cpu USE_CLAHE=true bash scripts/run_few_shot.sh
```

By default, the batch script runs two experiments for every subset:

- `tf_mobilevit`: Gabor + LBP + optional CLAHE + multi-scale TFGAM
- `mobilevit_no_trad`: MobileViT baseline without traditional image features

Outputs are written to:

```text
runs/few_shot/tf_mobilevit/subset_1
runs/few_shot/mobilevit_no_trad/subset_1
```

Each run writes terminal output and interruption/failure messages to `train.log` inside its output directory.

Run only one experiment when needed:

```bash
EXPERIMENTS="mobilevit_no_trad" bash scripts/run_few_shot.sh
```

Multi-GPU training uses PyTorch DDP through `torchrun`:

```bash
torchrun --nproc_per_node=2 -m tf_mobilevit.cli train --config configs/default.yaml
```

The few-shot script can launch DDP runs too:

```bash
GPUS=2 bash scripts/run_few_shot.sh
```

Only rank 0 writes TensorBoard logs, `history.csv`, metrics, and checkpoints.

Training writes scalar logs for TensorBoard under each run directory:

```bash
tensorboard --logdir runs/few_shot
```

Each run also keeps `history.csv`, `metrics.json`, and checkpoints in its output directory. During training, the terminal prints batch loss, learning rate, timing, and epoch-level validation metrics.

Validation runs every 5 epochs by default and once at the final epoch. The best checkpoint is updated only on validation epochs using foreground Dice.

## Smoke Check

```bash
python -m tf_mobilevit.smoke
```
