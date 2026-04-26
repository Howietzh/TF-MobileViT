# TF-MobileViT

Traditional Feature-Guided MobileViT Network for few-shot multi-class compact camera module defect segmentation.

This repository provides a runnable PyTorch research implementation with:

- overlapping high-resolution patch training and inference
- Gabor, LBP, and CLAHE traditional feature priors
- a Traditional Feature-Guided Attention Module (TFGAM)
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

## Smoke Check

```bash
python -m tf_mobilevit.smoke
```
