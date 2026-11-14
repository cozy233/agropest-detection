# AgroPest Detection — Faster R-CNN branch

[![CPU smoke tests](https://github.com/cozy233/agropest-detection/actions/workflows/cpu-smoke.yml/badge.svg)](https://github.com/cozy233/agropest-detection/actions/workflows/cpu-smoke.yml)

A curated version of the locally available Faster R-CNN implementation from an
UNSW COMP9517 course team project on agricultural pest detection.

**Scope:** this repository contains the Faster R-CNN ResNet-50 FPN branch.
The team report also discusses YOLO, RT-DETR and other methods, but their training
implementations were not found in the materials used here. This repository does
not claim to reproduce the report's selected YOLOv8-L results.

## What is included

- YOLO-format annotations converted to torchvision targets for 12 pest classes.
- A 13-class detector head (12 pests plus background).
- Training with Adam/SGD, a step learning-rate schedule and early stopping.
- Separate validation loss and COCO-style mAP evaluation commands.
- Data checks and focused annotation-conversion regression checks.

## Setup

Python **3.11** is the recommended environment and the version configured in CI.
Install a matching pair of PyTorch/torchvision wheels for your hardware, then
install the remaining packages:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
```

Dependency ranges are proposed requirements, not a replayed historical lockfile.
The original detector checkpoint, dataset and course documents are not included.

The CPU smoke workflow installs the official PyTorch 2.6.0 / torchvision 0.21.0
CPU wheels. It builds a random-weight 13-class detector, runs one tiny synthetic
train/validation/evaluation pass and checks finite loss and backward gradients, restored model modes,
unchanged validation normalization statistics and prediction schema. It also
checks fine-tuned checkpoint round-trip equality and the random/local/mocked-COCO
initialization paths. It never downloads a model checkpoint. Run the same checks with:

```bash
python -m unittest discover -s tests -v
```

Without torch/torchvision installed, the detector check explicitly reports a skip;
the label tests still run. CI requires the detector check and treats missing
dependencies as a failure. This smoke check is not a trained-model accuracy result.

## Dataset

Obtain [AgroPest-12 from its original Kaggle page](https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset)
and check its terms before use. Keep the provided train, validation and test
splits separate:

```text
data/agropest12/
  train/images/       train/labels/
  valid/images/       valid/labels/
  test/images/        test/labels/
```

Labels must contain `class_id x_center y_center width height` with zero-based
class IDs and normalized coordinates. Images must have a matching `.txt` file;
an empty label file represents a negative image. No example images are redistributed.

```bash
python audit_data.py --data-root data/agropest12 --split train
python -m unittest discover -s tests -v
```

## Train and evaluate

```bash
# --pretrained explicitly permits downloading torchvision's COCO weights.
python train.py --data-root data/agropest12 --pretrained --epochs 20 --batch-size 2
python evaluate_map.py --data-root data/agropest12 --weights artifacts/best_fasterrcnn.pth --split valid
# Use test only after model selection is complete.
python evaluate_map.py --data-root data/agropest12 --weights artifacts/best_fasterrcnn.pth --split test --output artifacts/test_metrics.json
```

Without `--pretrained` or `--init-weights`, training starts from random weights
and makes no implicit model download. `--init-weights` accepts a local original
COCO checkpoint; `evaluate_map.py --weights` instead expects a fine-tuned 13-class
checkpoint. Training configuration, history, weights and metrics are written to
the ignored `artifacts/` directory.

All initialization and evaluation paths build the same ResNet-50 FPN backbone:
`FrozenBatchNorm2d` with epsilon `0.0`, three trainable backbone stages (layers
2–4), and the same RPN/FPN layout. This keeps random, COCO-initialized and restored
fine-tuned models structurally consistent. Random initialization still freezes the
early backbone stages and normalization; it is mainly useful for offline smoke
checks unless you deliberately choose that training setup.

Validation loss uses training-mode detection heads under `no_grad`, with
normalization statistics frozen, because torchvision returns predictions in evaluation
mode. Proposal sampling makes this loss stochastic; mAP is evaluated separately.

## Results and provenance

No new training run or dataset-level detector evaluation has been performed for
this curated version. See [results status](docs/results.md). Packaging fixes
include correct validation-loader use, safe annotation handling and explicit
weight initialization. Geometric augmentations that did not transform bounding
boxes together with images were not carried over from the alternative local code.

This originated in a course **team** project; this repository does not claim sole
authorship of the full team's work. Course specifications and the team report are
excluded. Dataset and third-party software retain their respective terms. See
[source and packaging changes](docs/source_changes.md) for the implementation's
origin and the changes made while organizing it.
