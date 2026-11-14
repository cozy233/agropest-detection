# Source and packaging changes

The starting point was the supplied local COMP9517 project materials, not a new
implementation of all experiments in the team report. File paths below refer to
the original material folders, not files bundled with this repository.

| Repository file | Starting material | Organization changes |
| --- | --- | --- |
| `dataset.py`, `labels.py` | `project/dataset.py`, `project/archive/data.yaml` | Extracted class names and label parsing; validate malformed annotations; preserve empty annotations instead of recursively selecting another image; no image-only geometric augmentation. |
| `model.py` | `project/model.py` | Explicit COCO/local/random initialization; construct the same FrozenBatchNorm/epsilon/trainable-stage configuration in all paths; disable implicit backbone downloads; load checkpoint on CPU with `weights_only=True`. |
| `train.py` | `project/train.py`, with cross-check of `code/train.py` | Configurable paths/arguments and output history; validate on the validation loader; request detection losses in the appropriate model mode while freezing BatchNorm statistics. |
| `evaluate_map.py` | `project/evaluate_map.py` | Configurable split/checkpoint/device/output; avoid downloading original weights when loading a trained checkpoint. |
| `audit_data.py` | `project/1.py` pairing-check idea | Added annotation parsing checks and machine-readable summary without the ML dependency stack. |
| `tests/test_labels.py` | New packaging checks | Check coordinate conversion, clipping, blank lines and invalid annotations. |
| `tests/test_model_smoke.py`, `.github/workflows/cpu-smoke.yml` | New integration checks | Exercise tiny CPU detector train/validation/evaluation passes, checkpoint round-trip equality, and random/local/mocked-COCO initialization without model downloads; CI installs the dependency stack and requires the checks. |

The alternative `code/` material included experimental image-only geometric
augmentation and a single-level anchor configuration. These were not carried
over into this FPN baseline. This is a cleaned branch, not an exact archived
submission; original historical measurements must not be attributed to the
modified code without a new run.

Raw images/labels, downloaded COCO weights, course specification PDFs, the team
report, caches and compressed submission copies are not included. No repository-wide
license has been assigned to the original team code; external dependencies keep
their existing licenses.
