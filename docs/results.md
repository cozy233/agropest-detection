# Results status

This candidate preserves and repairs the locally available Faster R-CNN branch.
It does not include a fine-tuned detector checkpoint, original prediction outputs
or an independently replayed model evaluation. The only checkpoint located with
the source materials was the original COCO pretrained model.

The course team report's final selected model is YOLOv8-L. Its reported mAP,
precision, recall and latency must not be presented as measurements produced by
this Faster R-CNN candidate. The report also contains a Faster R-CNN comparison,
but the original evaluation output and matching final weights were not located;
those values are not advertised as reproduced results here.

Packaging validation passed Python syntax checks, both ML command help screens,
four annotation-conversion tests, and label audits for all 11,502 training,
1,095 validation and 546 test images. The training split contains three empty
annotation files, now preserved as negative images rather than replaced by another
index. No label-pairing or annotation-format errors were found. The audit checks
labels and file pairing; it does not decode every image or evaluate a detector.
Details are recorded in [validation.json](validation.json).

Detector training, inference and mAP computation
require the full PyTorch stack and remain to be verified. New results, when run,
should record the dataset split, dependency versions, hardware, seed and checkpoint.

An automated CPU detector smoke check is included in `tests/test_model_smoke.py`.
The local test run passed four label tests and explicitly skipped three detector
integration checks because torch/torchvision were absent. The GitHub Actions workflow is
configured to install the CPU stack and require this check; a workflow pass must
be observed before calling the detector integration check passed. The synthetic
smoke is separate from dataset-level training and accuracy evaluation.
