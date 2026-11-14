"""One consistent Faster R-CNN architecture for initialization and evaluation."""

import torch
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

COCO_NUM_CLASSES = 91
TRAINABLE_BACKBONE_LAYERS = 3
FROZEN_BATCHNORM_EPS = 0.0


def get_faster_rcnn(num_classes=13, *, pretrained=False, init_weights=None):
    if pretrained and init_weights:
        raise ValueError("Choose COCO weights or a local COCO checkpoint")
    if num_classes < 2:
        raise ValueError("num_classes must include background and at least one foreground class")
    # The torchvision convenience factory selects different normalization and
    # trainable layers for pretrained versus random initialization. Construct
    # these explicitly so saved fine-tuned checkpoints have identical semantics
    # when rebuilt for evaluation without downloading pretrained weights.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, norm_layer=FrozenBatchNorm2d,
        trainable_layers=TRAINABLE_BACKBONE_LAYERS,
    )
    model = FasterRCNN(backbone, num_classes=COCO_NUM_CLASSES)
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = FROZEN_BATCHNORM_EPS
    if pretrained:
        state = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
        model.load_state_dict(state)
    elif init_weights:
        state = torch.load(init_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    if num_classes != COCO_NUM_CLASSES:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
