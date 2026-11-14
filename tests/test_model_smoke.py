"""Small CPU integration check; no dataset, checkpoint or model download.

The test skips only when torch/torchvision are not installed. CI installs both
and sets REQUIRE_MODEL_SMOKE=1 so that a skipped integration check fails there.
"""

from contextlib import contextmanager
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_MODEL_STACK = all(importlib.util.find_spec(name) is not None
                      for name in ("torch", "torchvision"))


@contextmanager
def temporary_checkpoint():
    # Only this one generated file is removed; no directory cleanup or recursive
    # file operation is used. Close the handle before torch.save for Windows.
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
        path = Path(handle.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        if not HAS_MODEL_STACK:
            message = "torch/torchvision are not installed; detector smoke was not run"
            if os.environ.get("REQUIRE_MODEL_SMOKE") == "1":
                self.fail(message)
            self.skipTest(message)
        import torch
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)

    @staticmethod
    def shrink(model):
        model.transform.min_size = (64,)
        model.transform.max_size = 64
        model.rpn._pre_nms_top_n = {"training": 64, "testing": 32}
        model.rpn._post_nms_top_n = {"training": 32, "testing": 16}
        model.rpn.fg_bg_sampler.batch_size_per_image = 32
        model.roi_heads.fg_bg_sampler.batch_size_per_image = 32
        model.roi_heads.detections_per_img = 8
        return model

    @staticmethod
    def norm_signature(model):
        from torchvision.ops.misc import FrozenBatchNorm2d
        return [(name, type(module).__name__, module.eps) for name, module in model.named_modules()
                if isinstance(module, FrozenBatchNorm2d)]

    def test_detection_modes_on_cpu(self):

        # Import errors from an incompatible/broken installed stack are failures,
        # not skips. This catches mismatched torchvision compiled operators.
        import torch
        from model import get_faster_rcnn
        from train import validation_loss

        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("No downloads allowed")), \
                 patch("torchvision.models._api.load_state_dict_from_url", side_effect=AssertionError("No downloads allowed")):
            torch.manual_seed(7)
            model = get_faster_rcnn(num_classes=13, pretrained=False).cpu()
            self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 13)
            self.shrink(model)
            images = [torch.rand(3, 64, 64)]
            targets = [{"boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
                        "labels": torch.tensor([1], dtype=torch.int64)}]

            model.train()
            losses = model(images, targets)
            self.assertIsInstance(losses, dict)
            self.assertTrue({"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
                            .issubset(losses))
            self.assertTrue(all(value.ndim == 0 and torch.isfinite(value).item()
                                for value in losses.values()))
            sum(losses.values()).backward()
            gradients = [parameter.grad for parameter in model.parameters()
                         if parameter.requires_grad and parameter.grad is not None]
            self.assertTrue(gradients, "Trainable parameters must receive gradients")
            self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
            model.zero_grad(set_to_none=True)

            from torchvision.ops.misc import FrozenBatchNorm2d
            batchnorms = [module for module in model.modules() if isinstance(module, FrozenBatchNorm2d)]
            self.assertTrue(batchnorms)
            self.assertFalse(any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                                 for module in model.modules()))
            self.assertTrue(all(module.eps == 0.0 for module in batchnorms))
            # A mixed-mode state checks restoration of individual submodules,
            # rather than only the top-level model.training flag.
            batchnorms[0].eval()
            flags_before = [(module, module.training) for module in model.modules()]
            buffers_before = [(module, module.running_mean.clone(), module.running_var.clone())
                              for module in batchnorms]
            value = validation_loss(model, [(images, targets)], torch.device("cpu"))
            self.assertTrue(math.isfinite(value))
            for module, training in flags_before:
                self.assertEqual(module.training, training)
            for module, mean, variance in buffers_before:
                self.assertTrue(torch.equal(module.running_mean, mean))
                self.assertTrue(torch.equal(module.running_var, variance))

            model.eval()
            with torch.no_grad():
                predictions = model(images)
            self.assertIsInstance(predictions, list)
            self.assertEqual(len(predictions), 1)
            prediction = predictions[0]
            self.assertTrue({"boxes", "labels", "scores"}.issubset(prediction))
            boxes, labels, scores = (prediction[key] for key in ("boxes", "labels", "scores"))
            self.assertEqual(tuple(boxes.shape), (len(labels), 4))
            self.assertEqual(tuple(scores.shape), (len(labels),))
            self.assertEqual(labels.dtype, torch.int64)
            self.assertTrue(torch.isfinite(boxes).all().item())
            self.assertTrue(torch.isfinite(scores).all().item())
            self.assertTrue(((scores >= 0) & (scores <= 1)).all().item())
            self.assertTrue(((labels >= 1) & (labels <= 12)).all().item())

    def test_finetuned_checkpoint_round_trip(self):
        import torch
        from model import get_faster_rcnn

        with patch("torchvision.models._api.load_state_dict_from_url", side_effect=AssertionError("No downloads allowed")):
            torch.manual_seed(11)
            original = self.shrink(get_faster_rcnn(num_classes=13)).cpu().eval()
            images = [torch.rand(3, 64, 64)]
            with torch.no_grad():
                expected = original(images)[0]
            with temporary_checkpoint() as path:
                torch.save(original.state_dict(), path)
                restored = self.shrink(get_faster_rcnn(num_classes=13)).cpu()
                restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                restored.eval()
                self.assertEqual(self.norm_signature(original), self.norm_signature(restored))
                with torch.no_grad():
                    actual = restored(images)[0]
            for key in ("boxes", "labels", "scores"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_random_local_and_mocked_coco_initialization_match(self):
        import torch
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        from model import get_faster_rcnn

        # A generated 91-class state has the same key/shape contract as the COCO
        # checkpoint. The download is mocked; no network or pretrained artifact
        # is needed to verify that both loading routes preserve the backbone.
        torch.manual_seed(13)
        reference = get_faster_rcnn(num_classes=91)
        state = reference.state_dict()
        expected_norm = self.norm_signature(reference)
        self.assertTrue(expected_norm)
        expected_grad = {name: value.requires_grad for name, value in reference.backbone.named_parameters()}
        self.assertTrue(all(not value for name, value in expected_grad.items()
                            if name.startswith(("body.conv1.", "body.layer1."))))
        self.assertTrue(any(value for name, value in expected_grad.items() if name.startswith("body.layer2.")))
        with temporary_checkpoint() as path:
            torch.save(state, path)
            with patch.object(FasterRCNN_ResNet50_FPN_Weights.DEFAULT, "get_state_dict", return_value=state) as download:
                pretrained = get_faster_rcnn(pretrained=True)
            download.assert_called_once_with(progress=True, check_hash=True)
            with patch("torchvision.models._api.load_state_dict_from_url", side_effect=AssertionError("No downloads allowed")):
                local = get_faster_rcnn(init_weights=path)
            for model in (pretrained, local):
                self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 13)
                self.assertEqual(self.norm_signature(model), expected_norm)
                self.assertEqual({name: value.requires_grad for name, value in model.backbone.named_parameters()}, expected_grad)
                for key, value in reference.backbone.state_dict().items():
                    torch.testing.assert_close(model.backbone.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
