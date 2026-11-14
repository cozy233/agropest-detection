"""Evaluate a local fine-tuned detector with COCO-style mAP."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics.json"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive; workers >= 0")
    import torch
    from torch.utils.data import DataLoader
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from dataset import PestDataset, collate_fn
    from model import get_faster_rcnn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    dataset = PestDataset(args.data_root / args.split / "images", args.data_root / args.split / "labels")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_fn)
    model = get_faster_rcnn()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.to(device).eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    with torch.no_grad():
        for images, targets in loader:
            predictions = model([image.to(device) for image in images])
            predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]
            targets = [{k: v.cpu() for k, v in t.items() if k in {"boxes", "labels"}} for t in targets]
            metric.update(predictions, targets)
    values = {k: v.tolist() for k, v in metric.compute().items()}
    report = {"split": args.split, "images": len(dataset), "metrics": values}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
