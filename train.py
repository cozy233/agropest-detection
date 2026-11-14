"""Portable training entry point adapted from the local Faster R-CNN branch."""

import argparse
import json
from pathlib import Path
import random


def validation_loss(model, loader, device):
    """Request detection losses without gradients or BatchNorm stat updates.

    torchvision returns detections in eval mode, not a loss dictionary. Loss
    validation therefore uses training-mode heads while freezing BatchNorm.
    Proposal sampling remains stochastic; this is a loss proxy, not mAP.
    """
    import torch
    states = [(module, module.training) for module in model.modules()]
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    total = 0.0
    try:
        with torch.no_grad():
            for images, targets in loader:
                images = [x.to(device) for x in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                losses = model(images, targets)
                total += sum(losses.values()).item()
    finally:
        for module, training in states:
            module.training = training
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="Adam")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretrained", action="store_true", help="Download official COCO weights")
    group.add_argument("--init-weights", type=Path, help="Local COCO checkpoint before replacing head")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("epochs, batch size, patience and learning rate must be positive; workers >= 0")

    import torch
    from torch.utils.data import DataLoader
    from dataset import PestDataset, collate_fn
    from model import get_faster_rcnn

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    loaders = []
    for split in ("train", "valid"):
        dataset = PestDataset(args.data_root / split / "images", args.data_root / split / "labels")
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                                  num_workers=args.workers, collate_fn=collate_fn))
    train_loader, val_loader = loaders
    model = get_faster_rcnn(pretrained=args.pretrained, init_weights=args.init_weights).to(device)
    optimizer = (torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "Adam"
                 else torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, targets in train_loader:
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        val = validation_loss(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total / len(train_loader), "validation_loss": val}
        history.append(row)
        print(json.dumps(row))
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        scheduler.step()
        if val < best:
            best, stale = val, 0
            torch.save(model.state_dict(), args.output_dir / "best_fasterrcnn.pth")
        else:
            stale += 1
        if stale >= args.patience:
            break


if __name__ == "__main__":
    main()
