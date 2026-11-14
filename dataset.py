"""AgroPest YOLO labels adapted to torchvision detection targets."""

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from labels import parse_label


class PestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir, self.label_dir = Path(img_dir), Path(label_dir)
        self.image_files = sorted(p for p in self.img_dir.iterdir()
                                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.image_files:
            raise ValueError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        with Image.open(path) as original:
            image = original.convert("RGB")
        label_path = self.label_dir / (path.stem + ".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        boxes, labels = [], []
        for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                item = parse_label(line, *image.size)
            except ValueError as exc:
                raise ValueError(f"{label_path}:{number}: {exc}") from exc
            if item is not None:
                label, box = item
                labels.append(label)
                boxes.append(box)
        # Empty annotation files are valid negative images; never recurse to a
        # different index. Geometric augmentations are omitted unless boxes are
        # transformed together with images.
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor(idx, dtype=torch.int64)}
        return to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))
