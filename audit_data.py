"""Check annotation pairing and class/coordinate validity without PyTorch."""

import argparse
from collections import Counter
import json
from pathlib import Path

from labels import CLASS_NAMES, parse_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--limit", type=int, help="Inspect the first N sorted images")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    images_dir, labels_dir = (args.data_root / args.split / d for d in ("images", "labels"))
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError("No images found")
    selected = images[:args.limit] if args.limit else images
    classes, empty, errors = Counter(), 0, []
    for image in selected:
        path = labels_dir / (image.stem + ".txt")
        if not path.is_file():
            errors.append({"image": image.name, "error": "missing label"})
            continue
        found = 0
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                item = parse_label(line, 1, 1)
                if item:
                    classes[CLASS_NAMES[item[0] - 1]] += 1
                    found += 1
            except ValueError as exc:
                errors.append({"label": path.name, "line": number, "error": str(exc)})
        empty += found == 0
    print(json.dumps({"split": args.split, "total_images": len(images), "checked_images": len(selected),
                      "empty_annotations": empty, "objects_by_class": dict(classes), "errors": errors}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
