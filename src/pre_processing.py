from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from sklearn.model_selection import KFold, train_test_split

from src.params import params


# ---------------------------------------------------------------------------
# Annotation preparation
# ---------------------------------------------------------------------------

def prepare_annotations(coco: COCO) -> tuple[int, int]:
    """Prepare annotations for training and evaluation.

    For each annotation:
      - Skips those without a segmentation mask (degenerate; dropped from output).
      - Derives bbox and area from the RLE mask so COCOeval scores against the
        same boxes that write_labels() uses for YOLO training labels.

    Mutates the COCO object in-place, rebuilds its index, and returns
    (n_dropped, n_updated).
    """
    kept: list = []
    updated = 0
    for ann in coco.dataset["annotations"]:
        if not ann.get("segmentation"):
            continue  # drop — no mask to derive a bbox from
        rle = coco.annToRLE(ann)
        x, y, w, h = maskUtils.toBbox(rle).tolist()
        if w > 0 and h > 0:
            ann["bbox"] = [x, y, w, h]
            ann["area"] = float(w * h)
            updated += 1
        kept.append(ann)
    dropped = len(coco.dataset["annotations"]) - len(kept)
    coco.dataset["annotations"] = kept
    coco.createIndex()
    return dropped, updated


def filter_categories(coco: COCO, exclude: list[str]) -> int:
    """Remove annotations and category entries whose name is in *exclude*.

    Mutates the COCO object in-place, rebuilds its index, and returns the
    number of annotations removed.
    """
    exclude_set = set(exclude)
    excluded_cat_ids = {c["id"] for c in coco.dataset["categories"] if c["name"] in exclude_set}

    before = len(coco.dataset["annotations"])
    coco.dataset["annotations"] = [
        a for a in coco.dataset["annotations"] if a["category_id"] not in excluded_cat_ids
    ]
    coco.dataset["categories"] = [
        c for c in coco.dataset["categories"] if c["name"] not in exclude_set
    ]
    coco.createIndex()
    return before - len(coco.dataset["annotations"])


# ---------------------------------------------------------------------------
# YOLO label writing
# ---------------------------------------------------------------------------

def write_labels(
    coco: COCO,
    img_ids: list[int],
    dataset_path: Path,
    images_txt: Path,
    class_map: dict[int, int],
) -> None:
    """Write YOLO .txt label files next to each image and an images.txt list.

    Each label line: ``<cls> <cx> <cy> <w> <h>`` (all normalised 0–1).
    Bboxes are derived from the RLE segmentation mask, matching the convention
    used throughout this project.
    """
    image_lines: list[str] = []
    for img_id in img_ids:
        img_meta = coco.imgs[img_id]
        img_w, img_h = img_meta["width"], img_meta["height"]
        file_name = img_meta["file_name"]

        img_path   = (dataset_path / file_name).resolve()
        label_path = img_path.with_suffix(".txt")
        image_lines.append(str(img_path))

        rows: list[str] = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] not in class_map:
                continue
            if not ann["segmentation"]:
                continue
            rle = coco.annToRLE(ann)
            x, y, w, h = maskUtils.toBbox(rle)
            if w <= 0 or h <= 0:
                continue
            cls = class_map[ann["category_id"]]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            rows.append(f"{cls} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}")

        label_path.write_text("\n".join(rows) + ("\n" if rows else ""))

    images_txt.write_text("\n".join(image_lines) + "\n")


def build_class_map(coco: COCO, model_names: dict[int, str]) -> dict[int, int]:
    """Map COCO category_id → YOLO class index by matching category names."""
    model_idx_by_name = {v: k for k, v in model_names.items()}
    return {
        c["id"]: model_idx_by_name[c["name"]]
        for c in coco.dataset["categories"]
        if c["name"] in model_idx_by_name
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tr      = params["training"]
    pre     = params["preprocessing"]
    exp_rgb = params["experiments"]["rgb"]
    exp_th  = params["experiments"]["thermal"]

    work_dir = Path(tr["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load annotations ──────────────────────────────────────────────
    print("Loading annotation files …")
    print(f"  RGB:     {pre['input_rgb_ann']}")
    print(f"  Thermal: {pre['input_thermal_ann']}")
    rgb_coco     = COCO(pre["input_rgb_ann"])
    thermal_coco = COCO(pre["input_thermal_ann"])

    # ── 2. Drop degenerate annotations and realign bboxes to RLE ────────
    print("Preparing annotations (drop degenerate, realign bboxes to RLE) …")
    for label, coco_obj in [("RGB", rgb_coco), ("Thermal", thermal_coco)]:
        dropped, updated = prepare_annotations(coco_obj)
        print(f"  [{label}] dropped={dropped}  bbox_updated={updated}")

    # ── 3. Filter to selected categories ────────────────────────────────
    exclude = pre["filter_categories"]
    print(f"Excluding categories: {exclude} …")
    for label, coco_obj in [("RGB", rgb_coco), ("Thermal", thermal_coco)]:
        removed = filter_categories(coco_obj, exclude)
        remaining = len(coco_obj.dataset["annotations"])
        print(f"  [{label}] removed={removed}  remaining={remaining}")

    # ── 4. Save prepared annotation JSONs ───────────────────────────────
    for label, coco_obj, out_path in [
        ("RGB",     rgb_coco,     pre["output_rgb_ann"]),
        ("Thermal", thermal_coco, pre["output_thermal_ann"]),
    ]:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(coco_obj.dataset, f)
        print(f"  [{label}] cleaned annotations saved → {out_path}")

    # ── 5. Build k-fold splits ──────────────────────────────────────────
    img_ids  = np.array(sorted(thermal_coco.imgs.keys()))
    n_splits = tr["n_splits"]
    if n_splits == 1:
        splits = [train_test_split(range(len(img_ids)), test_size=0.2, random_state=42)]
    else:
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(img_ids))

    print(f"\nWriting YOLO label files for {n_splits} fold(s) …")

    # Derive class maps from the annotation category names directly (no model
    # needed); map category_id → 0-based index sorted by category id.
    def _simple_class_map(coco_obj: COCO) -> dict[int, int]:
        sorted_cats = sorted(coco_obj.dataset["categories"], key=lambda c: c["id"])
        return {c["id"]: i for i, c in enumerate(sorted_cats)}

    rgb_class_map     = _simple_class_map(rgb_coco)
    thermal_class_map = _simple_class_map(thermal_coco)

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_dir = work_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for modality, coco_obj, class_map, base_dir in [
            ("rgb",     rgb_coco,     rgb_class_map,     Path(exp_rgb["dataset_base_dir"])),
            ("thermal", thermal_coco, thermal_class_map, Path(exp_th["dataset_base_dir"])),
        ]:
            train_txt = fold_dir / f"train_images_{modality}.txt"
            val_txt   = fold_dir / f"val_images_{modality}.txt"

            write_labels(coco_obj, img_ids[train_idx].tolist(), base_dir, train_txt, class_map)
            write_labels(coco_obj, img_ids[val_idx].tolist(),   base_dir, val_txt,   class_map)

            print(f"  fold {fold} / {modality}: "
                  f"train={len(train_idx)} val={len(val_idx)} → {fold_dir}")

    print("\nPre-processing complete.")


if __name__ == "__main__":
    main()
