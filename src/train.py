

import shutil
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from sklearn.model_selection import KFold, train_test_split

from src.params import params

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Label preparation
# ---------------------------------------------------------------------------

def build_class_map(coco: COCO, model_names: dict[int, str]) -> dict[int, int]:
    """Map COCO category_id → YOLO model class index by matching category names.

    Derives the mapping at runtime from the GT annotation file and the loaded
    model, so no hardcoded class lists are needed.
    """
    model_idx_by_name = {v: k for k, v in model_names.items()}
    return {
        c["id"]: model_idx_by_name[c["name"]]
        for c in coco.dataset["categories"]
        if c["name"] in model_idx_by_name
    }


def write_labels(coco: COCO, img_ids: list[int], dataset_path: Path,
                 images_txt: Path, class_map: dict[int, int]) -> None:
    """Write YOLO .txt label files next to each image (where YOLO looks after
    resolving symlinks) and an images.txt listing the real image paths."""
    image_lines = []
    for img_id in img_ids:
        img_meta = coco.imgs[img_id]
        img_w, img_h = img_meta["width"], img_meta["height"]
        file_name = img_meta["file_name"]

        img_path   = (dataset_path / file_name).resolve()
        label_path = img_path.with_suffix(".txt")
        image_lines.append(str(img_path))

        rows = []
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
            cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
            rows.append(f"{cls} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}")

        label_path.write_text("\n".join(rows) + ("\n" if rows else ""))

    images_txt.write_text("\n".join(image_lines) + "\n")


def write_yaml(fold_dir: Path, train_txt: Path, val_txt: Path,
               model_names: dict[int, str]) -> Path:
    yaml_path = fold_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {fold_dir.resolve()}\n"
        f"train: {train_txt.resolve()}\n"
        f"val: {val_txt.resolve()}\n"
        f"\nnc: {len(model_names)}\n"
        f"names: {list(model_names.values())}\n"
    )
    return yaml_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    exp = params["experiments"]["thermal"]
    tr  = params["training"]

    dataset_path = Path(exp["dataset_base_dir"])
    ann_file     = Path(exp["dataset_file"])
    work_dir     = Path(tr["work_dir"])

    from ultralytics import YOLO
    base_model = YOLO(tr["input_model"])
    class_map = build_class_map(COCO(str(ann_file)), base_model.names)
    print(f"Class map ({len(class_map)} categories matched): {class_map}")

    coco = COCO(str(ann_file))
    img_ids = np.array(sorted(coco.imgs.keys()))

    n_splits = tr["n_splits"]
    if n_splits == 1:
        splits = [train_test_split(range(len(img_ids)), test_size=0.2, random_state=42)]
    else:
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(img_ids)

    fold_results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n{'='*60}")
        print(f"  Fold {fold}/{tr['n_splits']}  —  train: {len(train_idx)}  val: {len(val_idx)}")
        print(f"{'='*60}")

        fold_dir = work_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_txt = fold_dir / "train_images.txt"
        val_txt   = fold_dir / "val_images.txt"

        write_labels(coco, img_ids[train_idx].tolist(), dataset_path, train_txt, class_map)
        write_labels(coco, img_ids[val_idx].tolist(),   dataset_path, val_txt,   class_map)
        yaml_path = write_yaml(fold_dir, train_txt, val_txt, base_model.names)

        # Fine-tune from base thermal weights
        model = YOLO(tr["input_model"])
        train_result = model.train(
            data=str(yaml_path),
            epochs=tr["epochs"],
            imgsz=tr["imgsz"],
            batch=tr["batch"],
            freeze=tr["freeze"],
            project=str(work_dir.resolve()),
            name=f"fold_{fold}_train",
            exist_ok=True,
        )

        # Evaluate on validation fold and save per-fold weights
        best_weights = Path(train_result.save_dir) / "weights" / "best.pt"
        val_model = YOLO(str(best_weights))
        metrics = val_model.val(data=str(yaml_path), split="val", verbose=False)

        output_model = Path(tr["output_model"])
        fold_output = output_model.with_stem(f"{output_model.stem}_fold_{fold}")
        shutil.copy(best_weights, fold_output)

        result = {
            "fold": fold,
            "map50":    round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map),   4),
            "path":     str(fold_output),
        }
        fold_results.append(result)
        print(f"  Fold {fold} → mAP50: {result['map50']}  mAP50-95: {result['map50_95']}  → {fold_output}")

    # Summary — copy best fold's weights to output_model path
    print(f"\n{'='*60}")
    print("  K-Fold Summary")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: mAP50={r['map50']}  mAP50-95={r['map50_95']}  {r['path']}")
    map50_vals   = [r["map50"]    for r in fold_results]
    map5095_vals = [r["map50_95"] for r in fold_results]
    print(f"\n  Mean mAP50:    {np.mean(map50_vals):.4f} ± {np.std(map50_vals):.4f}")
    print(f"  Mean mAP50-95: {np.mean(map5095_vals):.4f} ± {np.std(map5095_vals):.4f}")

    best_fold = fold_results[int(np.argmax(map50_vals))]
    output_model = Path(tr["output_model"])
    shutil.copy(best_fold["path"], output_model)
    print(f"\n  Best fold: {best_fold['fold']} (mAP50={best_fold['map50']}) → saved to {output_model}")


if __name__ == "__main__":
    main()
