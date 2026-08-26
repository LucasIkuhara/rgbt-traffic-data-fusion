

import shutil
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from sklearn.model_selection import KFold

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


def write_labels(coco: COCO, img_ids: list[int], labels_root: Path, images_root: Path,
                 class_map: dict[int, int]) -> None:
    """Write YOLO .txt label files and an images.txt list for a set of image IDs."""
    image_lines = []
    for img_id in img_ids:
        img_meta = coco.imgs[img_id]
        img_w, img_h = img_meta["width"], img_meta["height"]
        file_name = img_meta["file_name"]

        label_path = labels_root / Path(file_name).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_lines.append(str((images_root / file_name).resolve()))

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

    (labels_root.parent / "images.txt").write_text("\n".join(image_lines) + "\n")


def write_yaml(fold_dir: Path, train_images: Path, val_images: Path,
               model_names: dict[int, str]) -> Path:
    yaml_path = fold_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {fold_dir.resolve()}\n"
        f"train: {(train_images.parent / 'images.txt').resolve()}\n"
        f"val: {(val_images.parent / 'images.txt').resolve()}\n"
        f"\nnc: {len(model_names)}\n"
        f"names: {list(model_names.values())}\n"
    )
    return yaml_path


def symlink_scenes(images_root: Path, dataset_path: Path) -> None:
    """Symlink scene dirs so YOLO's images/→labels/ substitution resolves correctly."""
    images_root.mkdir(parents=True, exist_ok=True)
    for scene in dataset_path.iterdir():
        if scene.is_dir():
            link = images_root / scene.name
            if not link.exists():
                link.symlink_to(scene.resolve())


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

    kf = KFold(n_splits=1, shuffle=True, random_state=42)
    fold_results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(img_ids), start=1):
        print(f"\n{'='*60}")
        print(f"  Fold {fold}/{tr['n_splits']}  —  train: {len(train_idx)}  val: {len(val_idx)}")
        print(f"{'='*60}")

        fold_dir = work_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_imgs_root = fold_dir / "train" / "images"
        val_imgs_root   = fold_dir / "val"   / "images"
        symlink_scenes(train_imgs_root, dataset_path)
        symlink_scenes(val_imgs_root, dataset_path)

        write_labels(coco, img_ids[train_idx].tolist(), fold_dir / "train" / "labels", train_imgs_root, class_map)
        write_labels(coco, img_ids[val_idx].tolist(),   fold_dir / "val"   / "labels", val_imgs_root, class_map)
        yaml_path = write_yaml(fold_dir, train_imgs_root, val_imgs_root, base_model.names)

        # Fine-tune from base thermal weights
        model = YOLO(tr["input_model"])
        model.train(
            data=str(yaml_path),
            epochs=tr["epochs"],
            imgsz=tr["imgsz"],
            batch=tr["batch"],
            freeze=tr["freeze"],
            project=str(work_dir),
            name=f"fold_{fold}_train",
            exist_ok=True,
        )

        # Evaluate on validation fold
        best_weights = work_dir / f"fold_{fold}_train" / "weights" / "best.pt"
        val_model = YOLO(str(best_weights))
        metrics = val_model.val(data=str(yaml_path), split="val", verbose=False)

        result = {
            "fold": fold,
            "map50":    round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map),   4),
        }
        fold_results.append(result)
        print(f"  Fold {fold} → mAP50: {result['map50']}  mAP50-95: {result['map50_95']}")

    # Summary — copy best fold's weights to output_model path
    print(f"\n{'='*60}")
    print("  K-Fold Summary")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: mAP50={r['map50']}  mAP50-95={r['map50_95']}")
    map50_vals   = [r["map50"]    for r in fold_results]
    map5095_vals = [r["map50_95"] for r in fold_results]
    print(f"\n  Mean mAP50:    {np.mean(map50_vals):.4f} ± {np.std(map50_vals):.4f}")
    print(f"  Mean mAP50-95: {np.mean(map5095_vals):.4f} ± {np.std(map5095_vals):.4f}")

    best_fold = fold_results[int(np.argmax(map50_vals))]
    best_weights = work_dir / f"fold_{best_fold['fold']}_train" / "weights" / "best.pt"
    output_model = Path(tr["output_model"])
    shutil.copy(best_weights, output_model)
    print(f"\n  Best fold: {best_fold['fold']} (mAP50={best_fold['map50']}) → saved to {output_model}")


if __name__ == "__main__":
    main()
