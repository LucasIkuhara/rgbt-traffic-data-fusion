

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

def _train_one_model(
    modality: str,
    ann_file: Path,
    dataset_path: Path,
    img_ids: np.ndarray,
    train_idx,
    val_idx,
    fold: int,
    fold_dir: Path,
    tr: dict,
) -> dict:
    """Fine-tune one model for a single fold and return its metrics."""
    from ultralytics import YOLO

    input_key  = f"input_model_{modality}"
    output_key = f"output_model_{modality}"

    base_model = YOLO(tr[input_key])
    class_map  = build_class_map(COCO(str(ann_file)), base_model.names)

    train_txt = fold_dir / f"train_images_{modality}.txt"
    val_txt   = fold_dir / f"val_images_{modality}.txt"
    coco_obj  = COCO(str(ann_file))

    write_labels(coco_obj, img_ids[train_idx].tolist(), dataset_path, train_txt, class_map)
    write_labels(coco_obj, img_ids[val_idx].tolist(),   dataset_path, val_txt,   class_map)
    (fold_dir / modality).mkdir(parents=True, exist_ok=True)
    yaml_path = write_yaml(fold_dir / modality, train_txt, val_txt, base_model.names)

    model = YOLO(tr[input_key])
    train_result = model.train(
        data=str(yaml_path),
        epochs=tr["epochs"],
        imgsz=tr["imgsz"],
        batch=tr["batch"],
        freeze=tr["freeze"],
        project=str((fold_dir / modality).resolve()),
        name="train",
        exist_ok=True,
    )

    best_weights = Path(train_result.save_dir) / "weights" / "best.pt"
    val_model = YOLO(str(best_weights))
    metrics = val_model.val(data=str(yaml_path), split="val", verbose=False)

    output_model = Path(tr[output_key])
    fold_output  = output_model.with_stem(f"{output_model.stem}_fold_{fold}")
    shutil.copy(best_weights, fold_output)

    return {
        "fold":     fold,
        "modality": modality,
        "map50":    round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map),   4),
        "path":     str(fold_output),
    }


def _print_summary(label: str, results: list[dict], output_model: Path) -> None:
    map50_vals   = [r["map50"]    for r in results]
    map5095_vals = [r["map50_95"] for r in results]
    print(f"\n  [{label}] Mean mAP50: {np.mean(map50_vals):.4f} ± {np.std(map50_vals):.4f}"
          f"  |  mAP50-95: {np.mean(map5095_vals):.4f} ± {np.std(map5095_vals):.4f}")
    best = results[int(np.argmax(map50_vals))]
    shutil.copy(best["path"], output_model)
    print(f"  [{label}] Best fold: {best['fold']} (mAP50={best['map50']}) → {output_model}")


def main() -> None:
    tr       = params["training"]
    exp_rgb  = params["experiments"]["rgb"]
    exp_th   = params["experiments"]["thermal"]
    work_dir = Path(tr["work_dir"])

    # Use thermal image IDs to define the fold split (same IDs exist in both)
    thermal_coco = COCO(str(Path(exp_th["dataset_file"])))
    img_ids = np.array(sorted(thermal_coco.imgs.keys()))

    n_splits = tr["n_splits"]
    if n_splits == 1:
        splits = [train_test_split(range(len(img_ids)), test_size=0.2, random_state=42)]
    else:
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(img_ids))

    rgb_results:     list[dict] = []
    thermal_results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n{'='*60}")
        print(f"  Fold {fold}/{tr['n_splits']}  —  train: {len(train_idx)}  val: {len(val_idx)}")
        print(f"{'='*60}")

        fold_dir = work_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for modality, exp in [("rgb", exp_rgb), ("thermal", exp_th)]:
            print(f"\n  -- {modality} --")
            result = _train_one_model(
                modality    = modality,
                ann_file    = Path(exp["dataset_file"]),
                dataset_path= Path(exp["dataset_base_dir"]),
                img_ids     = img_ids,
                train_idx   = train_idx,
                val_idx     = val_idx,
                fold        = fold,
                fold_dir    = fold_dir,
                tr          = tr,
            )
            print(f"  [{modality}] mAP50: {result['map50']}  mAP50-95: {result['map50_95']}  → {result['path']}")
            (rgb_results if modality == "rgb" else thermal_results).append(result)

    print(f"\n{'='*60}")
    print("  K-Fold Summary")
    print(f"{'='*60}")
    _print_summary("rgb",     rgb_results,     Path(tr["output_model_rgb"]))
    _print_summary("thermal", thermal_results, Path(tr["output_model_thermal"]))


if __name__ == "__main__":
    main()
