from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import skimage.io as io
from pycocotools.coco import COCO
from ultralytics import YOLO

from src.bbox_fusion import CocoDetection, fuse_detections
from src.calibration import IMG_W, IMG_H, DATASET_PATH, load_calib, transform_bbox_rgb_to_thermal
from src.coco_eval import coco_eval, coco_eval_per_class
from src.masks import apply_mask
from src.params import params

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO-2014 annotation names that differ from YOLOv8's COCO80 class names.
# Maps YOLO prediction name → COCO category name used in the annotation file.
_YOLO_TO_COCO_NAME: dict[str, str] = {
    "motorcycle":   "motorbike",
    "airplane":     "aeroplane",
    "couch":        "sofa",
    "potted plant": "pottedplant",
    "dining table": "diningtable",
    "tv":           "tvmonitor",
}

# ---------------------------------------------------------------------------
# Fold image-ID loader
# ---------------------------------------------------------------------------

def load_fold_image_ids(fold: int) -> list[int]:
    """Return thermal val image IDs for *fold* by reading val_images_thermal.txt."""
    tr  = params["training"]
    exp = params["experiments"]["thermal"]

    val_txt = Path(tr["work_dir"]) / f"fold_{fold}" / "val_images_thermal.txt"
    thermal_coco = COCO(exp["dataset_file"])

    fname_to_id = {
        img["file_name"]: img_id
        for img_id, img in thermal_coco.imgs.items()
    }

    dataset_base = Path(exp["dataset_base_dir"]).resolve()
    image_ids: list[int] = []
    for line in val_txt.read_text().splitlines():
        path = Path(line.strip())
        rel  = path.relative_to(dataset_base)
        image_ids.append(fname_to_id[str(rel)])

    return image_ids


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def _yolo_xywh_to_coco(v: list[float]) -> list[float]:
    """YOLO centre-xywh (pixel) → COCO top-left-xywh (pixel)."""
    return [v[0] - v[2] / 2, v[1] - v[3] / 2, v[2], v[3]]


def _run_model_on_images(
    model: YOLO,
    image_ids: list[int],
    thermal_coco: COCO,
    rgb_coco: COCO,
    dataset_base_thermal: str,
    dataset_base_rgb: str,
    gt_cat_by_name: dict[str, int],
    modality: str,                   # "thermal" | "rgb"
) -> tuple[list[CocoDetection], list[CocoDetection]]:
    """Run *model* on either thermal or RGB images.

    Returns (thermal_space_detections, raw_detections).
    For the thermal modality the two lists are identical.
    For RGB the first list has bboxes projected into thermal space via
    the per-clip calibration homography.
    """
    raw_dets:  list[CocoDetection] = []
    proj_dets: list[CocoDetection] = []

    for img_id in image_ids:
        thermal_meta  = thermal_coco.imgs[img_id]
        thermal_fname = thermal_meta["file_name"]   # e.g. Egensevej/Eg-1/cam2-*.png

        if modality == "thermal":
            file_name  = thermal_fname
            db_base    = dataset_base_thermal
            is_thermal = True
        else:
            file_name  = rgb_coco.imgs[img_id]["file_name"]
            db_base    = dataset_base_rgb
            is_thermal = False

        img_data = io.imread(f"{db_base}/{file_name}")
        img_data = apply_mask(img_data, DATASET_PATH, file_name, thermal=is_thermal)

        prediction = model.predict(img_data, verbose=False, augment=True)[0]

        calib = load_calib(file_name) if modality == "rgb" else None

        for box in prediction.boxes:
            cls_name  = model.names[int(box.cls[0])]
            coco_name = _YOLO_TO_COCO_NAME.get(cls_name, cls_name)
            cat_id    = gt_cat_by_name.get(coco_name)
            if cat_id is None:
                continue

            bbox_raw = _yolo_xywh_to_coco(box.xywh.tolist()[0])
            score    = float(box.conf[0])

            det_raw = CocoDetection(
                image_id=img_id,
                category_id=cat_id,
                bbox=bbox_raw,
                score=score,
            )
            raw_dets.append(det_raw)

            if modality == "rgb":
                bbox_proj = transform_bbox_rgb_to_thermal(bbox_raw, calib)
                proj_dets.append(CocoDetection(
                    image_id=img_id,
                    category_id=cat_id,
                    bbox=bbox_proj,
                    score=score,
                ))
            else:
                proj_dets.append(det_raw)

    return proj_dets, raw_dets


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    fold: int,
    thermal_model: YOLO,
    rgb_model: YOLO,
    thermal_coco: COCO,
    rgb_coco: COCO,
) -> dict:
    """Evaluate one fold; returns a metrics dict."""
    exp_thermal = params["experiments"]["thermal"]
    exp_rgb     = params["experiments"]["rgb"]

    gt_cat_by_name = {
        c["name"]: c["id"] for c in thermal_coco.dataset["categories"]
    }

    image_ids = load_fold_image_ids(fold)
    print(f"\n{'='*60}")
    print(f"  Fold {fold}  |  {len(image_ids)} val images")
    print(f"{'='*60}")

    # ── Thermal predictions ──────────────────────────────────────────────
    print("  Running thermal model …")
    thermal_dets, _ = _run_model_on_images(
        model=thermal_model,
        image_ids=image_ids,
        thermal_coco=thermal_coco,
        rgb_coco=rgb_coco,
        dataset_base_thermal=exp_thermal["dataset_base_dir"],
        dataset_base_rgb=exp_rgb["dataset_base_dir"],
        gt_cat_by_name=gt_cat_by_name,
        modality="thermal",
    )

    # ── RGB predictions projected to thermal space ───────────────────────
    print("  Running RGB model + homography projection …")
    rgb_proj_dets, _ = _run_model_on_images(
        model=rgb_model,
        image_ids=image_ids,
        thermal_coco=thermal_coco,
        rgb_coco=rgb_coco,
        dataset_base_thermal=exp_thermal["dataset_base_dir"],
        dataset_base_rgb=exp_rgb["dataset_base_dir"],
        gt_cat_by_name=gt_cat_by_name,
        modality="rgb",
    )

    # ── Fuse with all methods ─────────────────────────────────────────────
    print("  Fusing detections …")
    thermal_by_img: dict[int, list[CocoDetection]] = defaultdict(list)
    for d in thermal_dets:
        thermal_by_img[d["image_id"]].append(d)

    rgb_by_img: dict[int, list[CocoDetection]] = defaultdict(list)
    for d in rgb_proj_dets:
        rgb_by_img[d["image_id"]].append(d)

    inf = params["inference"]
    fusion_methods = ("wbf", "nms", "soft_nms", "nmw")
    fused_by_method: dict[str, list[CocoDetection]] = {}

    for method in fusion_methods:
        dets: list[CocoDetection] = []
        for img_id in image_ids:
            dets.extend(fuse_detections(
                rgb_detections=rgb_by_img.get(img_id, []),
                thermal_detections=thermal_by_img.get(img_id, []),
                img_w=IMG_W,
                img_h=IMG_H,
                iou_thr=inf["fusion_iou_thr"],
                method=method,
                soft_nms_sigma=inf["soft_nms_sigma"],
                soft_nms_thresh=inf["soft_nms_thresh"],
            ))
        fused_by_method[method] = dets

    # ── COCO evaluation ──────────────────────────────────────────────────
    m_thermal = coco_eval(thermal_coco, thermal_dets,  image_ids, f"Fold {fold} – Thermal")
    m_rgb     = coco_eval(thermal_coco, rgb_proj_dets, image_ids, f"Fold {fold} – RGB→Thermal")
    m_fused   = {
        method: coco_eval(thermal_coco, fused_by_method[method], image_ids,
                          f"Fold {fold} – Fused ({method.upper()})")
        for method in fusion_methods
    }

    pc_thermal = coco_eval_per_class(thermal_coco, thermal_dets,  image_ids)
    pc_rgb     = coco_eval_per_class(thermal_coco, rgb_proj_dets, image_ids)
    pc_fused   = {
        method: coco_eval_per_class(thermal_coco, fused_by_method[method], image_ids)
        for method in fusion_methods
    }

    return {
        "fold":      fold,
        "thermal":   m_thermal,
        "rgb":       m_rgb,
        "fused":     m_fused,
        "per_class": {
            "thermal": pc_thermal,
            "rgb":     pc_rgb,
            "fused":   pc_fused,
        },
    }


# ---------------------------------------------------------------------------
# Results export
# ---------------------------------------------------------------------------

def _export_results(all_results: list[dict], path: str = "results.xlsx") -> None:
    """Aggregate fold results into a DataFrame and export to Excel.

    Sheet 1 (Overview): one row per configuration, mean±std across folds.
    Sheet 2 (Per-Class): one row per (configuration, class), mean±std across folds.
    """
    import pandas as pd

    fusion_methods = ("wbf", "nms", "soft_nms", "nmw")
    metrics = ("map50", "map50_95", "recall", "f1")

    # ── Sheet 1: overview ────────────────────────────────────────────────
    configs: list[tuple[str, list[dict]]] = [
        ("Thermal",     [r["thermal"]     for r in all_results]),
        ("RGB→Thermal", [r["rgb"]         for r in all_results]),
    ]
    for method in fusion_methods:
        configs.append((f"Fused ({method.upper()})", [r["fused"][method] for r in all_results]))

    rows = []
    col_labels = {"map50": "mAP@50", "map50_95": "mAP@[0.5:0.95]", "recall": "Recall", "f1": "F1"}
    for name, fold_metrics in configs:
        row: dict = {"Configuration": name}
        for metric in metrics:
            values = np.array([m[metric] for m in fold_metrics])
            col = col_labels[metric]
            row[f"{col} Mean"] = round(float(values.mean()), 4)
            row[f"{col} Std"]  = round(float(values.std()),  4)
        rows.append(row)

    df_overview = pd.DataFrame(rows)

    # ── Sheet 2: per-class ───────────────────────────────────────────────
    cat_names = sorted(all_results[0]["per_class"]["thermal"].keys())

    pc_configs: list[tuple[str, list[dict[str, dict[str, float]]]]] = [
        ("Thermal",     [r["per_class"]["thermal"]     for r in all_results]),
        ("RGB→Thermal", [r["per_class"]["rgb"]         for r in all_results]),
    ]
    for method in fusion_methods:
        pc_configs.append((
            f"Fused ({method.upper()})",
            [r["per_class"]["fused"][method] for r in all_results],
        ))

    pc_col_labels = {"map50": "mAP@50", "map50_95": "mAP@[0.5:0.95]", "f1": "F1"}
    pc_rows = []
    for name, fold_pc in pc_configs:
        for cat in cat_names:
            row = {"Configuration": name, "Class": cat,
                   "Instances": sum(fp[cat]["instances"] for fp in fold_pc)}
            for metric in ("map50", "map50_95", "f1"):
                values = np.array([fp[cat][metric] for fp in fold_pc])
                col = pc_col_labels[metric]
                row[f"{col} Mean"] = round(float(values.mean()), 4)
                row[f"{col} Std"]  = round(float(values.std()),  4)
            pc_rows.append(row)

    df_per_class = pd.DataFrame(pc_rows)

    with pd.ExcelWriter(path) as writer:
        df_overview.to_excel(writer,  sheet_name="Overview",  index=False)
        df_per_class.to_excel(writer, sheet_name="Per-Class", index=False)

    print(f"\n  [export] Results written to {path}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_fold_model(fold: int, modality: str) -> YOLO:
    """Load the fine-tuned weights for *fold* and *modality* ('rgb'|'thermal')."""
    tr = params["training"]
    output_model = Path(tr[f"output_model_{modality}"])
    fold_path = output_model.with_stem(f"{output_model.stem}_fold_{fold}")
    if not fold_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned {modality} model not found: {fold_path}\n"
            "Run `python -m src.train` first."
        )
    inf = params["inference"]
    model = YOLO(str(fold_path))
    model.overrides["conf"]         = inf[f"conf_{modality}"]
    model.overrides["iou"]          = inf[f"iou_{modality}"]
    model.overrides["agnostic_nms"] = False
    model.overrides["max_det"]      = 1000
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tr          = params["training"]
    exp_thermal = params["experiments"]["thermal"]
    exp_rgb     = params["experiments"]["rgb"]

    n_splits = tr["n_splits"]

    # Annotation files are already cleaned and RLE-realigned by pre_processing.py.
    thermal_coco = COCO(exp_thermal["dataset_file"])
    rgb_coco     = COCO(exp_rgb["dataset_file"])

    all_results: list[dict] = []
    print("PARAMS USED:", params["inference"])

    for fold in range(1, n_splits + 1):
        thermal_model = _load_fold_model(fold, "thermal")
        rgb_model     = _load_fold_model(fold, "rgb")
        result = evaluate_fold(
            fold=fold,
            thermal_model=thermal_model,
            rgb_model=rgb_model,
            thermal_coco=thermal_coco,
            rgb_coco=rgb_coco,
        )
        all_results.append(result)

    # ── Summary table ────────────────────────────────────────────────────
    fusion_methods = ("wbf", "nms", "soft_nms", "nmw")

    def _mean(key, subkey, method=None):
        if method is None:
            return np.mean([r[key][subkey] for r in all_results])
        return np.mean([r[key][method][subkey] for r in all_results])

    # Per-modality block (Thermal + RGB)
    print(f"\n\n{'='*60}")
    print("  SUMMARY — Thermal & RGB  (mAP50  |  mAP50-95  |  F1)")
    print(f"{'='*60}")
    print(f"  {'Fold':>6}  {'Therm mAP50':>12}  {'Therm mAP':>10}  {'Therm F1':>9}  {'RGB mAP50':>10}  {'RGB mAP':>8}  {'RGB F1':>7}")
    print(f"  {'-'*72}")
    for r in all_results:
        print(
            f"  {r['fold']:>6}  "
            f"{r['thermal']['map50']:>12.4f}  {r['thermal']['map50_95']:>10.4f}  {r['thermal']['f1']:>9.4f}  "
            f"{r['rgb']['map50']:>10.4f}  {r['rgb']['map50_95']:>8.4f}  {r['rgb']['f1']:>7.4f}"
        )
    print(f"  {'-'*72}")
    print(
        f"  {'Mean':>6}  "
        f"{_mean('thermal','map50'):>12.4f}  {_mean('thermal','map50_95'):>10.4f}  {_mean('thermal','f1'):>9.4f}  "
        f"{_mean('rgb','map50'):>10.4f}  {_mean('rgb','map50_95'):>8.4f}  {_mean('rgb','f1'):>7.4f}"
    )
    print(f"{'='*60}")

    # Per-fusion-method block
    for method in fusion_methods:
        print(f"\n\n{'='*60}")
        print(f"  SUMMARY — Fused ({method.upper()})  (mAP50  |  mAP50-95  |  F1)")
        print(f"{'='*60}")
        print(f"  {'Fold':>6}  {'mAP50':>10}  {'mAP50-95':>10}  {'F1':>8}")
        print(f"  {'-'*40}")
        for r in all_results:
            print(
                f"  {r['fold']:>6}  "
                f"{r['fused'][method]['map50']:>10.4f}  "
                f"{r['fused'][method]['map50_95']:>10.4f}  "
                f"{r['fused'][method]['f1']:>8.4f}"
            )
        print(f"  {'-'*40}")
        print(
            f"  {'Mean':>6}  "
            f"{_mean('fused','map50',method):>10.4f}  "
            f"{_mean('fused','map50_95',method):>10.4f}  "
            f"{_mean('fused','f1',method):>8.4f}"
        )
        print(f"{'='*60}")
    print()

    # ── Per-class summary ────────────────────────────────────────────────
    cat_names = sorted(all_results[0]["per_class"]["thermal"].keys())
    col_w = max(len(c) for c in cat_names) + 2

    pc_configs_print: list[tuple[str, list[dict[str, dict[str, float]]]]] = [
        ("Thermal",     [r["per_class"]["thermal"]     for r in all_results]),
        ("RGB→Thermal", [r["per_class"]["rgb"]         for r in all_results]),
    ]
    for method in fusion_methods:
        pc_configs_print.append((
            f"Fused ({method.upper()})",
            [r["per_class"]["fused"][method] for r in all_results],
        ))

    for cfg_name, fold_pc in pc_configs_print:
        print(f"\n\n{'='*60}")
        print(f"  PER-CLASS — {cfg_name}")
        print(f"{'='*60}")
        header = (
            f"  {'Class':<{col_w}}  {'Instances':>9}  {'mAP@50 Mean':>12}  {'mAP@50 Std':>10}  "
            f"{'mAP@0.5:95 Mean':>15}  {'mAP@0.5:95 Std':>14}  {'F1 Mean':>8}  {'F1 Std':>7}"
        )
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        for cat in cat_names:
            m50  = np.array([fp[cat]["map50"]    for fp in fold_pc])
            m595 = np.array([fp[cat]["map50_95"] for fp in fold_pc])
            f1   = np.array([fp[cat]["f1"]       for fp in fold_pc])
            instances = sum(fp[cat]["instances"] for fp in fold_pc)
            print(
                f"  {cat:<{col_w}}  "
                f"{instances:>9}  "
                f"{m50.mean():>12.4f}  {m50.std():>10.4f}  "
                f"{m595.mean():>15.4f}  {m595.std():>14.4f}  "
                f"{f1.mean():>8.4f}  {f1.std():>7.4f}"
            )
        print(f"{'='*60}")
    print()

    _export_results(all_results)


if __name__ == "__main__":
    main()
