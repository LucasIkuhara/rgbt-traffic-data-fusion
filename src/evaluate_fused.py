"""
evaluate_fused.py

For each k-fold split:
  1. Run the fold's fine-tuned thermal model on the thermal val images.
  2. Run the shared RGB model on the paired RGB val images; project the
     resulting bounding boxes into thermal camera space using the per-clip
     homography + distortion calibration (aauRainSnowUtility).
  3. Fuse the thermal and projected-RGB detections with Weighted Boxes
     Fusion (WBF).
  4. Evaluate all three sets of detections against the thermal-space ground
     truth with COCO mAP (IoU 0.50 : 0.95) and mAP50.

Usage:
    python -m src.evaluate_fused
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from src.bbox_fusion import CocoDetection, fuse_detections
from src.masks import apply_mask
from src.params import params

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_PATH = Path("aau-rainsnow")
IMG_W, IMG_H = 640.0, 480.0

# COCO-2014 annotation names that differ from YOLOv8's COCO80 class names.
# Maps YOLO prediction name → COCO category name used in the annotation file.
_YOLO_TO_COCO_NAME: dict[str, str] = {
    "motorcycle":  "motorbike",
    "airplane":    "aeroplane",
    "couch":       "sofa",
    "potted plant":"pottedplant",
    "dining table":"diningtable",
    "tv":          "tvmonitor",
}

# ---------------------------------------------------------------------------
# Calibration helpers  (based on aauRainSnowUtility.py)
# ---------------------------------------------------------------------------

def _load_calib(file_name: str) -> dict:
    """Load the calib.yml for the clip that contains *file_name*.

    file_name is relative to the dataset root, e.g.
        'Egensevej/Egensevej-1/cam2-00055.png'
    Returns a dict with keys: homCam1Cam2, cam1CamMat, cam1DistCoeff,
                               cam2CamMat, cam2DistCoeff
    """
    parts = Path(file_name).parts        # (scene, clip, imgfile)
    scene, clip = parts[0], parts[1]
    calib_path = DATASET_PATH / scene / f"{clip}-calib.yml"
    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
    return {
        "homCam1Cam2":    fs.getNode("homCam1Cam2").mat(),
        "homCam2Cam1":    fs.getNode("homCam2Cam1").mat(),
        "cam1CamMat":     fs.getNode("cam1CamMat").mat(),
        "cam2CamMat":     fs.getNode("cam2CamMat").mat(),
        "cam1DistCoeff":  fs.getNode("cam1DistCoeff").mat(),
        "cam2DistCoeff":  fs.getNode("cam2DistCoeff").mat(),
    }


def _register_points_rgb_to_thermal(
    points: np.ndarray,          # (N, 2) float32
    calib: dict,
) -> np.ndarray:
    """Project points from RGB (cam1) into thermal (cam2) space.

    Steps mirror aauRainSnowUtility.registerRgbPointsToThermal:
      1. Undistort with cam1 intrinsics
      2. Apply homCam1Cam2
      3. Re-distort with cam2 intrinsics
    """
    pts = points.astype(np.float64).reshape(-1, 1, 2)

    # 1. Undistort
    undist = cv2.undistortPoints(
        pts,
        calib["cam1CamMat"],
        calib["cam1DistCoeff"],
        P=calib["cam1CamMat"],
    )

    # 2. Homography
    proj = cv2.perspectiveTransform(undist, calib["homCam1Cam2"])  # (N,1,2)

    # 3. Re-distort: normalise by cam2 intrinsics, call projectPoints with
    #    zero rotation/translation so it only applies distortion
    K2 = calib["cam2CamMat"]
    D2 = calib["cam2DistCoeff"]
    normalised = []
    for pt in proj[:, 0, :]:
        nx = (pt[0] - K2[0, 2]) / K2[0, 0]
        ny = (pt[1] - K2[1, 2]) / K2[1, 1]
        normalised.append([nx, ny, 1.0])

    distorted, _ = cv2.projectPoints(
        np.array(normalised, dtype=np.float32).reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        K2,
        D2,
    )
    return distorted.reshape(-1, 2)  # (N, 2)


def _transform_bbox_rgb_to_thermal(
    bbox: list[float],   # COCO [x, y, w, h] in RGB pixel space
    rgb_file_name: str,  # used to locate the calib file
    calib: dict,
) -> list[float]:
    """Project a COCO bbox from RGB space to thermal space.

    We project all four corners, take the axis-aligned bounding box of the
    projected corners, and clamp to the image bounds.
    """
    x, y, w, h = bbox
    corners = np.array([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h],
    ], dtype=np.float32)

    projected = _register_points_rgb_to_thermal(corners, calib)

    x1 = float(np.clip(projected[:, 0].min(), 0, IMG_W))
    y1 = float(np.clip(projected[:, 1].min(), 0, IMG_H))
    x2 = float(np.clip(projected[:, 0].max(), 0, IMG_W))
    y2 = float(np.clip(projected[:, 1].max(), 0, IMG_H))

    return [x1, y1, x2 - x1, y2 - y1]


# ---------------------------------------------------------------------------
# COCO annotation cleaner
# ---------------------------------------------------------------------------

def _remove_degenerate_annotations(coco: COCO) -> None:
    """Remove annotations with empty segmentation (area=0, invalid bbox).

    These are present in the aauRainSnow dataset and are skipped during
    training (write_labels guards on segmentation), but COCOeval counts them
    as ground-truth misses if left in, artificially suppressing recall/AP.
    Mutates the COCO object in-place and rebuilds its index.
    """
    valid = [a for a in coco.dataset["annotations"] if a.get("segmentation")]
    removed = len(coco.dataset["annotations"]) - len(valid)
    if removed:
        print(f"  [coco_clean] removed {removed} degenerate annotations (no segmentation)")
    coco.dataset["annotations"] = valid
    coco.createIndex()


def _realign_gt_bboxes_to_rle(coco: COCO) -> None:
    """Replace each annotation's stored bbox with the one derived from its
    segmentation mask (via maskUtils.toBbox), matching what write_labels writes
    to the YOLO .txt training files.

    The aauRainSnow JSON stores float bboxes that can differ from the
    integer-pixel RLE-derived bbox by up to 208px (IoU < 0.50 for ~5% of GT).
    COCOeval uses ann['bbox'] for GT matching, so without this alignment the
    model is scored against different boxes than it was trained on, artificially
    suppressing AP.
    """
    from pycocotools import mask as maskUtils

    for ann in coco.dataset["annotations"]:
        if not ann.get("segmentation"):
            continue
        rle  = coco.annToRLE(ann)
        x, y, w, h = maskUtils.toBbox(rle).tolist()
        if w > 0 and h > 0:
            ann["bbox"] = [x, y, w, h]
            ann["area"] = float(w * h)

    coco.createIndex()


# ---------------------------------------------------------------------------
# Fold image-ID loader  (unchanged from the original scaffold)
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
    """Run *model* on either thermal or RGB images; return
    (thermal_space_detections, raw_detections).

    For the thermal modality the two lists are identical.
    For RGB the first list has bboxes projected into thermal space via
    the per-clip calibration homography.
    """
    raw_dets: list[CocoDetection]  = []
    proj_dets: list[CocoDetection] = []

    for img_id in image_ids:
        thermal_meta = thermal_coco.imgs[img_id]
        thermal_fname = thermal_meta["file_name"]   # e.g. Egensevej/Eg-1/cam2-*.png

        if modality == "thermal":
            img_meta  = thermal_meta
            file_name = thermal_fname
            db_base   = dataset_base_thermal
            is_thermal = True
        else:
            img_meta  = rgb_coco.imgs[img_id]
            file_name = img_meta["file_name"]
            db_base   = dataset_base_rgb
            is_thermal = False

        img_data = io.imread(f"{db_base}/{file_name}")
        img_data = apply_mask(img_data, DATASET_PATH, file_name, thermal=is_thermal)

        prediction = model.predict(img_data, verbose=False, augment=True)[0]

        # Load calib once per image for RGB→thermal projection
        calib = None
        if modality == "rgb":
            calib = _load_calib(file_name)

        for box in prediction.boxes:
            cls_name = model.names[int(box.cls[0])]
            coco_name = _YOLO_TO_COCO_NAME.get(cls_name, cls_name)
            cat_id   = gt_cat_by_name.get(coco_name)
            if cat_id is None:
                continue

            bbox_raw  = _yolo_xywh_to_coco(box.xywh.tolist()[0])
            score     = float(box.conf[0])

            det_raw = CocoDetection(
                image_id=img_id,
                category_id=cat_id,
                bbox=bbox_raw,
                score=score,
            )
            raw_dets.append(det_raw)

            if modality == "rgb":
                bbox_proj = _transform_bbox_rgb_to_thermal(bbox_raw, file_name, calib)
                proj_dets.append(CocoDetection(
                    image_id=img_id,
                    category_id=cat_id,
                    bbox=bbox_proj,
                    score=score,
                ))
            else:
                proj_dets.append(det_raw)

    return proj_dets, raw_dets


def _f1_from_eval(ev: COCOeval) -> float:
    """Compute max-F1 at IoU=0.50 from an already-accumulated COCOeval.

    ``ev.eval["precision"]`` has shape [T, R, K, A, M]:
      T = 10 IoU thresholds (0.50 … 0.95), index 0 → IoU=0.50
      R = 101 recall points (0.00 … 1.00)
      K = categories
      A = area ranges
      M = max-det thresholds

    We average over categories (K), take area=all (A=0) and the largest
    max-det slot (M=-1), then find the recall point that maximises F1.
    Recall values mirror the 101 linearly-spaced points [0, 0.01, …, 1.0].
    """
    # precision[0, :, :, 0, -1] → shape [R, K]; -1 means "no detection" slots
    prec = ev.eval["precision"][0, :, :, 0, -1]   # IoU=0.50, area=all, maxDets=largest
    prec = prec[prec > -1]                          # remove unset entries
    if prec.size == 0:
        return 0.0
    # Mean precision over categories at each recall point
    prec_mean = ev.eval["precision"][0, :, :, 0, -1].mean(axis=1)  # shape [R]
    recall_pts = np.linspace(0.0, 1.0, len(prec_mean))
    valid = prec_mean > -1
    if not valid.any():
        return 0.0
    p = prec_mean[valid]
    r = recall_pts[valid]
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
    return float(f1.max())


def _f1_from_eval_single_cat(ev: COCOeval) -> float:
    """Same as ``_f1_from_eval`` but for a single-category COCOeval (K=1)."""
    prec_curve = ev.eval["precision"][0, :, 0, 0, -1]   # shape [R]
    recall_pts = np.linspace(0.0, 1.0, len(prec_curve))
    valid = prec_curve > -1
    if not valid.any():
        return 0.0
    p = prec_curve[valid]
    r = recall_pts[valid]
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
    return float(f1.max())


def _coco_eval(
    gt_coco: COCO,
    detections: list[CocoDetection],
    image_ids: list[int],
    label: str,
) -> dict[str, float]:
    """Run COCOeval on *detections* restricted to *image_ids*."""
    if not detections:
        print(f"  [{label}] No detections — skipping eval.")
        return {"map50": 0.0, "map50_95": 0.0, "recall": 0.0, "f1": 0.0}

    res    = gt_coco.loadRes(detections)
    ev     = COCOeval(gt_coco, res, "bbox")
    ev.params.imgIds  = image_ids
    ev.params.maxDets = [1, 10, 100, 1000]  # 1000 matches YOLO's max_det; 100 kept so _summarize(default) stays valid
    ev.evaluate()
    ev.accumulate()
    print(f"\n  ── {label} ──")
    ev.summarize()

    return {
        "map50":    float(ev.stats[1]),   # AP  @ IoU=0.50
        "map50_95": float(ev.stats[0]),   # AP  @ IoU=0.50:0.95
        "recall":   float(ev.stats[8]),   # AR  @ IoU=0.50:0.95, maxDets=1000
        "f1":       _f1_from_eval(ev),    # max-F1 @ IoU=0.50
    }


def _coco_eval_per_class(
    gt_coco: COCO,
    detections: list[CocoDetection],
    image_ids: list[int],
) -> dict[str, dict[str, float]]:
    """Return per-category AP metrics keyed by category name.

    For each category we run a separate COCOeval restricted to that category
    and extract mAP@50, mAP@[0.5:0.95], and max-F1@IoU=0.50.
    """
    results: dict[str, dict[str, float]] = {}
    cat_id_to_name = {c["id"]: c["name"] for c in gt_coco.dataset["categories"]}

    if not detections:
        return {name: {"map50": 0.0, "map50_95": 0.0, "f1": 0.0} for name in cat_id_to_name.values()}

    res = gt_coco.loadRes(detections)

    for cat_id, cat_name in cat_id_to_name.items():
        ev = COCOeval(gt_coco, res, "bbox")
        ev.params.imgIds  = image_ids
        ev.params.catIds  = [cat_id]
        ev.params.maxDets = [1, 10, 100, 1000]
        ev.evaluate()
        ev.accumulate()
        results[cat_name] = {
            "map50":    float(ev.stats[1]),
            "map50_95": float(ev.stats[0]),
            "f1":       _f1_from_eval_single_cat(ev),
        }

    return results


def evaluate_fold(
    fold: int,
    thermal_model: YOLO,
    rgb_model: YOLO,
    thermal_coco: COCO,
    rgb_coco: COCO,
) -> dict:
    """Evaluate one fold; returns a metrics dict."""
    tr  = params["training"]
    exp_thermal = params["experiments"]["thermal"]
    exp_rgb     = params["experiments"]["rgb"]

    dataset_base_thermal = exp_thermal["dataset_base_dir"]
    dataset_base_rgb     = exp_rgb["dataset_base_dir"]

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
        dataset_base_thermal=dataset_base_thermal,
        dataset_base_rgb=dataset_base_rgb,
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
        dataset_base_thermal=dataset_base_thermal,
        dataset_base_rgb=dataset_base_rgb,
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
    m_thermal = _coco_eval(thermal_coco, thermal_dets,  image_ids, f"Fold {fold} – Thermal")
    m_rgb     = _coco_eval(thermal_coco, rgb_proj_dets, image_ids, f"Fold {fold} – RGB→Thermal")
    m_fused   = {
        method: _coco_eval(thermal_coco, fused_by_method[method], image_ids,
                           f"Fold {fold} – Fused ({method.upper()})")
        for method in fusion_methods
    }

    pc_thermal = _coco_eval_per_class(thermal_coco, thermal_dets,  image_ids)
    pc_rgb     = _coco_eval_per_class(thermal_coco, rgb_proj_dets, image_ids)
    pc_fused   = {
        method: _coco_eval_per_class(thermal_coco, fused_by_method[method], image_ids)
        for method in fusion_methods
    }

    return {
        "fold":           fold,
        "thermal":        m_thermal,
        "rgb":            m_rgb,
        "fused":          m_fused,          # dict keyed by method name
        "per_class":      {
            "thermal": pc_thermal,
            "rgb":     pc_rgb,
            "fused":   pc_fused,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yolo_xywh_to_coco(v: list[float]) -> list[float]:
    """YOLO centre-xywh (pixel) → COCO top-left-xywh (pixel)."""
    return [v[0] - v[2] / 2, v[1] - v[3] / 2, v[2], v[3]]


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
        ("Thermal",     [r["thermal"]         for r in all_results]),
        ("RGB→Thermal", [r["rgb"]             for r in all_results]),
    ]
    for method in fusion_methods:
        configs.append((f"Fused ({method.upper()})", [r["fused"][method] for r in all_results]))

    rows = []
    for name, fold_metrics in configs:
        row: dict = {"Configuration": name}
        for metric in metrics:
            values = np.array([m[metric] for m in fold_metrics])
            col_label = {"map50": "mAP@50", "map50_95": "mAP@[0.5:0.95]", "recall": "Recall", "f1": "F1"}[metric]
            row[f"{col_label} Mean"] = round(float(values.mean()), 4)
            row[f"{col_label} Std"]  = round(float(values.std()),  4)
        rows.append(row)

    df_overview = pd.DataFrame(rows)

    # ── Sheet 2: per-class ───────────────────────────────────────────────
    cat_names = sorted(all_results[0]["per_class"]["thermal"].keys())

    pc_configs: list[tuple[str, list[dict[str, dict[str, float]]]]] = [
        ("Thermal",     [r["per_class"]["thermal"]         for r in all_results]),
        ("RGB→Thermal", [r["per_class"]["rgb"]             for r in all_results]),
    ]
    for method in fusion_methods:
        pc_configs.append((
            f"Fused ({method.upper()})",
            [r["per_class"]["fused"][method] for r in all_results],
        ))

    pc_rows = []
    for name, fold_pc in pc_configs:
        for cat in cat_names:
            row = {"Configuration": name, "Class": cat}
            for metric in ("map50", "map50_95", "f1"):
                values = np.array([fp[cat][metric] for fp in fold_pc])
                col_label = {"map50": "mAP@50", "map50_95": "mAP@[0.5:0.95]", "f1": "F1"}[metric]
                row[f"{col_label} Mean"] = round(float(values.mean()), 4)
                row[f"{col_label} Std"]  = round(float(values.std()),  4)
            pc_rows.append(row)

    df_per_class = pd.DataFrame(pc_rows)

    with pd.ExcelWriter(path) as writer:
        df_overview.to_excel(writer,   sheet_name="Overview",   index=False)
        df_per_class.to_excel(writer,  sheet_name="Per-Class",  index=False)

    print(f"\n  [export] Results written to {path}")


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
    tr  = params["training"]
    exp_thermal = params["experiments"]["thermal"]
    exp_rgb     = params["experiments"]["rgb"]

    n_splits = tr["n_splits"]

    # Load GT annotation sets once; strip degenerate annotations and optionally
    # realign GT bboxes to match the RLE-derived boxes written to YOLO labels.
    thermal_coco = COCO(exp_thermal["dataset_file"])
    rgb_coco     = COCO(exp_rgb["dataset_file"])
    _remove_degenerate_annotations(thermal_coco)
    _remove_degenerate_annotations(rgb_coco)
    if params["inference"]["gt_bbox_source"] == "rle":
        print("  [gt_bbox] using RLE-derived bboxes (from segmentation masks)")
        _realign_gt_bboxes_to_rle(thermal_coco)
        _realign_gt_bboxes_to_rle(rgb_coco)
    else:
        print("  [gt_bbox] using original stored bboxes (ann['bbox'] from JSON)")

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
        ("Thermal",     [r["per_class"]["thermal"]         for r in all_results]),
        ("RGB→Thermal", [r["per_class"]["rgb"]             for r in all_results]),
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
        header = f"  {'Class':<{col_w}}  {'mAP@50 Mean':>12}  {'mAP@50 Std':>10}  {'mAP@0.5:95 Mean':>15}  {'mAP@0.5:95 Std':>14}  {'F1 Mean':>8}  {'F1 Std':>7}"
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        for cat in cat_names:
            m50  = np.array([fp[cat]["map50"]    for fp in fold_pc])
            m595 = np.array([fp[cat]["map50_95"] for fp in fold_pc])
            f1   = np.array([fp[cat]["f1"]       for fp in fold_pc])
            print(
                f"  {cat:<{col_w}}  "
                f"{m50.mean():>12.4f}  {m50.std():>10.4f}  "
                f"{m595.mean():>15.4f}  {m595.std():>14.4f}  "
                f"{f1.mean():>8.4f}  {f1.std():>7.4f}"
            )
        print(f"{'='*60}")
    print()

    _export_results(all_results)


if __name__ == "__main__":
    main()
