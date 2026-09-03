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
WBF_IOU_THR = 0.55

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
# Fold image-ID loader  (unchanged from the original scaffold)
# ---------------------------------------------------------------------------

def load_fold_image_ids(fold: int) -> list[int]:
    """Return thermal val image IDs for *fold* by reading val_images.txt."""
    tr  = params["training"]
    exp = params["experiments"]["thermal"]

    val_txt = Path(tr["work_dir"]) / f"fold_{fold}" / "val_images.txt"
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
# Per-fold evaluation
# ---------------------------------------------------------------------------

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
            cat_id   = gt_cat_by_name.get(cls_name)
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


def _coco_eval(
    gt_coco: COCO,
    detections: list[CocoDetection],
    image_ids: list[int],
    label: str,
) -> dict[str, float]:
    """Run COCOeval on *detections* restricted to *image_ids*."""
    if not detections:
        print(f"  [{label}] No detections — skipping eval.")
        return {"map50": 0.0, "map50_95": 0.0}

    res    = gt_coco.loadRes(detections)
    ev     = COCOeval(gt_coco, res, "bbox")
    ev.params.imgIds = image_ids
    ev.evaluate()
    ev.accumulate()
    print(f"\n  ── {label} ──")
    ev.summarize()

    return {
        "map50":    float(ev.stats[1]),   # AP @ IoU=0.50
        "map50_95": float(ev.stats[0]),   # AP @ IoU=0.50:0.95
    }


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

    # ── WBF fusion ───────────────────────────────────────────────────────
    print("  Fusing with WBF …")
    thermal_by_img: dict[int, list[CocoDetection]] = defaultdict(list)
    for d in thermal_dets:
        thermal_by_img[d["image_id"]].append(d)

    rgb_by_img: dict[int, list[CocoDetection]] = defaultdict(list)
    for d in rgb_proj_dets:
        rgb_by_img[d["image_id"]].append(d)

    fused_dets: list[CocoDetection] = []
    for img_id in image_ids:
        fused = fuse_detections(
            rgb_detections=rgb_by_img.get(img_id, []),
            thermal_detections=thermal_by_img.get(img_id, []),
            img_w=IMG_W,
            img_h=IMG_H,
            iou_thr=WBF_IOU_THR,
        )
        fused_dets.extend(fused)

    # ── COCO evaluation ──────────────────────────────────────────────────
    m_thermal = _coco_eval(thermal_coco, thermal_dets,   image_ids, f"Fold {fold} – Thermal")
    m_rgb     = _coco_eval(thermal_coco, rgb_proj_dets,  image_ids, f"Fold {fold} – RGB→Thermal")
    m_fused   = _coco_eval(thermal_coco, fused_dets,     image_ids, f"Fold {fold} – Fused (WBF)")

    return {
        "fold":    fold,
        "thermal": m_thermal,
        "rgb":     m_rgb,
        "fused":   m_fused,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yolo_xywh_to_coco(v: list[float]) -> list[float]:
    """YOLO centre-xywh (pixel) → COCO top-left-xywh (pixel)."""
    return [v[0] - v[2] / 2, v[1] - v[3] / 2, v[2], v[3]]


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
    model = YOLO(str(fold_path))
    model.overrides["conf"]         = 0.05
    model.overrides["iou"]          = 0.45
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

    # Load GT annotation sets once
    thermal_coco = COCO(exp_thermal["dataset_file"])
    rgb_coco     = COCO(exp_rgb["dataset_file"])

    all_results: list[dict] = []

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
    print(f"\n\n{'='*70}")
    print("  SUMMARY  (mAP50  |  mAP50-95)")
    print(f"{'='*70}")
    header = f"  {'Fold':>6}  {'Thermal mAP50':>14}  {'Thermal mAP':>11}  "
    header += f"{'RGB mAP50':>10}  {'RGB mAP':>8}  "
    header += f"{'Fused mAP50':>12}  {'Fused mAP':>10}"
    print(header)
    print(f"  {'-'*64}")
    for r in all_results:
        print(
            f"  {r['fold']:>6}  "
            f"{r['thermal']['map50']:>14.4f}  {r['thermal']['map50_95']:>11.4f}  "
            f"{r['rgb']['map50']:>10.4f}  {r['rgb']['map50_95']:>8.4f}  "
            f"{r['fused']['map50']:>12.4f}  {r['fused']['map50_95']:>10.4f}"
        )

    def _mean(key, subkey):
        return np.mean([r[key][subkey] for r in all_results])

    print(f"  {'-'*64}")
    print(
        f"  {'Mean':>6}  "
        f"{_mean('thermal','map50'):>14.4f}  {_mean('thermal','map50_95'):>11.4f}  "
        f"{_mean('rgb','map50'):>10.4f}  {_mean('rgb','map50_95'):>8.4f}  "
        f"{_mean('fused','map50'):>12.4f}  {_mean('fused','map50_95'):>10.4f}"
    )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
