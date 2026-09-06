"""
bbox_fusion.py

Fuses RGB and thermal bounding-box predictions using one of four methods from
the ensemble-boxes library:

  - "wbf"      Weighted Boxes Fusion (default)
  - "nms"      Non-Maximum Suppression
  - "soft_nms" Soft-NMS
  - "nmw"      Non-Maximum Weighted

Prediction files are the JSON outputs produced by predict.py, where every
detection is a COCO-style dict:
    {"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}

All bbox coordinates are in pixel space (COCO top-left xywh).
ensemble-boxes requires normalised [x_min, y_min, x_max, y_max], so we convert
before calling and invert the conversion on the output.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from ensemble_boxes import (
    non_maximum_weighted,
    nms,
    soft_nms,
    weighted_boxes_fusion,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

FusionMethod = Literal["wbf", "nms", "soft_nms", "nmw"]

class CocoDetection(TypedDict):
    image_id: int
    category_id: int
    bbox: list[float]   # COCO xywh, pixel space
    score: float


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _coco_to_norm_xyxy(
    bbox: list[float], img_w: float, img_h: float
) -> list[float]:
    """COCO [x, y, w, h] → normalised [x1, y1, x2, y2], clamped to [0, 1]."""
    x, y, w, h = bbox
    x1 = max(0.0, x / img_w)
    y1 = max(0.0, y / img_h)
    x2 = min(1.0, (x + w) / img_w)
    y2 = min(1.0, (y + h) / img_h)
    return [x1, y1, x2, y2]


def _norm_xyxy_to_coco(
    box: list[float], img_w: float, img_h: float
) -> list[float]:
    """Normalised [x1, y1, x2, y2] → COCO [x, y, w, h] in pixel space."""
    x1, y1, x2, y2 = box
    x = x1 * img_w
    y = y1 * img_h
    w = (x2 - x1) * img_w
    h = (y2 - y1) * img_h
    return [x, y, w, h]


# ---------------------------------------------------------------------------
# Core fusion function
# ---------------------------------------------------------------------------

def fuse_detections(
    rgb_detections: list[CocoDetection],
    thermal_detections: list[CocoDetection],
    img_w: float = 640.0,
    img_h: float = 480.0,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    weights: list[float] | None = None,
    method: FusionMethod = "wbf",
    soft_nms_sigma: float = 0.5,
    soft_nms_thresh: float = 0.001,
) -> list[CocoDetection]:
    """Fuse RGB and thermal detections for a *single image*.

    Parameters
    ----------
    rgb_detections:
        Detections from the RGB model for one image (COCO xywh, pixel space).
    thermal_detections:
        Detections from the thermal model for the same image.
    img_w, img_h:
        Image dimensions used for normalisation (default: 640×480).
    iou_thr:
        IoU threshold for box merging / suppression.
    skip_box_thr:
        Boxes with confidence below this value are ignored before fusion.
        Only used by WBF and NMW; ignored by NMS / Soft-NMS.
    weights:
        Per-model confidence weights; defaults to [1, 1] (equal weight).
    method:
        Fusion algorithm: ``"wbf"`` (default), ``"nms"``, ``"soft_nms"``,
        or ``"nmw"``.
    soft_nms_sigma:
        Sigma for Gaussian Soft-NMS decay (only used when method="soft_nms").
    soft_nms_thresh:
        Score threshold for Soft-NMS output boxes (only used when
        method="soft_nms").

    Returns
    -------
    list[CocoDetection]
        Fused detections.  ``image_id`` and ``category_id`` are carried over
        from the source detections.
    """
    if weights is None:
        weights = [1.0, 1.0]

    # ensemble-boxes fuses per-class; we must split by category and reassemble
    all_fused: list[CocoDetection] = []

    # Derive image_id from whichever list is non-empty
    sample = (rgb_detections or thermal_detections)
    if not sample:
        return []
    image_id: int = sample[0]["image_id"]

    # Collect all category ids present in either modality
    category_ids = {d["category_id"] for d in rgb_detections + thermal_detections}

    for cat_id in category_ids:
        rgb_cat  = [d for d in rgb_detections     if d["category_id"] == cat_id]
        ther_cat = [d for d in thermal_detections if d["category_id"] == cat_id]

        # Build the two per-model input lists required by ensemble-boxes
        boxes_list:  list[list[list[float]]] = []
        scores_list: list[list[float]]       = []
        labels_list: list[list[int]]         = []

        for dets in (rgb_cat, ther_cat):
            if not dets:
                continue
            boxes_list.append([
                _coco_to_norm_xyxy(d["bbox"], img_w, img_h) for d in dets
            ])
            scores_list.append([d["score"] for d in dets])
            labels_list.append([d["category_id"] for d in dets])

        if not boxes_list:
            continue

        # Trim weights to match the number of models that actually contributed
        # boxes for this category; ensemble-boxes rejects a mismatched list.
        effective_weights = weights[: len(boxes_list)]

        if method == "wbf":
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=effective_weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        elif method == "nms":
            fused_boxes, fused_scores, fused_labels = nms(
                boxes_list,
                scores_list,
                labels_list,
                iou_thr=iou_thr,
                weights=effective_weights,
            )
        elif method == "soft_nms":
            fused_boxes, fused_scores, fused_labels = soft_nms(
                boxes_list,
                scores_list,
                labels_list,
                iou_thr=iou_thr,
                sigma=soft_nms_sigma,
                thresh=soft_nms_thresh,
                weights=effective_weights,
            )
        elif method == "nmw":
            fused_boxes, fused_scores, fused_labels = non_maximum_weighted(
                boxes_list,
                scores_list,
                labels_list,
                weights=effective_weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        else:
            raise ValueError(
                f"Unknown fusion method {method!r}. "
                "Choose from: 'wbf', 'nms', 'soft_nms', 'nmw'."
            )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            all_fused.append(
                CocoDetection(
                    image_id=image_id,
                    category_id=int(label),
                    bbox=_norm_xyxy_to_coco(box.tolist(), img_w, img_h),
                    score=float(score),
                )
            )

    return all_fused


# ---------------------------------------------------------------------------
# Batch helper: fuse full prediction files
# ---------------------------------------------------------------------------

def fuse_prediction_files(
    rgb_path: str | Path,
    thermal_path: str | Path,
    output_path: str | Path,
    img_w: float = 640.0,
    img_h: float = 480.0,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    weights: list[float] | None = None,
    method: FusionMethod = "wbf",
    soft_nms_sigma: float = 0.5,
    soft_nms_thresh: float = 0.001,
) -> list[CocoDetection]:
    """Load two COCO-format prediction JSON files, fuse them per image, and
    write the fused results to *output_path*.

    Parameters
    ----------
    rgb_path:
        Path to the JSON file produced by ``predict.py`` for the RGB model.
    thermal_path:
        Path to the JSON file produced by ``predict.py`` for the thermal model.
    output_path:
        Destination JSON file for the fused detections.
    img_w, img_h:
        Image dimensions used for coordinate normalisation.
    iou_thr:
        IoU threshold forwarded to :func:`fuse_detections`.
    skip_box_thr:
        Minimum score threshold forwarded to :func:`fuse_detections`.
    weights:
        Per-model weights forwarded to :func:`fuse_detections`.
    method:
        Fusion algorithm forwarded to :func:`fuse_detections`.
    soft_nms_sigma:
        Sigma forwarded to :func:`fuse_detections` (Soft-NMS only).
    soft_nms_thresh:
        Score threshold forwarded to :func:`fuse_detections` (Soft-NMS only).

    Returns
    -------
    list[CocoDetection]
        All fused detections across every image.
    """
    with open(rgb_path) as f:
        rgb_dets: list[CocoDetection] = json.load(f)
    with open(thermal_path) as f:
        thermal_dets: list[CocoDetection] = json.load(f)

    # Group by image_id for O(n) per-image processing
    rgb_by_image: dict[int, list[CocoDetection]] = defaultdict(list)
    for d in rgb_dets:
        rgb_by_image[d["image_id"]].append(d)

    thermal_by_image: dict[int, list[CocoDetection]] = defaultdict(list)
    for d in thermal_dets:
        thermal_by_image[d["image_id"]].append(d)

    all_image_ids = set(rgb_by_image) | set(thermal_by_image)

    all_fused: list[CocoDetection] = []
    for img_id in sorted(all_image_ids):
        fused = fuse_detections(
            rgb_detections=rgb_by_image.get(img_id, []),
            thermal_detections=thermal_by_image.get(img_id, []),
            img_w=img_w,
            img_h=img_h,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            weights=weights,
            method=method,
            soft_nms_sigma=soft_nms_sigma,
            soft_nms_thresh=soft_nms_thresh,
        )
        all_fused.extend(fused)

    with open(output_path, "w") as f:
        json.dump(all_fused, f)

    print(
        f"Fused {len(rgb_dets)} RGB + {len(thermal_dets)} thermal detections "
        f"→ {len(all_fused)} fused detections written to {output_path} "
        f"(method={method})"
    )
    return all_fused


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.params import params

    rgb_path     = params["experiments"]["rgb"]["output_path"]
    thermal_path = params["experiments"]["thermal"]["output_path"]
    output_path  = "fused_predictions.json"
    inf          = params["inference"]

    fuse_prediction_files(
        rgb_path=rgb_path,
        thermal_path=thermal_path,
        output_path=output_path,
        iou_thr=inf["wbf_iou_thr"],
        method=inf["fusion_method"],
        soft_nms_sigma=inf.get("soft_nms_sigma", 0.5),
        soft_nms_thresh=inf.get("soft_nms_thresh", 0.001),
    )
