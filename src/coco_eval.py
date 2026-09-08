from __future__ import annotations

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.bbox_fusion import CocoDetection


# ---------------------------------------------------------------------------
# F1 helpers
# ---------------------------------------------------------------------------

def f1_from_eval(ev: COCOeval) -> float:
    """Compute max-F1 at IoU=0.50 from an already-accumulated COCOeval.

    ``ev.eval["precision"]`` has shape [T, R, K, A, M]:
      T = 10 IoU thresholds (0.50 … 0.95), index 0 → IoU=0.50
      R = 101 recall points (0.00 … 1.00)
      K = categories
      A = area ranges
      M = max-det thresholds

    Averages precision over categories (K), takes area=all (A=0) and the
    largest max-det slot (M=-1), then finds the recall point that maximises F1.
    """
    prec = ev.eval["precision"][0, :, :, 0, -1]   # IoU=0.50, area=all, maxDets=largest
    if (prec > -1).sum() == 0:
        return 0.0
    prec_mean = prec.mean(axis=1)                  # shape [R]
    recall_pts = np.linspace(0.0, 1.0, len(prec_mean))
    valid = prec_mean > -1
    if not valid.any():
        return 0.0
    p = prec_mean[valid]
    r = recall_pts[valid]
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
    return float(f1.max())


def f1_from_eval_single_cat(ev: COCOeval) -> float:
    """Same as ``f1_from_eval`` but for a single-category COCOeval (K=1)."""
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


# ---------------------------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------------------------

def coco_eval(
    gt_coco: COCO,
    detections: list[CocoDetection],
    image_ids: list[int],
    label: str,
) -> dict[str, float]:
    """Run COCOeval on *detections* restricted to *image_ids*.

    Returns a dict with keys: map50, map50_95, recall, f1.
    """
    if not detections:
        print(f"  [{label}] No detections — skipping eval.")
        return {"map50": 0.0, "map50_95": 0.0, "recall": 0.0, "f1": 0.0}

    res = gt_coco.loadRes(detections)
    ev  = COCOeval(gt_coco, res, "bbox")
    ev.params.imgIds  = image_ids
    ev.params.maxDets = [1, 10, 100, 1000]  # 1000 = YOLO max_det; 100 kept for summarize default
    ev.evaluate()
    ev.accumulate()
    print(f"\n  ── {label} ──")
    ev.summarize()

    return {
        "map50":    float(ev.stats[1]),   # AP  @ IoU=0.50
        "map50_95": float(ev.stats[0]),   # AP  @ IoU=0.50:0.95
        "recall":   float(ev.stats[8]),   # AR  @ IoU=0.50:0.95, maxDets=1000
        "f1":       f1_from_eval(ev),     # max-F1 @ IoU=0.50
    }


# ---------------------------------------------------------------------------
# Per-class evaluation
# ---------------------------------------------------------------------------

def coco_eval_per_class(
    gt_coco: COCO,
    detections: list[CocoDetection],
    image_ids: list[int],
) -> dict[str, dict[str, float]]:
    """Return per-category AP metrics keyed by category name.

    For each category runs a separate COCOeval restricted to that category
    and extracts mAP@50, mAP@[0.5:0.95], and max-F1@IoU=0.50.
    AP values are read directly from ``ev.eval["precision"]`` to avoid
    calling summarize() (which would flood stdout).
    """
    cat_id_to_name = {c["id"]: c["name"] for c in gt_coco.dataset["categories"]}

    # Count GT instances per category restricted to the evaluated image set.
    image_id_set = set(image_ids)
    instance_counts: dict[int, int] = {cat_id: 0 for cat_id in cat_id_to_name}
    for ann in gt_coco.dataset.get("annotations", []):
        if ann["image_id"] in image_id_set:
            cat = ann["category_id"]
            if cat in instance_counts:
                instance_counts[cat] += 1

    if not detections:
        return {
            name: {"map50": 0.0, "map50_95": 0.0, "f1": 0.0,
                   "instances": instance_counts[cat_id]}
            for cat_id, name in cat_id_to_name.items()
        }

    res = gt_coco.loadRes(detections)
    results: dict[str, dict[str, float]] = {}

    for cat_id, cat_name in cat_id_to_name.items():
        ev = COCOeval(gt_coco, res, "bbox")
        ev.params.imgIds  = image_ids
        ev.params.catIds  = [cat_id]
        ev.params.maxDets = [1, 10, 100, 1000]
        ev.evaluate()
        ev.accumulate()

        # precision shape: [T, R, K, A, M]  (K=1 here)
        prec = ev.eval["precision"][:, :, 0, 0, -1]  # [T, R]
        valid_all = prec[prec > -1]
        map50_95 = float(np.mean(valid_all)) if valid_all.size > 0 else 0.0
        valid50  = prec[0][prec[0] > -1]
        map50    = float(np.mean(valid50))   if valid50.size  > 0 else 0.0

        results[cat_name] = {
            "map50":    map50,
            "map50_95": map50_95,
            "f1":       f1_from_eval_single_cat(ev),
            "instances": instance_counts[cat_id],
        }

    return results
