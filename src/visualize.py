from __future__ import annotations

from typing import Sequence

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
import matplotlib.patches as patches
from pycocotools.coco import COCO

from src.bbox_fusion import CocoDetection, fuse_detections
from src.calibration import DATASET_PATH, IMG_W, IMG_H, load_calib, transform_bbox_rgb_to_thermal, transform_bbox_thermal_to_rgb
from src.masks import apply_mask
from src.models import get_thermal_detector, get_rgb_detector
from src.params import params
from src.predict import xywh_yolo_to_coco

# Same name remapping used in evaluate_fused.py
_YOLO_TO_COCO_NAME: dict[str, str] = {
    "motorcycle":   "motorbike",
    "airplane":     "aeroplane",
    "couch":        "sofa",
    "potted plant": "pottedplant",
    "dining table": "diningtable",
    "tv":           "tvmonitor",
}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_boxes(
    ax: maxes.Axes,
    anns: Sequence[dict],
    coco_obj: COCO,
    color: str,
) -> None:
    for ann in anns:
        x, y, w, h = ann["bbox"]
        ax.add_patch(
            patches.Rectangle(
                (x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none"
            )
        )
        cat_name = coco_obj.loadCats(ann["category_id"])[0]["name"]
        score = f' {ann["score"]:.2f}' if "score" in ann else ""
        ax.text(
            x,
            y - 4,
            f"{cat_name}{score}",
            color=color,
            fontsize=7,
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
        )


def _predict(model, img: np.ndarray, gt_cat_by_name: dict[str, int]) -> list[dict]:
    """Run *model* on *img* and return predictions as COCO-style annotation dicts."""
    prediction = model.predict(img, verbose=False, augment=True)[0]
    anns: list[dict] = []
    for box in prediction.boxes:
        cls_name  = model.names[int(box.cls[0])]
        coco_name = _YOLO_TO_COCO_NAME.get(cls_name, cls_name)
        cat_id    = gt_cat_by_name.get(coco_name)  # type: ignore[arg-type]
        if cat_id is None:
            continue
        anns.append({
            "bbox":        xywh_yolo_to_coco(box.xywh.tolist()[0]),
            "category_id": cat_id,
            "score":       float(box.conf[0]),
        })
    return anns


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def visualize(image_id: int) -> None:
    """Show a 2×3 figure for *image_id*:
        [thermal GT | thermal predictions | NMW fused (thermal space)]
        [RGB GT     | RGB predictions     | NMW fused (RGB space)    ]
    """
    exp_thermal = params["experiments"]["thermal"]
    exp_rgb     = params["experiments"]["rgb"]

    thermal_coco = COCO(exp_thermal["dataset_file"])
    rgb_coco     = COCO(exp_rgb["dataset_file"])

    thermal_model = get_thermal_detector()
    rgb_model     = get_rgb_detector()

    # ── Load images ──────────────────────────────────────────────────────
    thermal_meta  = thermal_coco.imgs[image_id]
    thermal_fname = thermal_meta["file_name"]
    thermal_img   = io.imread(f"{exp_thermal['dataset_base_dir']}/{thermal_fname}")
    thermal_img   = apply_mask(thermal_img, DATASET_PATH, thermal_fname, thermal=True)

    rgb_meta  = rgb_coco.imgs[image_id]
    rgb_fname = rgb_meta["file_name"]
    rgb_img   = io.imread(f"{exp_rgb['dataset_base_dir']}/{rgb_fname}")
    rgb_img   = apply_mask(rgb_img, DATASET_PATH, rgb_fname, thermal=False)

    # ── Ground truth ─────────────────────────────────────────────────────
    thermal_gt: list[dict] = thermal_coco.loadAnns(thermal_coco.getAnnIds(imgIds=[image_id]))  # type: ignore[assignment]
    rgb_gt: list[dict]     = rgb_coco.loadAnns(rgb_coco.getAnnIds(imgIds=[image_id]))          # type: ignore[assignment]

    # ── Predictions ───────────────────────────────────────────────────────
    thermal_cat_by_name = {c["name"]: c["id"] for c in thermal_coco.dataset["categories"]}
    rgb_cat_by_name     = {c["name"]: c["id"] for c in rgb_coco.dataset["categories"]}

    thermal_pred = _predict(thermal_model, thermal_img, thermal_cat_by_name)
    rgb_pred     = _predict(rgb_model,     rgb_img,     rgb_cat_by_name)

    # ── NMW fusion (RGB projected to thermal space) ───────────────────────
    calib = load_calib(thermal_fname)   # thermal_fname encodes scene/clip
    inf   = params["inference"]

    thermal_dets: list[CocoDetection] = [
        CocoDetection(image_id=image_id, category_id=d["category_id"],
                      bbox=d["bbox"], score=d["score"])
        for d in thermal_pred
    ]
    rgb_proj_dets: list[CocoDetection] = [
        CocoDetection(image_id=image_id, category_id=d["category_id"],
                      bbox=transform_bbox_rgb_to_thermal(d["bbox"], calib),
                      score=d["score"])
        for d in rgb_pred
    ]
    fused_dets = fuse_detections(
        rgb_detections=rgb_proj_dets,
        thermal_detections=thermal_dets,
        img_w=IMG_W,
        img_h=IMG_H,
        iou_thr=inf["fusion_iou_thr"],
        method="nmw",
    )

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    (ax_t_gt, ax_t_pred, ax_t_fused), (ax_r_gt, ax_r_pred, ax_r_fused) = axes

    ax_t_gt.imshow(thermal_img)
    ax_t_gt.set_title("Thermal — Ground Truth")
    ax_t_gt.axis("off")
    _draw_boxes(ax_t_gt, thermal_gt, thermal_coco, color="lime")

    ax_t_pred.imshow(thermal_img)
    ax_t_pred.set_title("Thermal — Predictions")
    ax_t_pred.axis("off")
    _draw_boxes(ax_t_pred, thermal_pred, thermal_coco, color="deepskyblue")

    ax_t_fused.imshow(thermal_img)
    ax_t_fused.set_title("Fused NMW (thermal space)")
    ax_t_fused.axis("off")
    _draw_boxes(ax_t_fused, fused_dets, thermal_coco, color="red")  # type: ignore[arg-type]

    ax_r_gt.imshow(rgb_img)
    ax_r_gt.set_title("RGB — Ground Truth")
    ax_r_gt.axis("off")
    _draw_boxes(ax_r_gt, rgb_gt, rgb_coco, color="lime")

    ax_r_pred.imshow(rgb_img)
    ax_r_pred.set_title("RGB — Predictions")
    ax_r_pred.axis("off")
    _draw_boxes(ax_r_pred, rgb_pred, rgb_coco, color="deepskyblue")

    fused_dets_rgb: list[dict] = [
        {**d, "bbox": transform_bbox_thermal_to_rgb(d["bbox"], calib)}
        for d in fused_dets
    ]
    ax_r_fused.imshow(rgb_img)
    ax_r_fused.set_title("Fused NMW (RGB space)")
    ax_r_fused.axis("off")
    _draw_boxes(ax_r_fused, fused_dets_rgb, thermal_coco, color="red")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    thermal_coco = COCO(params["experiments"]["thermal"]["dataset_file"])
    print(f"Available image IDs: 0 – {max(thermal_coco.imgs)}")

    while True:
        image_id = int(input("Image ID: ").strip())
        visualize(image_id)
