import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
import matplotlib.patches as patches
from pycocotools import coco as cocolib
from src.models import MODELS
from src.params import params
from src.predict import xywh_yolo_to_coco


def draw_boxes(ax: maxes.Axes, anns: list[dict], coco_obj: cocolib.COCO, color: str, label_prefix: str = "") -> None:
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
            f"{label_prefix}{cat_name}{score}",
            color=color,
            fontsize=7,
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
        )


def plot(img: np.ndarray, gt_anns: list[dict], pred_anns: list[dict], coco_obj: cocolib.COCO, title: str, pred_label: str) -> None:
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=9)

    ax_gt.imshow(img)
    ax_gt.set_title("Ground truth")
    ax_gt.axis("off")
    draw_boxes(ax_gt, gt_anns, coco_obj, color="lime")

    ax_pred.imshow(img)
    ax_pred.set_title(f"Predictions  ({pred_label})")
    ax_pred.axis("off")
    draw_boxes(ax_pred, pred_anns, coco_obj, color="red")

    plt.tight_layout()
    plt.show()


def visualize(exp_id: str, image_id: int) -> None:
    experiment = params["experiments"][exp_id]
    dataset_base_dir = experiment["dataset_base_dir"]
    model = MODELS[experiment["model_name"]]

    gt_coco = cocolib.COCO(experiment["dataset_file"])

    # Load and run prediction on the chosen image
    img_meta = gt_coco.imgs[image_id]
    img = io.imread(f"{dataset_base_dir}/{img_meta['file_name']}")
    prediction = model.predict(img, verbose=False)[0]

    gt_cat_by_name = {c["name"]: c["id"] for c in gt_coco.dataset["categories"]}

    # Build predicted annotations in COCO format for drawing
    pred_anns = []
    for box in prediction.boxes:
        cls_name = model.names[int(box.cls[0])]
        cat_id = gt_cat_by_name.get(cls_name)
        if cat_id is None:
            continue
        pred_anns.append(
            {
                "bbox": xywh_yolo_to_coco(box.xywh.tolist()[0]),
                "category_id": cat_id,
                "score": float(box.conf[0]),
            }
        )

    gt_anns = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=[image_id]))

    plot(
        img,
        gt_anns,
        pred_anns,
        gt_coco,
        title=f"[{exp_id}] image_id={image_id}  —  {img_meta['file_name']}",
        pred_label=experiment["model_name"],
    )


if __name__ == "__main__":
    exp_ids = list(params["experiments"].keys())
    print(f"Available experiments: {exp_ids}")
    exp_id = input("Experiment: ").strip()

    gt_coco = cocolib.COCO(params["experiments"][exp_id]["dataset_file"])
    print(f"Image IDs range: 0 - {max(gt_coco.imgs)}")

    while True:
        image_id = int(input("Image ID: ").strip())
        visualize(exp_id, image_id)
