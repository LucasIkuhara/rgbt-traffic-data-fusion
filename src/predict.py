import json
from typing import TypedDict
from src.models import get_rgb_detector, get_thermal_detector
from pycocotools import coco
import skimage.io as io

DATASET_PATH = "aau-rainsnow/"


class CocoDetection(TypedDict):
    image_id: str
    category_id: int
    bbox: list
    score: float


# YOLO xywh (centre-based) to COCO xywh (top-left-based)
def xywh_yolo_to_coco(v: list[float]) -> list[float]:
    return [v[0] - (v[2] / 2), v[1] - (v[3] / 2), v[2], v[3]]


experiments = [
    (DATASET_PATH + "aauRainSnow-rgb.json", get_rgb_detector(), "rgb_predictions.json")
]

for dataset_path, model, output_path in experiments:

    rainSnowRgbGt = coco.COCO(dataset_path)
    annotations = []
    gt_cat_by_name = {c["name"]: c["id"] for c in rainSnowRgbGt.dataset["categories"]}

    for img_idx in rainSnowRgbGt.imgs:
        print(f"Running...\t{100*img_idx/len(rainSnowRgbGt.imgs):.2f} %")
        img_meta = rainSnowRgbGt.imgs[img_idx]
        img_data = io.imread(DATASET_PATH + img_meta["file_name"])

        prediction = model.predict(img_data, verbose=False)[0]

        for box in prediction.boxes:
            # Map 0-based model class index to COCO category id via model names;
            # skip classes that have no counterpart in the GT (e.g. frisbee, sports ball)
            cls_name = model.names[int(box.cls[0])]
            cat_id = gt_cat_by_name.get(cls_name)
            if cat_id is None:
                continue
            coco_detection_ann = CocoDetection(
                image_id=img_meta["id"],
                bbox=xywh_yolo_to_coco(box.xywh.tolist()[0]),
                category_id=cat_id,
                score=float(box.conf[0]),
            )
            annotations.append(coco_detection_ann)

    with open(output_path, "w") as f:
        print(f"Writing output to {f.name}...")
        f.write(json.dumps(annotations))
        print("Done.")
