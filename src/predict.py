import json
from typing import TypedDict
from src.models import get_rgb_detector, get_thermal_detector
from pycocotools import coco
import skimage.io as io

DATASET_PATH = "aau-rainsnow/"


class CocoDetection(TypedDict):
    image_id: str
    category_id: str
    bbox: list
    score: float


experiments = [
    (DATASET_PATH + "aauRainSnow-rgb.json", get_rgb_detector(), "rgb_predictions.json")
]

for dataset_path, model, output_path in experiments:

    rainSnowRgbGt = coco.COCO(dataset_path)
    annotations = []

    for img_idx in rainSnowRgbGt.imgs:
        print(f"Running...\t{100*img_idx/len(rainSnowRgbGt.imgs):.2f} %")
        img_meta = rainSnowRgbGt.imgs[img_idx]
        img_data = io.imread(DATASET_PATH + img_meta["file_name"])

        # Use the first prediction
        # as it only generates one per frame wih N boxes
        prediction = model.predict(img_data)[0]

        for box in prediction.boxes:
            coco_detection_ann = CocoDetection(
                image_id=img_meta["id"],
                bbox=box.xywh.tolist()[0],
                category_id=float(box.cls[0]),
                score=float(box.conf[0]),
            )
            print(coco_detection_ann)
            annotations.append(coco_detection_ann)

        if img_idx > 130:
            break

    with open(output_path, "w") as f:
        print(f"Writing output to {f.name}...")
        f.write(json.dumps(annotations))
        print("Done.")
