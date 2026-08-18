from typing import TypedDict

from src.models import get_rgb_detector, get_thermal_detector

from pycocotools import coco
import numpy as np
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt


class CocoDetection(TypedDict):
    image_id: str
    category_id: str
    bbox: list
    score: float


DATASET_PATH = "aau-rainsnow/"
RGB_COCO = "aauRainSnow-rgb.json"
rgbAnnFile = DATASET_PATH + "aauRainSnow-rgb.json"

rgb_model = get_rgb_detector()
rainSnowRgbGt = coco.COCO(rgbAnnFile)
rainSnowRgbGt.loadRes

for img_idx in rainSnowRgbGt.imgs:
    img_meta = rainSnowRgbGt.imgs[img_idx]
    img_data = io.imread(DATASET_PATH + img_meta["file_name"])

    # Use the first prediction
    # as it only generates one per frame wih N boxes
    prediction = rgb_model.predict(img_data)[0]

    # plt.show()

    # print(img_meta)

    for box in prediction.boxes:
        coco_detection_ann = CocoDetection(
            image_id=img_meta["id"],
            bbox=box.xywh.tolist(),
            category_id=box.cls,
            score=box.conf,
        )
        print(coco_detection_ann, end="\n\n")
    input()
