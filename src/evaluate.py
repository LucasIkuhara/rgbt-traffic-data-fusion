from pycocotools import coco
from pycocotools.cocoeval import COCOeval
from src.params import Experiment, params

for exp_id in params["experiments"]:

    experiment: Experiment = params["experiments"][exp_id]
    predictions_path = experiment["output_path"]
    ground_truth_annotations = experiment["dataset_file"]

    coco_dataset = coco.COCO(ground_truth_annotations)
    res = coco_dataset.loadRes(predictions_path)
    ev = COCOeval(coco_dataset, res, "bbox")

    print(f"\n\n == Evaluating: ", experiment["name"], "==")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
