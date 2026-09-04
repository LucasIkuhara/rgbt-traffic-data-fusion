from typing import TypedDict
from tomllib import load


class Experiment(TypedDict):
    name: str
    dataset_base_dir: str
    dataset_file: str
    model_name: str
    output_path: str


class ExperimentList(TypedDict):
    rgb: Experiment
    thermal: Experiment


class Inference(TypedDict):
    conf_rgb: float
    iou_rgb: float
    conf_thermal: float
    iou_thermal: float
    wbf_iou_thr: float
    gt_bbox_source: str   # "rle" | "json"


class Training(TypedDict):
    work_dir: str
    input_model_rgb: str
    output_model_rgb: str
    input_model_thermal: str
    output_model_thermal: str
    n_splits: int
    epochs: int
    imgsz: int
    batch: int
    freeze: int


class Params(TypedDict):
    experiments: ExperimentList
    inference: Inference
    training: Training


params: Params = load(open("src/params.toml", "rb"))

if __name__ == "__main__":
    print(params)
