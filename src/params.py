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
    fusion_iou_thr: float
    soft_nms_sigma: float
    soft_nms_thresh: float


class PreProcessing(TypedDict):
    input_rgb_ann:      str        # raw RGB annotation JSON
    input_thermal_ann:  str        # raw thermal annotation JSON
    output_rgb_ann:     str        # cleaned RGB annotation JSON (output)
    output_thermal_ann: str        # cleaned thermal annotation JSON (output)
    filter_categories:  list[str]  # remove annotations with these category names


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
    experiments:    ExperimentList
    preprocessing:  PreProcessing
    inference:      Inference
    training:       Training


params: Params = load(open("src/params.toml", "rb"))

if __name__ == "__main__":
    print(params)
