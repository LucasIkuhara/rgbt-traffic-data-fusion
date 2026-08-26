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


class Training(TypedDict):
    work_dir: str
    input_model: str
    output_model: str
    n_splits: int
    epochs: int
    imgsz: int
    batch: int
    freeze: int


class Params(TypedDict):
    experiments: ExperimentList
    training: Training


params: Params = load(open("src/params.toml", "rb"))

if __name__ == "__main__":
    print(params)
