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


class Params(TypedDict):
    experiments: ExperimentList


params: Params = load(open("src/params.toml", "rb"))

if __name__ == "__main__":
    print(params)
