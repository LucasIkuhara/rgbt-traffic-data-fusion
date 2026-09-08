# RGBT Traffic Data Fusion

Reproducing multi-modal object detection for traffic scenes by fusing RGB and thermal (LWIR) detections from YOLOv8 models. Detections are projected into a common thermal camera space using per-clip homography calibration and then combined with four ensemble methods: **WBF**, **NMS**, **Soft-NMS**, and **NMW**.

## Dependencies

| Tool | Purpose | Install |
|------|---------|---------|
| [Python ≥ 3.12](https://www.python.org/downloads/) | Runtime | — |
| [Poetry](https://python-poetry.org/docs/#installation) | Dependency & virtual-environment management | `pipx install poetry` |
| [just](https://just.systems/man/en/) | Task runner (like `make` but simpler) | `brew install just` |
| [Kaggle CLI](https://github.com/Kaggle/kaggle-api) | Dataset download | `pipx install kaggle` |

## Setup

```bash
# 1. Install Python dependencies into an isolated virtual environment
poetry install

# 2. (First time only) Authenticate the Kaggle CLI
#    Place your kaggle.json token at ~/.kaggle/kaggle.json
#    See https://github.com/Kaggle/kaggle-api#api-credentials
```

## Reproducing the experiment

All steps are orchestrated through `just`. Run each command from the repository root.

### Step-by-step

```bash
# Download and extract the AAU RainSnow dataset (~3 GB)
just dataset

# Download the base YOLOv8-x weights
just get-base-weights

# Clean annotations and generate k-fold splits
just pre-process

# Fine-tune RGB and thermal YOLOv8 models (5-fold cross-validation, 50 epochs each)
just train

# Run evaluation: thermal-only, RGB->thermal, and all four fusion methods
just eval
```

Results are written to `results.xlsx` with two sheets:

- **Overview** — mAP@50, mAP@[0.5:0.95], Recall, and F1 per configuration (mean ± std across folds).
- **Per-Class** — the same metrics broken down by object class, including instance counts.

### All-in-one

```bash
just run
```

Runs `dataset -> get-base-weights -> pre-process -> train -> eval` in sequence.

### Visualisation

```bash
just visualize
```

Renders sample detections for qualitative inspection.

## Available `just` commands

| Command | Description |
|---------|-------------|
| `just dataset` | Download and unzip the AAU RainSnow dataset from Kaggle |
| `just get-base-weights` | Pull pre-trained YOLOv8-x weights via Ultralytics |
| `just pre-process` | Clean COCO annotations and prepare k-fold splits |
| `just train` | Fine-tune RGB and thermal models with 5-fold CV |
| `just eval` | Evaluate all configurations; export `results.xlsx` |
| `just run` | End-to-end: dataset -> weights -> train -> eval |
| `just visualize` | Visualise predictions on sample images |


## Dataset

[AAU RainSnow Traffic Surveillance Dataset](https://www.kaggle.com/datasets/aalborguniversity/aau-rainsnow) — paired RGB and thermal video sequences of traffic intersections under varied weather conditions, annotated in COCO format.
