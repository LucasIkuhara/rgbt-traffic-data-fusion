# Downloads and extracts the dataset (AAU RainSnow)
dataset:
    curl -L -o aau-rainsnow.zip \
    https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow && \
    unzip aau-rainsnow.zip -d aau-rainsnow && rm aau-rainsnow.zip

# Downloads base YOLOv8-x weights
get-base-weights:
    poetry run python -c 'from ultralytics import YOLO; YOLO("yolov8x.pt")'

# Run preprocessing scripts
pre-process:
    poetry run python -m src.pre_processing

# Fine-tunes models 
train:
    poetry run python -m src.train

# Evaluate models and fused results based on validation sets
eval:
    poetry run python -m src.evaluate_fused

# Reproduce entire experiment
run: dataset get-base-weights pre-process train eval

visualize:
    poetry run python -m src.visualize