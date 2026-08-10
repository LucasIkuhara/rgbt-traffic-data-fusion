from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def get_thermal_detector() -> YOLO:
    """
    Instances an YOLO detector with thermal imaging weights and parameters.
    Returns:
      The model instance
    """

    model_path = hf_hub_download(
        repo_id="foduucom/thermal-image-object-detection", filename="best.pt"
    )

    model = YOLO(model_path)

    # set model parameters
    # params from: https://huggingface.co/foduucom/thermal-image-object-detection
    model.overrides["conf"] = 0.25  # NMS confidence threshold
    model.overrides["iou"] = 0.45  # NMS IoU threshold
    model.overrides["agnostic_nms"] = False  # NMS class-agnostic
    model.overrides["max_det"] = 1000  # maximum number of detections per image

    return model


# Run prediction with configurations set directly in the predict call
th = get_thermal_detector()
results = th.predict(source="test.png", conf=0.25, iou=0.45)
print(results)
# Process or view results natively
for r in results:
    r.show()  # Opens the annotated image
