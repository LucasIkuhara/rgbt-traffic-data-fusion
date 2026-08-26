from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def get_thermal_detector() -> YOLO:
    """
    Instances an YOLO detector with thermal imaging weights and parameters.
    Returns:
      The model instance
    """
    model = YOLO("theramal_yolo2.pt")

    # set model parameters
    # Reference: https://huggingface.co/foduucom/thermal-image-object-detection
    model.overrides["conf"] = 0.05  # NMS confidence threshold
    model.overrides["iou"] = 0.45  # NMS IoU threshold
    model.overrides["agnostic_nms"] = False  # NMS class-agnostic
    model.overrides["max_det"] = 1000  # maximum number of detections per image

    return model


def get_rgb_detector() -> YOLO:
    """
    Instances an YOLO detector with rgb imaging weights and parameters.
    Returns:
      The model instance
    """
    # Reference: https://huggingface.co/Ultralytics/YOLOv8
    model = YOLO("yolov8s.pt")
    model.overrides["conf"] = 0.3  # NMS confidence threshold

    return model


MODELS = {
    "rbg_yolo_v8_s": get_rgb_detector(),
    "thermal_yolo_v8_s": get_thermal_detector(),
}

if __name__ == "__main__":
    th = get_rgb_detector()
    results = th.predict(source="images.jpeg", conf=0.25, iou=0.45)
    print(results)
    # Process or view results natively
    for r in results:
        r.show()  # Opens the annotated image
