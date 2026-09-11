from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_PATH = Path("aau-rainsnow")
IMG_W, IMG_H = 640.0, 480.0


# ---------------------------------------------------------------------------
# Calibration I/O
# ---------------------------------------------------------------------------

def load_calib(file_name: str) -> dict:
    """Load the calib.yml for the clip that contains *file_name*.

    *file_name* is relative to the dataset root, e.g.
        'Egensevej/Egensevej-1/cam2-00055.png'

    Returns a dict with keys:
        homCam1Cam2, homCam2Cam1,
        cam1CamMat, cam2CamMat,
        cam1DistCoeff, cam2DistCoeff
    """
    parts = Path(file_name).parts        # (scene, clip, imgfile)
    scene, clip = parts[0], parts[1]
    calib_path = DATASET_PATH / scene / f"{clip}-calib.yml"
    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
    return {
        "homCam1Cam2":   fs.getNode("homCam1Cam2").mat(),
        "homCam2Cam1":   fs.getNode("homCam2Cam1").mat(),
        "cam1CamMat":    fs.getNode("cam1CamMat").mat(),
        "cam2CamMat":    fs.getNode("cam2CamMat").mat(),
        "cam1DistCoeff": fs.getNode("cam1DistCoeff").mat(),
        "cam2DistCoeff": fs.getNode("cam2DistCoeff").mat(),
    }


# ---------------------------------------------------------------------------
# Point projection
# ---------------------------------------------------------------------------

def register_points_rgb_to_thermal(
    points: np.ndarray,   # (N, 2) float32
    calib: dict,
) -> np.ndarray:
    """Project points from RGB (cam1) into thermal (cam2) space.

    Steps mirror aauRainSnowUtility.registerRgbPointsToThermal:
      1. Undistort with cam1 intrinsics
      2. Apply homCam1Cam2
      3. Re-distort with cam2 intrinsics
    """
    pts = points.astype(np.float64).reshape(-1, 1, 2)

    # 1. Undistort
    undist = cv2.undistortPoints(
        pts,
        calib["cam1CamMat"],
        calib["cam1DistCoeff"],
        P=calib["cam1CamMat"],
    )

    # 2. Homography
    proj = cv2.perspectiveTransform(undist, calib["homCam1Cam2"])  # (N,1,2)

    # 3. Re-distort: normalise by cam2 intrinsics, apply projectPoints with
    #    zero rotation/translation so it only applies distortion.
    K2 = calib["cam2CamMat"]
    D2 = calib["cam2DistCoeff"]
    normalised = []
    for pt in proj[:, 0, :]:
        nx = (pt[0] - K2[0, 2]) / K2[0, 0]
        ny = (pt[1] - K2[1, 2]) / K2[1, 1]
        normalised.append([nx, ny, 1.0])

    distorted, _ = cv2.projectPoints(
        np.array(normalised, dtype=np.float32).reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        K2,
        D2,
    )
    return distorted.reshape(-1, 2)  # (N, 2)


def register_points_thermal_to_rgb(
    points: np.ndarray,   # (N, 2) float32
    calib: dict,
) -> np.ndarray:
    """Project points from thermal (cam2) into RGB (cam1) space.

    Inverse of register_points_rgb_to_thermal:
      1. Undistort with cam2 intrinsics
      2. Apply homCam2Cam1
      3. Re-distort with cam1 intrinsics
    """
    pts = points.astype(np.float64).reshape(-1, 1, 2)

    # 1. Undistort
    undist = cv2.undistortPoints(
        pts,
        calib["cam2CamMat"],
        calib["cam2DistCoeff"],
        P=calib["cam2CamMat"],
    )

    # 2. Homography
    proj = cv2.perspectiveTransform(undist, calib["homCam2Cam1"])  # (N,1,2)

    # 3. Re-distort with cam1 intrinsics
    K1 = calib["cam1CamMat"]
    D1 = calib["cam1DistCoeff"]
    normalised = []
    for pt in proj[:, 0, :]:
        nx = (pt[0] - K1[0, 2]) / K1[0, 0]
        ny = (pt[1] - K1[1, 2]) / K1[1, 1]
        normalised.append([nx, ny, 1.0])

    distorted, _ = cv2.projectPoints(
        np.array(normalised, dtype=np.float32).reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        K1,
        D1,
    )
    return distorted.reshape(-1, 2)  # (N, 2)


# ---------------------------------------------------------------------------
# Bounding-box projection
# ---------------------------------------------------------------------------

def transform_bbox_rgb_to_thermal(
    bbox: list[float],  # COCO [x, y, w, h] in RGB pixel space
    calib: dict,
) -> list[float]:
    """Project a COCO bbox from RGB space to thermal space.

    Projects all four corners, takes the axis-aligned bounding box of the
    result, and clamps to the image bounds.
    """
    x, y, w, h = bbox
    corners = np.array([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h],
    ], dtype=np.float32)

    projected = register_points_rgb_to_thermal(corners, calib)

    x1 = float(np.clip(projected[:, 0].min(), 0, IMG_W))
    y1 = float(np.clip(projected[:, 1].min(), 0, IMG_H))
    x2 = float(np.clip(projected[:, 0].max(), 0, IMG_W))
    y2 = float(np.clip(projected[:, 1].max(), 0, IMG_H))

    return [x1, y1, x2 - x1, y2 - y1]


def transform_bbox_thermal_to_rgb(
    bbox: list[float],  # COCO [x, y, w, h] in thermal pixel space
    calib: dict,
) -> list[float]:
    """Project a COCO bbox from thermal space to RGB space.

    Projects all four corners, takes the axis-aligned bounding box of the
    result, and clamps to the image bounds.
    """
    x, y, w, h = bbox
    corners = np.array([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h],
    ], dtype=np.float32)

    projected = register_points_thermal_to_rgb(corners, calib)

    x1 = float(np.clip(projected[:, 0].min(), 0, IMG_W))
    y1 = float(np.clip(projected[:, 1].min(), 0, IMG_H))
    x2 = float(np.clip(projected[:, 0].max(), 0, IMG_W))
    y2 = float(np.clip(projected[:, 1].max(), 0, IMG_H))

    return [x1, y1, x2 - x1, y2 - y1]
