from functools import lru_cache
from pathlib import Path

import numpy as np
import skimage.io as io


def get_mask_path(dataset_path: Path, file_name: str, thermal: bool) -> Path:
    """Derive the clip mask path from an image file_name.

    E.g. 'Egensevej/Egensevej-1/cam2-00055.png', thermal=True
      → <dataset_path>/Egensevej/Egensevej-1-mask-thermal.png
    """
    parts = Path(file_name).parts  # (scene, clip, filename)
    scene, clip = parts[0], parts[1]
    suffix = "-mask-thermal.png" if thermal else "-mask.png"
    return dataset_path / scene / f"{clip}{suffix}"


@lru_cache(maxsize=64)
def _load_mask_array(mask_path: Path) -> np.ndarray:
    """Load a clip mask and return a boolean (H, W) array (True = valid pixel).
    Cached so each mask file is read from disk only once per process.
    """
    img = io.imread(str(mask_path))
    if img.ndim == 2:
        channel = img
    elif img.shape[2] == 4:
        channel = img[:, :, 3]  # alpha channel
    else:
        channel = img[:, :, 0]
    return channel > 0


def apply_mask(img: np.ndarray, dataset_path: Path, file_name: str, thermal: bool) -> np.ndarray:
    """Zero out pixels in the ignore regions of an image.

    The mask is loaded once and cached. If no mask file exists for the clip
    the image is returned unmodified. The returned image has the same shape
    and dtype as the input.
    """
    mask_path = get_mask_path(dataset_path, file_name, thermal=thermal)
    if not mask_path.exists():
        return img
    mask = _load_mask_array(mask_path)         # (H, W) bool
    return (img * mask[:, :, np.newaxis]).astype(img.dtype)
