"""
CLIP image preprocessing in pure PIL + numpy.
Replicates the open_clip preprocessing pipeline for ViT-B-32-256/datacomp_s34b_b86k:
  Resize(256, bicubic, antialias) → CenterCrop(256) → RGB float32 [0,1] → Normalize(CLIP mean/std)
"""

import numpy as np
import numpy.typing as npt
from PIL import Image

_IMAGE_SIZE = 256
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess(image: Image.Image) -> npt.NDArray[np.float32]:
  """Preprocess one PIL image into a float32 array of shape [3, 256, 256]."""
  image = image.convert("RGB")

  # Resize shortest side to 256. PIL.Image.resize expects (width, height).
  w, h = image.size
  if w < h:
    new_w = _IMAGE_SIZE
    new_h = int(h * _IMAGE_SIZE / w)
  else:
    new_h = _IMAGE_SIZE
    new_w = int(w * _IMAGE_SIZE / h)
  image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

  # Center crop 256x256
  w, h = image.size
  left = round((w - _IMAGE_SIZE) / 2.0)
  top = round((h - _IMAGE_SIZE) / 2.0)
  image = image.crop((left, top, left + _IMAGE_SIZE, top + _IMAGE_SIZE))

  # To float32 [0, 1], shape [H, W, 3]
  arr = np.array(image, dtype=np.float32) / 255.0

  # Normalize
  arr = (arr - _MEAN) / _STD

  # HWC → CHW
  arr = arr.transpose(2, 0, 1)
  return arr
