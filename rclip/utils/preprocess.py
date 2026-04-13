"""
CLIP image preprocessing in pure PIL + numpy.
Replicates the open_clip preprocessing pipeline for ViT-B-32-256/datacomp_s34b_b86k:
  Resize(256, bicubic, antialias) → CenterCrop(256) → RGB float32 [0,1] → Normalize(CLIP mean/std)
"""

import numpy as np
import numpy.typing as npt
from PIL import Image

IMAGE_SIZE = 256
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess(image: Image.Image) -> npt.NDArray[np.float32]:
  """Preprocess one PIL image into a float32 array of shape [3, 256, 256]."""
  image = image.convert("RGB")

  # Resize shortest side to 256. PIL.Image.resize expects (width, height).
  width, height = image.size
  if width < height:
    resized_width = IMAGE_SIZE
    resized_height = int(height * IMAGE_SIZE / width)
  else:
    resized_height = IMAGE_SIZE
    resized_width = int(width * IMAGE_SIZE / height)
  image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

  # Center crop 256x256
  width, height = image.size
  left = round((width - IMAGE_SIZE) / 2.0)
  top = round((height - IMAGE_SIZE) / 2.0)
  image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

  # To float32 [0, 1], shape [H, W, 3]
  pixel_array = np.array(image, dtype=np.float32) / 255.0

  # Normalize
  pixel_array = (pixel_array - MEAN) / STD

  # HWC → CHW
  pixel_array = pixel_array.transpose(2, 0, 1)
  return pixel_array
