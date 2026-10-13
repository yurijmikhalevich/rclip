from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rclip.utils import preprocess


def _save_image(path: Path, size: tuple[int, int], mode: str = "RGB") -> None:
  width, height = size
  gradient = (np.add.outer(np.linspace(0, 255, height), np.linspace(0, 255, width)) / 2).astype(np.uint8)
  pixels = gradient if mode == "L" else np.stack([gradient, np.fliplr(gradient), gradient // 2 + 64], axis=-1)
  Image.fromarray(pixels, mode).save(path, quality=85)


@pytest.mark.parametrize("size", [(4096, 3072), (3072, 4096), (1024, 768), (256, 256), (200, 100), (100, 200)])
def test_preprocess_returns_the_clip_input_for_any_aspect_ratio(tmp_path: Path, size: tuple[int, int]):
  path = tmp_path / "photo.jpg"
  _save_image(path, size)

  result = preprocess.preprocess(Image.open(path))

  assert result.shape == (3, preprocess.IMAGE_SIZE, preprocess.IMAGE_SIZE)
  assert result.dtype == np.float32


def test_preprocess_decodes_large_jpegs_at_a_reduced_scale(tmp_path: Path):
  path = tmp_path / "large.jpg"
  _save_image(path, (4096, 3072))
  image = Image.open(path)

  preprocess.preprocess(image)

  # libjpeg's largest DCT reduction that keeps both sides >= 256: 4096x3072 -> 1/8 -> 512x384
  assert image.size == (512, 384)


def test_preprocess_drafts_grayscale_jpegs_too(tmp_path: Path):
  path = tmp_path / "gray.jpg"
  _save_image(path, (2048, 1536), mode="L")
  image = Image.open(path)

  result = preprocess.preprocess(image)

  assert result.shape == (3, preprocess.IMAGE_SIZE, preprocess.IMAGE_SIZE)
  assert image.size == (512, 384)


def test_preprocess_leaves_small_images_and_other_formats_undrafted(tmp_path: Path):
  small_jpeg = tmp_path / "small.jpg"
  _save_image(small_jpeg, (400, 300))
  small = Image.open(small_jpeg)

  large_png = tmp_path / "large.png"
  _save_image(large_png, (4096, 3072))
  png = Image.open(large_png)

  preprocess.preprocess(small)
  preprocess.preprocess(png)

  assert small.size == (400, 300)
  assert png.size == (4096, 3072)


def test_preprocess_draft_stays_close_to_the_full_resolution_result(tmp_path: Path):
  path = tmp_path / "large.jpg"
  _save_image(path, (4096, 3072))

  drafted = preprocess.preprocess(Image.open(path))
  # copy() decodes at full resolution, so the draft is a no-op for it
  full_resolution = preprocess.preprocess(Image.open(path).copy())

  np.testing.assert_allclose(drafted, full_resolution, atol=0.2)
