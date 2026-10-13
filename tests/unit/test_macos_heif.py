import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from rclip.utils import native_heif, preprocess

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS Image I/O is required")

FIXTURE = Path(__file__).parent / "fixtures" / "large.heic"
FIXTURE_SIZE = (1600, 1200)


@pytest.fixture(scope="module")
def macos_heif() -> Any:
  return importlib.import_module("rclip.utils._macos_heif")


def test_decode_path_downscales_large_heic(macos_heif: Any):
  image = macos_heif.decode_path(FIXTURE, None)

  width, height = image.size
  assert min(width, height) >= preprocess.IMAGE_SIZE
  assert width * height < FIXTURE_SIZE[0] * FIXTURE_SIZE[1]


def test_thumbnail_decode_matches_the_full_frame_decode(macos_heif: Any, monkeypatch: pytest.MonkeyPatch):
  thumbnail = macos_heif.decode_path(FIXTURE, None)
  with monkeypatch.context() as patch:
    patch.setattr(macos_heif, "_create_thumbnail", lambda *_: 0)
    full = macos_heif.decode_path(FIXTURE, None)

  assert full.size == FIXTURE_SIZE
  expected = full.resize(thumbnail.size, Image.Resampling.BICUBIC)
  assert np.abs(np.asarray(thumbnail, np.float32) - np.asarray(expected, np.float32)).mean() < 10


def test_decode_path_enforces_the_pixel_limit_on_the_source_size(macos_heif: Any):
  with pytest.raises(native_heif.NativeHeifTooLargeError) as raised:
    macos_heif.decode_path(FIXTURE, 1_000_000)

  assert raised.value.pixels == FIXTURE_SIZE[0] * FIXTURE_SIZE[1]
