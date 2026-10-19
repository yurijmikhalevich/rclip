import ctypes
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from rclip.utils import native_heif, preprocess
from rclip.utils._windows_heif import _is_file_error, _scaled_size

requires_wic = pytest.mark.skipif(sys.platform != "win32", reason="Windows Imaging Component is required")

FIXTURE = Path(__file__).parent / "fixtures" / "large.heic"
FIXTURE_SIZE = (1600, 1200)


@pytest.fixture(scope="module")
def windows_heif() -> Any:
  module = importlib.import_module("rclip.utils._windows_heif")
  try:
    module.decode_path(FIXTURE, None)
  except native_heif.NativeHeifCodecUnavailableError as error:
    pytest.skip(f"the operating system HEIF codec is unavailable: {error}")
  return module


@pytest.mark.parametrize("size", [(4000, 3000), (6000, 4000), (1600, 1200), (1000, 1000), (5120, 2880), (1000, 2000)])
def test_scaled_size_covers_min_side(size: tuple[int, int]):
  width, height = _scaled_size(*size, preprocess.IMAGE_SIZE)

  assert min(width, height) >= preprocess.IMAGE_SIZE
  assert width <= size[0]
  assert height <= size[1]


@pytest.mark.parametrize("size", [(384, 256), (256, 256), (100, 50), (256, 1000), (2000, 256)])
def test_scaled_size_leaves_images_that_already_fit_untouched(size: tuple[int, int]):
  assert _scaled_size(*size, preprocess.IMAGE_SIZE) == size


@pytest.mark.parametrize(
  ("hresult", "expected"),
  [
    (0x80070002, True),  # ERROR_FILE_NOT_FOUND
    (0x80070005, True),  # ERROR_ACCESS_DENIED
    (0x80070020, True),  # ERROR_SHARING_VIOLATION
    (0x88982F50, False),  # WINCODEC_ERR_COMPONENTNOTFOUND
    (0x88982F07, False),  # WINCODEC_ERR_UNKNOWNIMAGEFORMAT
    (0x00000000, False),  # S_OK
  ],
)
def test_is_file_error_tells_file_failures_from_missing_codecs(hresult: int, expected: bool):
  assert _is_file_error(ctypes.c_int32(hresult).value) is expected


@requires_wic
def test_decode_path_returns_the_full_frame_by_default(windows_heif: Any):
  assert windows_heif.decode_path(FIXTURE, None).size == FIXTURE_SIZE


@requires_wic
def test_decode_path_downscales_large_heic_to_min_side(windows_heif: Any):
  image = windows_heif.decode_path(FIXTURE, None, preprocess.IMAGE_SIZE)

  assert image.size == _scaled_size(*FIXTURE_SIZE, preprocess.IMAGE_SIZE)
  assert min(image.size) >= preprocess.IMAGE_SIZE


@requires_wic
def test_decode_path_enforces_the_pixel_limit_on_the_source_size(windows_heif: Any):
  with pytest.raises(native_heif.NativeHeifTooLargeError) as raised:
    windows_heif.decode_path(FIXTURE, 1_000_000, preprocess.IMAGE_SIZE)

  assert raised.value.pixels == FIXTURE_SIZE[0] * FIXTURE_SIZE[1]
