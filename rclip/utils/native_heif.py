"""Platform-native HEIF decoding.

The platform modules deliberately use only APIs supplied by the operating system.  In particular,
this package must not grow a fallback to libheif/libde265: an unavailable system codec means HEIF is
unavailable on that machine.
"""

import errno
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image


class NativeHeifError(Exception):
  """Base class for failures reported by a platform HEIF decoder."""


class NativeHeifCodecUnavailableError(NativeHeifError):
  """The operating system has no codec capable of decoding this HEIF image."""


class NativeHeifDecodeError(NativeHeifError):
  """The operating system rejected or failed to decode the image."""


class NativeHeifTooLargeError(NativeHeifError):
  """The image dimensions exceed rclip's configured pixel limit."""

  def __init__(self, pixels: int, limit: int):
    self.pixels = pixels
    self.limit = limit
    super().__init__(f"image has {pixels} pixels, which exceeds the limit of {limit} pixels")


def decode_path(path: str | os.PathLike[str], max_pixels: int | None) -> Image.Image:
  """Decode a HEIF image using the current operating system's imaging API."""
  if not Path(path).exists():
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
  if sys.platform == "darwin":
    from rclip.utils._macos_heif import decode_path as platform_decode_path

    return platform_decode_path(path, max_pixels)
  if sys.platform == "win32":
    from rclip.utils._windows_heif import decode_path as platform_decode_path

    return platform_decode_path(path, max_pixels)
  raise NativeHeifCodecUnavailableError("native HEIF decoding is available only on macOS and Windows")


def decode_bytes(data: bytes, max_pixels: int | None) -> Image.Image:
  """Decode in-memory HEIF data through a temporary file accepted by both native APIs."""
  temporary_file = tempfile.NamedTemporaryFile(suffix=".heic", delete=False)
  temporary_path = Path(temporary_file.name)
  try:
    with temporary_file:
      temporary_file.write(data)
    return decode_path(temporary_path, max_pixels)
  finally:
    temporary_path.unlink(missing_ok=True)
