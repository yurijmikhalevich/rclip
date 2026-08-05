from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from rclip.utils import helpers, native_heif


def test_native_heif_is_unavailable_on_linux(monkeypatch):
  monkeypatch.setattr(native_heif.sys, "platform", "linux")

  with pytest.raises(native_heif.NativeHeifCodecUnavailableError):
    native_heif.decode_path(__file__, None)


def test_missing_native_heif_preserves_file_not_found_error():
  with pytest.raises(FileNotFoundError) as raised:
    native_heif.decode_path("missing.heic", None)

  assert raised.value.filename == "missing.heic"


def test_decode_bytes_removes_its_temporary_file(monkeypatch):
  temporary_path: Path | None = None

  def fake_decode_path(path, max_pixels):
    nonlocal temporary_path
    temporary_path = Path(path)
    assert temporary_path.read_bytes() == b"heif data"
    assert max_pixels == 123
    return Image.new("RGB", (1, 1))

  monkeypatch.setattr(native_heif, "decode_path", fake_decode_path)
  image = native_heif.decode_bytes(b"heif data", 123)

  assert image.size == (1, 1)
  assert temporary_path is not None
  assert not temporary_path.exists()


def test_read_image_routes_heic_to_native_decoder(monkeypatch):
  expected = Image.new("RGB", (2, 3))
  monkeypatch.setattr(helpers, "IS_MACOS", True)
  monkeypatch.setattr(helpers, "IS_WINDOWS", False)
  monkeypatch.setattr(helpers, "_native_heif_path", lambda path: expected)

  assert helpers.read_image("photo.heic") is expected


def test_read_image_does_not_use_native_decoder_on_linux(monkeypatch):
  monkeypatch.setattr(helpers, "IS_MACOS", False)
  monkeypatch.setattr(helpers, "IS_WINDOWS", False)
  monkeypatch.setattr(helpers, "_native_heif_path", lambda path: pytest.fail("native decoder was called"))
  monkeypatch.setattr(helpers.Image, "open", lambda path: (_ for _ in ()).throw(UnidentifiedImageError()))

  with pytest.raises(UnidentifiedImageError):
    helpers.read_image("photo.heic")


def test_native_pixel_limit_is_reported_as_rclip_error(monkeypatch):
  def too_large(path, max_pixels):
    raise native_heif.NativeHeifTooLargeError(200, 100)

  monkeypatch.setattr(native_heif, "decode_path", too_large)

  with pytest.raises(helpers.ImageTooLargeError) as raised:
    helpers._native_heif_path("photo.heic")

  assert raised.value.path == "photo.heic"
  assert raised.value.pixels == 200
  assert raised.value.limit == 100
