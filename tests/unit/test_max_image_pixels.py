import pytest
from PIL import Image

from rclip.utils import helpers


@pytest.fixture(autouse=True)
def _restore_cap():
  # the cap is process-global (it is mirrored onto PIL), so reset it after each test
  original = helpers.get_max_image_pixels()
  yield
  helpers.configure_max_image_pixels(original, workers=1)


def test_default_cap_never_drops_below_the_floor():
  # even with many workers, ordinary large photos must still index
  assert helpers.compute_default_max_image_pixels(64) >= helpers.MIN_MAX_IMAGE_PIXELS


def test_default_cap_falls_back_to_the_floor_without_memory_info(monkeypatch):
  monkeypatch.setattr(helpers, "_get_total_memory_bytes", lambda: None)
  assert helpers.compute_default_max_image_pixels(16) == helpers.MIN_MAX_IMAGE_PIXELS


def test_explicit_cap_rejects_oversized_image_before_decoding(tmp_path):
  helpers.configure_max_image_pixels(100, workers=1)
  path = str(tmp_path / "big.png")
  Image.new("RGB", (50, 50)).save(path)  # 2500 px > 100

  with pytest.raises(helpers.ImageTooLargeError) as exc:
    helpers.read_image(path)
  assert exc.value.pixels == 2500
  assert exc.value.limit == 100


def test_cap_allows_images_within_the_limit(tmp_path):
  helpers.configure_max_image_pixels(100, workers=1)
  path = str(tmp_path / "small.png")
  Image.new("RGB", (5, 5)).save(path)  # 25 px < 100
  assert helpers.read_image(path).size == (5, 5)


def test_disabled_cap_opens_any_image(tmp_path):
  helpers.configure_max_image_pixels(None, workers=1)
  assert helpers.get_max_image_pixels() is None
  path = str(tmp_path / "big.png")
  Image.new("RGB", (50, 50)).save(path)
  assert helpers.read_image(path).size == (50, 50)


def test_arg_type_parses_disable_keywords_and_megapixels():
  assert helpers.max_image_megapixels_arg_type("none") is None
  assert helpers.max_image_megapixels_arg_type("0") is None
  # the value is given in megapixels and converted to an internal pixel count
  assert helpers.max_image_megapixels_arg_type("500") == 500_000_000
  assert helpers.max_image_megapixels_arg_type("89.5") == 89_500_000
  with pytest.raises(Exception):
    helpers.max_image_megapixels_arg_type("-5")
