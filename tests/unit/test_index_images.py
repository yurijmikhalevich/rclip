from unittest.mock import Mock

import numpy as np
import PIL
from PIL import Image

from rclip.main import ImageMeta, RClip
from rclip.utils import helpers


def _make_rclip(model, database):
  return RClip(model, database, indexing_batch_size=8, exclude_dirs=None)


def _fail_on_b(path: str) -> Image.Image:
  if path == "b.jpg":
    raise PIL.UnidentifiedImageError()
  return Image.new("RGB", (1, 1))


def test_load_images_preserves_order_and_skips_failures(monkeypatch):
  monkeypatch.setattr(helpers, "_ensure_image_loading_configured", lambda: None)
  monkeypatch.setattr(helpers, "read_image", _fail_on_b)

  meta_a = ImageMeta(modified_at=1.0, size=100)
  meta_b = ImageMeta(modified_at=2.0, size=200)
  meta_c = ImageMeta(modified_at=3.0, size=300)

  rclip = _make_rclip(Mock(), Mock())
  try:
    loaded = list(rclip._load_images([("a.jpg", meta_a), ("b.jpg", meta_b), ("c.jpg", meta_c)]))
  finally:
    rclip.close()

  # b.jpg failed to load and is dropped; the survivors keep their order and their own metas/images
  assert [(path, meta) for path, meta, _image in loaded] == [("a.jpg", meta_a), ("c.jpg", meta_c)]
  # the loader threads preprocess the images, so it yields ready-to-encode CLIP tensors
  assert all(isinstance(image, np.ndarray) and image.shape == (3, 256, 256) for _path, _meta, image in loaded)


def test_index_images_keeps_meta_aligned_when_an_image_fails_to_load(monkeypatch):
  # the middle image fails to load, shrinking the surviving paths/features
  monkeypatch.setattr(helpers, "_ensure_image_loading_configured", lambda: None)
  monkeypatch.setattr(helpers, "read_image", _fail_on_b)

  meta_a = ImageMeta(modified_at=1.0, size=100)
  meta_b = ImageMeta(modified_at=2.0, size=200)
  meta_c = ImageMeta(modified_at=3.0, size=300)

  model = Mock()
  # one feature vector per surviving image (a and c)
  model.compute_image_features_from_preprocessed.return_value = [
    np.zeros(4, dtype=np.float32),
    np.ones(4, dtype=np.float32),
  ]
  database = Mock()

  rclip = _make_rclip(model, database)
  try:
    rclip._index_images([("a.jpg", meta_a), ("b.jpg", meta_b), ("c.jpg", meta_c)])
  finally:
    rclip.close()

  upserted = {
    (call.args[0]["filepath"], call.args[0]["modified_at"], call.args[0]["size"])
    for call in database.upsert_image.call_args_list
  }
  # each surviving image must keep its own meta; a desync would attribute meta_b to c.jpg
  assert upserted == {
    ("a.jpg", meta_a["modified_at"], meta_a["size"]),
    ("c.jpg", meta_c["modified_at"], meta_c["size"]),
  }
