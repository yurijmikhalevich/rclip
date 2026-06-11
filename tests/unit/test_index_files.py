from unittest.mock import Mock

import numpy as np
import PIL
from PIL import Image

from rclip.main import ImageMeta, RClip
from rclip.utils import helpers


def test_index_files_keeps_meta_aligned_when_an_image_fails_to_load(monkeypatch):
  # the middle image in the batch fails to load, shrinking the surviving paths/features
  def fake_read_image(path: str) -> Image.Image:
    if path == "b.jpg":
      raise PIL.UnidentifiedImageError()
    return Image.new("RGB", (1, 1))

  monkeypatch.setattr(helpers, "read_image", fake_read_image)

  meta_a = ImageMeta(modified_at=1.0, size=100)
  meta_b = ImageMeta(modified_at=2.0, size=200)
  meta_c = ImageMeta(modified_at=3.0, size=300)

  model = Mock()
  # one feature vector per surviving image (a and c)
  model.compute_image_features.return_value = [np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)]
  database = Mock()

  rclip = RClip(model, database, indexing_batch_size=8, exclude_dirs=None)
  try:
    rclip._index_files(["a.jpg", "b.jpg", "c.jpg"], [meta_a, meta_b, meta_c])
  finally:
    rclip.close()

  upserted = {
    (call.args[0]["filepath"], call.args[0]["modified_at"], call.args[0]["size"])
    for call in database.upsert_image.call_args_list
  }
  # the surviving images must keep their own metas; before the fix c.jpg got meta_b
  assert upserted == {
    ("a.jpg", meta_a["modified_at"], meta_a["size"]),
    ("c.jpg", meta_c["modified_at"], meta_c["size"]),
  }
