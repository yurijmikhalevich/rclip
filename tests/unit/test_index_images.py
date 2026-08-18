from pathlib import Path
from threading import Event
from unittest.mock import Mock
import tempfile

import numpy as np
import PIL
from PIL import Image
import pytest

from rclip.db import DB, NewImage
from rclip.main import ImageMeta, RClip
from rclip.utils import helpers


def _make_rclip(model, database):
  return RClip(model, database, indexing_batch_size=8, exclude_dirs=None)


def _fail_on_b(path: str) -> Image.Image:
  if path == "b.jpg":
    raise PIL.UnidentifiedImageError()
  return Image.new("RGB", (1, 1))


def _too_large_on_b(path: str) -> Image.Image:
  if path == "b.jpg":
    raise helpers.ImageTooLargeError(path, 200_000_000, 100_000_000)
  return Image.new("RGB", (1, 1))


def test_search_stops_loading_vectors_when_cancelled() -> None:
  cancel_event = Event()
  database = Mock()

  def rows(_directory: str):
    yield {"filepath": "a.jpg", "vector": np.zeros(512, dtype=np.float32).tobytes()}
    cancel_event.set()
    yield {"filepath": "b.jpg", "vector": np.zeros(512, dtype=np.float32).tobytes()}

  database.get_image_vectors_by_dir_path.side_effect = rows
  rclip = _make_rclip(Mock(), database)

  with pytest.raises(InterruptedError):
    rclip.search("cat", ".", cancel_event=cancel_event)


def test_load_images_preserves_order_and_skips_failures(monkeypatch):
  monkeypatch.setattr(helpers, "_ensure_image_loading_configured", lambda: None)
  monkeypatch.setattr(helpers, "read_image", _fail_on_b)
  monkeypatch.setattr(helpers, "compute_file_hash", lambda path: "dummy_hash")

  meta_a = ImageMeta(modified_at=1.0, size=100)
  meta_b = ImageMeta(modified_at=2.0, size=200)
  meta_c = ImageMeta(modified_at=3.0, size=300)

  rclip = _make_rclip(Mock(), Mock())
  try:
    loaded = list(rclip._load_images([("a.jpg", meta_a), ("b.jpg", meta_b), ("c.jpg", meta_c)]))
  finally:
    rclip.close()

  # b.jpg failed to load and is dropped; the survivors keep their order and their own metas/images
  assert [(path, meta) for path, meta, _hash, _image in loaded] == [("a.jpg", meta_a), ("c.jpg", meta_c)]
  # the loader threads preprocess the images, so it yields ready-to-encode CLIP tensors
  assert all(isinstance(image, np.ndarray) and image.shape == (3, 256, 256) for _path, _meta, _hash, image in loaded)


def test_load_images_skips_images_that_are_too_large(monkeypatch, capsys):
  monkeypatch.setattr(helpers, "_ensure_image_loading_configured", lambda: None)
  monkeypatch.setattr(helpers, "read_image", _too_large_on_b)
  monkeypatch.setattr(helpers, "compute_file_hash", lambda path: "dummy_hash")

  meta_a = ImageMeta(modified_at=1.0, size=100)
  meta_b = ImageMeta(modified_at=2.0, size=200)
  meta_c = ImageMeta(modified_at=3.0, size=300)

  rclip = _make_rclip(Mock(), Mock())
  try:
    loaded = list(rclip._load_images([("a.jpg", meta_a), ("b.jpg", meta_b), ("c.jpg", meta_c)]))
  finally:
    rclip.close()

  # the too-large image is dropped, the rest survive in order
  assert [path for path, _meta, _hash, _image in loaded] == ["a.jpg", "c.jpg"]
  # the user gets a friendly, actionable message naming the file and the limit
  err = capsys.readouterr().err
  assert "skipping b.jpg" in err
  assert "too large" in err
  assert "--max-image-megapixels" in err


def test_index_images_keeps_meta_aligned_when_an_image_fails_to_load(monkeypatch):
  # the middle image fails to load, shrinking the surviving paths/features
  monkeypatch.setattr(helpers, "_ensure_image_loading_configured", lambda: None)
  monkeypatch.setattr(helpers, "read_image", _fail_on_b)
  monkeypatch.setattr(helpers, "compute_file_hash", lambda path: "dummy_hash")

  meta_a = ImageMeta(modified_at=1.0, size=100)
  meta_b = ImageMeta(modified_at=2.0, size=200)
  meta_c = ImageMeta(modified_at=3.0, size=300)

  model = Mock()
  # one feature vector per surviving image (a and c)
  model.compute_preprocessed_image_features.return_value = [
    np.zeros(4, dtype=np.float32),
    np.ones(4, dtype=np.float32),
  ]
  database = Mock()
  # Configure mock to return empty list for hash lookups (no existing images)
  database.get_images_by_hash.return_value = []

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


def test_rename_reuses_vector_without_recomputing(monkeypatch):
  monkeypatch.setattr(helpers, "_ensure_image_loading_configured", lambda: None)
  monkeypatch.setattr(helpers, "read_image", lambda path: Image.new("RGB", (1, 1)))

  with tempfile.TemporaryDirectory() as tmp_dir:
    database = DB(Path(tmp_dir) / "test.db")

    # Pre-populate with an "old" image (simulates prior indexing run)
    old_hash = "abc123"
    old_vector = b"\x01\x02\x03\x04"
    database.upsert_image(
      NewImage(
        filepath="/old/path/cat.jpg",
        modified_at=1.0,
        size=100,
        vector=old_vector,
        hash=old_hash,
      )
    )
    database.commit()

    # The "renamed" file produces the same hash and has the same size as the old one
    monkeypatch.setattr(helpers, "compute_file_hash", lambda path: old_hash)

    model = Mock()
    rclip = _make_rclip(model, database)
    try:
      rclip._index_images([("/new/path/renamed_cat.jpg", ImageMeta(modified_at=2.0, size=100))])
    finally:
      rclip.close()

    # Core assertion: model was never called (vector was reused)
    model.compute_preprocessed_image_features.assert_not_called()

    # New path has the old vector
    new_record = database.get_image(filepath="/new/path/renamed_cat.jpg")
    assert new_record is not None
    assert new_record["vector"] == old_vector
    assert new_record["hash"] == old_hash

    database.close()


def test_same_hash_different_size_reindexes(monkeypatch):
  monkeypatch.setattr(helpers, "_ensure_image_loading_configured", lambda: None)
  monkeypatch.setattr(helpers, "read_image", lambda path: Image.new("RGB", (1, 1)))

  with tempfile.TemporaryDirectory() as tmp_dir:
    database = DB(Path(tmp_dir) / "test.db")

    # Pre-populate with an "old" image (simulates prior indexing run)
    old_hash = "abc123"
    old_vector = b"\x01\x02\x03\x04"
    database.upsert_image(
      NewImage(
        filepath="/old/path/cat.jpg",
        modified_at=1.0,
        size=100,
        vector=old_vector,
        hash=old_hash,
      )
    )
    database.commit()

    # A file with the same hash but different size is NOT a rename -> recompute the vector
    monkeypatch.setattr(helpers, "compute_file_hash", lambda path: old_hash)

    new_vector = np.zeros(4, dtype=np.float32)
    model = Mock()
    model.compute_preprocessed_image_features.return_value = [new_vector]
    rclip = _make_rclip(model, database)
    try:
      rclip._index_images([("/new/path/edited_cat.jpg", ImageMeta(modified_at=2.0, size=200))])
    finally:
      rclip.close()

    # Model was called because size differs, so the vector was recomputed
    model.compute_preprocessed_image_features.assert_called_once()

    # New path has the recomputed vector, not the old one
    new_record = database.get_image(filepath="/new/path/edited_cat.jpg")
    assert new_record is not None
    assert new_record["vector"] == new_vector.tobytes()
    assert new_record["vector"] != old_vector
    assert new_record["hash"] == old_hash

    database.close()
