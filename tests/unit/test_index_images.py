from pathlib import Path
import os
from threading import Event
from unittest.mock import call, Mock
import tempfile

import numpy as np
import PIL
from PIL import Image
import pytest

from rclip import main as main_module
from rclip.db import DB, NewImage
from rclip.main import ImageMeta, RClip
from rclip.utils import helpers


def _make_rclip(model, database, exclude_dirs=None):
  return RClip(model, database, indexing_batch_size=8, exclude_dirs=exclude_dirs)


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
  model = Mock()
  model.compute_features_for_queries.return_value = np.zeros(512, dtype=np.float32)
  rclip = _make_rclip(model, database)

  with pytest.raises(InterruptedError):
    rclip.search("cat", ".", cancel_event=cancel_event)


def test_search_stops_between_query_groups_when_cancelled() -> None:
  cancel_event = Event()
  model = Mock()
  calls: list[list[str]] = []

  def compute(queries: list[str]) -> np.ndarray:
    calls.append(queries)
    cancel_event.set()
    return np.zeros(512, dtype=np.float32)

  model.compute_features_for_queries.side_effect = compute
  database = Mock()
  rclip = _make_rclip(model, database)

  with pytest.raises(InterruptedError):
    rclip.search("cat", ".", cancel_event=cancel_event)
  assert calls == [["cat"]]
  database.get_image_vectors_by_dir_path.assert_not_called()


def test_search_stops_when_cancelled_during_final_sort(monkeypatch) -> None:
  cancel_event = Event()
  model = Mock()
  model.compute_features_for_queries.return_value = np.zeros(2, dtype=np.float32)
  database = Mock()
  database.get_image_vectors_by_dir_path.return_value = [
    {"filepath": "a.jpg", "vector": np.zeros(2, dtype=np.float32).tobytes()}
  ]

  def sort_results(results, *, key):
    def cancelling_key(result):
      cancel_event.set()
      return key(result)

    return sorted(results, key=cancelling_key)

  monkeypatch.setattr(main_module, "sorted", sort_results, raising=False)
  rclip = _make_rclip(model, database)

  with pytest.raises(InterruptedError):
    rclip.search("cat", ".", top_k=None, cancel_event=cancel_event)


@pytest.mark.parametrize("top_k", [1, None])
def test_search_continues_after_an_entirely_excluded_batch(monkeypatch, tmp_path: Path, top_k) -> None:
  monkeypatch.setattr(RClip, "SEARCH_BATCH_SIZE", 2)
  model = Mock()
  model.compute_features_for_queries.side_effect = [
    np.array([1, 0], dtype=np.float32),
    np.zeros(2, dtype=np.float32),
  ]
  database = Mock()
  database.get_image_vectors_by_dir_path.return_value = [
    {"filepath": str(tmp_path / filepath), "vector": np.array([score, 0], dtype=np.float32).tobytes()}
    for filepath, score in (
      ("private/a.jpg", 4),
      ("query.jpg", 3),
      ("b.jpg", 1),
      ("a.jpg", 2),
    )
  ]
  rclip = _make_rclip(model, database, ["private"])

  assert rclip.search(str(tmp_path / "query.jpg"), str(tmp_path), top_k=top_k) == [
    RClip.SearchResult(str(tmp_path / "a.jpg"), 2),
    RClip.SearchResult(str(tmp_path / "b.jpg"), 1),
  ][:top_k]


def test_search_keeps_only_global_top_results_across_vector_batches(monkeypatch) -> None:
  model = Mock()
  model.compute_features_for_queries.side_effect = [
    np.array([1, 0], dtype=np.float32),
    np.array([0, 0], dtype=np.float32),
  ]
  database = Mock()
  database.get_image_vectors_by_dir_path.return_value = [
    {"filepath": filepath, "vector": np.array(vector, dtype=np.float32).tobytes()}
    for filepath, vector in (
      ("z.jpg", [0.8, 0]),
      ("b.jpg", [0.9, 0]),
      ("c.jpg", [0.2, 0]),
      ("a.jpg", [0.9, 0]),
      ("d.jpg", [0.7, 0]),
    )
  ]
  rclip = _make_rclip(model, database)
  monkeypatch.setattr(RClip, "SEARCH_BATCH_SIZE", 2)
  batch_sizes: list[int] = []
  stack = main_module.np.stack

  def record_stack(features):
    batch_sizes.append(len(features))
    return stack(features)

  monkeypatch.setattr(main_module.np, "stack", record_stack)

  results = rclip.search("cat", ".", 3, ["bright"], ["dark"])

  assert [result.filepath for result in results] == ["a.jpg", "b.jpg", "z.jpg"]
  np.testing.assert_allclose([result.score for result in results], [0.9, 0.9, 0.8])
  assert model.compute_features_for_queries.call_args_list == [call(["cat", "bright"]), call(["dark"])]
  assert batch_sizes == [2, 2, 1]


def test_search_can_return_every_ranked_result(monkeypatch) -> None:
  monkeypatch.setattr(RClip, "SEARCH_BATCH_SIZE", 2)
  database = Mock()
  database.get_image_vectors_by_dir_path.return_value = [
    {"filepath": f"{index}.jpg", "vector": np.array([index, 0], dtype=np.float32).tobytes()}
    for index in range(12)
  ]
  model = Mock()
  model.compute_features_for_queries.side_effect = [
    np.array([1, 0], dtype=np.float32),
    np.array([0, 0], dtype=np.float32),
  ]
  rclip = _make_rclip(model, database)

  assert rclip.search("cat", ".", top_k=None) == [
    RClip.SearchResult(f"{index}.jpg", float(index)) for index in reversed(range(12))
  ]


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


def test_list_images_applies_exclusions_before_the_limit(tmp_path: Path) -> None:
  database = DB(tmp_path / "db.sqlite3")
  private = tmp_path / "private" / "new.jpg"
  first = tmp_path / "first.jpg"
  second = tmp_path / "second.jpg"
  database.upsert_image(NewImage(filepath=str(private), modified_at=3, size=1, vector=b"x", hash=None))
  database.upsert_image(NewImage(filepath=str(first), modified_at=2, size=1, vector=b"x", hash=None))
  database.upsert_image(NewImage(filepath=str(second), modified_at=1, size=1, vector=b"x", hash=None))
  rclip = _make_rclip(Mock(), database, ["private"])

  try:
    first_page = rclip.list_images(str(tmp_path), 1)
    assert first_page.filepaths == [str(first)]
    assert first_page.next_cursor == RClip.ImageCursor(2, str(first))
    assert rclip.list_images(str(tmp_path), 1, after=first_page.next_cursor) == RClip.ImagePage([str(second)], None)
  finally:
    rclip.close()
    database.close()


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


@pytest.mark.parametrize(
  "excluded_name,include_hidden,exclude_dirs",
  [(".worktrees", False, None), (".git", True, None), ("private", True, ["private"])],
)
def test_exclusions_preserve_cache_and_apply_to_search_and_browse(
  tmp_path: Path, excluded_name: str, include_hidden: bool, exclude_dirs: list[str] | None
):
  # The same excluded name above the search root must not affect its descendants.
  root = tmp_path / excluded_name / "checkout"
  visible = root / "nested" / "photo.jpg"
  excluded = root / excluded_name / "photo.jpg"
  dotfile = root / ".photo.jpg"
  for path in (visible, excluded, dotfile):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(path)

  database = DB(tmp_path / "db.sqlite3")
  model = Mock()
  model.compute_preprocessed_image_features.side_effect = lambda images, **kwargs: np.ones(
    (len(images), 512), dtype=np.float32
  )
  model.compute_features_for_queries.return_value = np.zeros(512, dtype=np.float32)
  enabled = RClip(model, database, 8, ["unused"], include_hidden=True)
  restricted = RClip(model, database, 8, exclude_dirs, include_hidden=include_hidden)

  def results(app: RClip, expected: set[str]):
    assert {row.filepath for row in app.search("cat", str(root), 10)} == expected
    assert set(app.list_images(str(root), 10).filepaths) == expected
    assert len(app.search("cat", str(root), 1)) == 1
    page = app.list_images(str(root), 1)
    browsed = list(page.filepaths)
    assert len(browsed) == 1
    while page.next_cursor is not None:
      page = app.list_images(str(root), 1, after=page.next_cursor)
      browsed.extend(page.filepaths)
    assert set(browsed) == expected
    assert len(browsed) == len(expected)

  all_paths = {str(visible), str(excluded), str(dotfile)}
  visible_paths = {str(visible), str(dotfile)} if include_hidden else {str(visible)}
  try:
    enabled.ensure_index(str(root))
    model.compute_preprocessed_image_features.reset_mock()
    results(enabled, all_paths)
    results(restricted, visible_paths)  # --no-indexing must still filter cached results.
    restricted.ensure_index(str(root))
    results(restricted, visible_paths)
    results(enabled, all_paths)
    enabled.ensure_index(str(root))
    results(enabled, all_paths)
    model.compute_preprocessed_image_features.assert_not_called()

    # A parent scan excludes this entire checkout; searching inside it still works.
    restricted.ensure_index(str(tmp_path))
    restricted.ensure_index(str(root))
    results(restricted, visible_paths)
    results(enabled, all_paths)

    excluded.unlink()
    restricted.ensure_index(str(root))
    results(enabled, all_paths)  # Out-of-scope deletions remain unverified.
    enabled.ensure_index(str(root))
    results(enabled, all_paths - {str(excluded)})
  finally:
    enabled.close()
    restricted.close()
    database.close()


def test_index_restores_unchanged_deleted_image(tmp_path: Path):
  photo = tmp_path / "photo.jpg"
  photo.touch()
  stat = photo.stat()
  vector = np.ones(512, dtype=np.float32).tobytes()
  database = DB(tmp_path / "db.sqlite3")
  database.upsert_image(
    NewImage(filepath=str(photo), modified_at=stat.st_mtime, size=stat.st_size, vector=vector, hash=None)
  )
  model = Mock()
  app = _make_rclip(model, database)
  try:
    photo.unlink()
    app.ensure_index(str(tmp_path))
    assert app.list_images(str(tmp_path), 10).filepaths == []
    photo.touch()
    os.utime(photo, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    app.ensure_index(str(tmp_path))
    assert app.list_images(str(tmp_path), 10).filepaths == [str(photo)]
    restored = database.get_image(filepath=str(photo))
    assert restored is not None
    assert restored["deleted"] is None
    assert restored["vector"] == vector
    model.compute_preprocessed_image_features.assert_not_called()
  finally:
    app.close()
    database.close()
