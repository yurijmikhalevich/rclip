from concurrent.futures import ThreadPoolExecutor
import tempfile

from rclip.db import DB, NewImage


def _new_image(filepath: str, modified_at: float = 0.0) -> NewImage:
  return NewImage(filepath=filepath, modified_at=modified_at, size=1, vector=b"x", hash=None)


def test_get_image_vectors_by_dir_path_matches_windows_drive_root_prefix():
  with tempfile.TemporaryDirectory() as tmpdirname:
    database = DB(f"{tmpdirname}/db.sqlite3")
    try:
      database.upsert_image(_new_image(r"Y:\cat.jpg"))
      database.upsert_image(_new_image(r"Y:\nested\dog.jpg"))
      database.upsert_image(_new_image(r"Z:\other.jpg"))

      rows = list(database.get_image_vectors_by_dir_path("Y:\\"))

      assert [row["filepath"] for row in rows] == [r"Y:\cat.jpg", r"Y:\nested\dog.jpg"]
    finally:
      database.close()


def test_get_image_vectors_by_dir_path_matches_windows_subdir_with_trailing_separator():
  with tempfile.TemporaryDirectory() as tmpdirname:
    database = DB(f"{tmpdirname}/db.sqlite3")
    try:
      database.upsert_image(_new_image(r"Y:\photos\cat.jpg"))
      database.upsert_image(_new_image(r"Y:\photos\nested\dog.jpg"))
      database.upsert_image(_new_image(r"Y:\photos-archive\bird.jpg"))

      rows = list(database.get_image_vectors_by_dir_path("Y:\\photos\\"))

      assert [row["filepath"] for row in rows] == [r"Y:\photos\cat.jpg", r"Y:\photos\nested\dog.jpg"]
    finally:
      database.close()


def test_get_image_vectors_by_dir_path_matches_windows_subdir_without_trailing_separator():
  with tempfile.TemporaryDirectory() as tmpdirname:
    database = DB(f"{tmpdirname}/db.sqlite3")
    try:
      database.upsert_image(_new_image(r"Y:\photos\cat.jpg"))
      database.upsert_image(_new_image(r"Y:\photos\nested\dog.jpg"))
      database.upsert_image(_new_image(r"Y:\photos-archive\bird.jpg"))

      rows = list(database.get_image_vectors_by_dir_path(r"Y:\photos"))

      assert [row["filepath"] for row in rows] == [r"Y:\photos\cat.jpg", r"Y:\photos\nested\dog.jpg"]
    finally:
      database.close()


def test_get_dirpath_like_pattern_escapes_like_wildcards():
  with tempfile.TemporaryDirectory() as tmpdirname:
    database = DB(f"{tmpdirname}/db.sqlite3")
    try:
      database.upsert_image(_new_image(r"Y:\100% real\cat.jpg"))
      database.upsert_image(_new_image(r"Y:\1000 real\dog.jpg"))
      database.upsert_image(_new_image(r"Y:\100_ real\bird.jpg"))

      rows = list(database.get_image_vectors_by_dir_path(r"Y:\100% real"))

      assert [row["filepath"] for row in rows] == [r"Y:\100% real\cat.jpg"]
    finally:
      database.close()


def test_database_allows_configured_cross_thread_reads(tmp_path):
  database = DB(tmp_path / "db.sqlite3", allow_cross_thread=True)
  try:
    database.upsert_image(_new_image("/photos/cat.jpg"))

    with ThreadPoolExecutor(max_workers=1) as executor:
      rows = executor.submit(lambda: list(database.get_image_vectors_by_dir_path("/photos"))).result()

    assert [row["filepath"] for row in rows] == ["/photos/cat.jpg"]
  finally:
    database.close()


def test_lists_recent_image_filepaths_with_an_optional_limit(tmp_path):
  database = DB(tmp_path / "db.sqlite3")
  try:
    database.upsert_image(_new_image("/photos/old.jpg", modified_at=1))
    database.upsert_image(_new_image("/photos/new.jpg", modified_at=2))
    database.upsert_image(_new_image("/elsewhere/image.jpg", modified_at=3))

    assert database.get_image_filepaths_by_dir_path("/photos") == ["/photos/new.jpg", "/photos/old.jpg"]
    assert database.get_image_filepaths_by_dir_path("/photos", 1) == ["/photos/new.jpg"]
  finally:
    database.close()
