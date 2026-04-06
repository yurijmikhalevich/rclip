import tempfile

from rclip.db import DB, NewImage


def _new_image(filepath: str) -> NewImage:
  return NewImage(filepath=filepath, modified_at=0.0, size=1, vector=b"x")


def test_get_image_vectors_by_dir_path_matches_windows_drive_root_prefix():
  with tempfile.TemporaryDirectory() as tmpdirname:
    database = DB(f"{tmpdirname}/db.sqlite3")
    database.upsert_image(_new_image(r"Y:\cat.jpg"))
    database.upsert_image(_new_image(r"Y:\nested\dog.jpg"))
    database.upsert_image(_new_image(r"Z:\other.jpg"))

    rows = list(database.get_image_vectors_by_dir_path("Y:\\"))

    assert [row["filepath"] for row in rows] == [r"Y:\cat.jpg", r"Y:\nested\dog.jpg"]
    database.close()


def test_get_image_vectors_by_dir_path_matches_windows_subdir_with_trailing_separator():
  with tempfile.TemporaryDirectory() as tmpdirname:
    database = DB(f"{tmpdirname}/db.sqlite3")
    database.upsert_image(_new_image(r"Y:\photos\cat.jpg"))
    database.upsert_image(_new_image(r"Y:\photos\nested\dog.jpg"))
    database.upsert_image(_new_image(r"Y:\photos-archive\bird.jpg"))

    rows = list(database.get_image_vectors_by_dir_path("Y:\\photos\\"))

    assert [row["filepath"] for row in rows] == [r"Y:\photos\cat.jpg", r"Y:\photos\nested\dog.jpg"]
    database.close()


def test_get_image_vectors_by_dir_path_matches_windows_subdir_without_trailing_separator():
  with tempfile.TemporaryDirectory() as tmpdirname:
    database = DB(f"{tmpdirname}/db.sqlite3")
    database.upsert_image(_new_image(r"Y:\photos\cat.jpg"))
    database.upsert_image(_new_image(r"Y:\photos\nested\dog.jpg"))
    database.upsert_image(_new_image(r"Y:\photos-archive\bird.jpg"))

    rows = list(database.get_image_vectors_by_dir_path(r"Y:\photos"))

    assert [row["filepath"] for row in rows] == [r"Y:\photos\cat.jpg", r"Y:\photos\nested\dog.jpg"]
    database.close()


def test_get_dirpath_like_pattern_escapes_like_wildcards():
  with tempfile.TemporaryDirectory() as tmpdirname:
    database = DB(f"{tmpdirname}/db.sqlite3")
    database.upsert_image(_new_image(r"Y:\100% real\cat.jpg"))
    database.upsert_image(_new_image(r"Y:\1000 real\dog.jpg"))
    database.upsert_image(_new_image(r"Y:\100_ real\bird.jpg"))

    rows = list(database.get_image_vectors_by_dir_path(r"Y:\100% real"))

    assert [row["filepath"] for row in rows] == [r"Y:\100% real\cat.jpg"]
    database.close()
