import tempfile
import sqlite3

import pytest

from rclip.db import DB, NewImage


def _new_image(filepath: str) -> NewImage:
  return NewImage(filepath=filepath, modified_at=0.0, size=1, vector=b"x")


def _create_v2_database(path: str, *, with_images: bool = True) -> None:
  con = sqlite3.connect(path)
  try:
    con.execute(
      """
      CREATE TABLE images (
        id INTEGER PRIMARY KEY,
        deleted BOOLEAN,
        filepath TEXT NOT NULL UNIQUE,
        modified_at DATETIME NOT NULL,
        size INTEGER NOT NULL,
        vector BLOB NOT NULL,
        indexing BOOLEAN
      )
      """
    )
    con.execute("CREATE UNIQUE INDEX existing_images ON images(filepath) WHERE deleted IS NULL")
    con.execute("CREATE TABLE db_version (version INTEGER)")
    con.execute("INSERT INTO db_version(version) VALUES (2)")
    if with_images:
      con.execute(
        "INSERT INTO images(filepath, modified_at, size, vector, deleted, indexing) VALUES (?, ?, ?, ?, ?, ?)",
        ("/tmp/cat.jpg", 0.0, 1, b"x", None, None),
      )
    con.commit()
  finally:
    con.close()


def _read_image_count_and_version(path: str) -> tuple[int, int]:
  con = sqlite3.connect(path)
  try:
    count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    version = con.execute("SELECT version FROM db_version").fetchone()[0]
    return count, version
  finally:
    con.close()


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


def test_v3_migration_requires_confirmation_before_deleting_cached_vectors(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
  with tempfile.TemporaryDirectory() as tmpdirname:
    db_path = f"{tmpdirname}/db.sqlite3"
    _create_v2_database(db_path)

    monkeypatch.setattr("builtins.input", lambda: "n")

    with pytest.raises(SystemExit) as ex:
      DB(db_path)

    assert ex.value.code == 0
    err = capsys.readouterr().err
    assert "rclip v3 is incompatible with the existing vector cache" in err
    assert "Delete the vector cache and continue? [y/N]: " in err
    assert "Aborting without changing the vector cache." in err

    count, version = _read_image_count_and_version(db_path)

    assert count == 1
    assert version == 2


def test_v3_migration_deletes_cached_vectors_after_confirmation(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
  with tempfile.TemporaryDirectory() as tmpdirname:
    db_path = f"{tmpdirname}/db.sqlite3"
    _create_v2_database(db_path)

    monkeypatch.setattr("builtins.input", lambda: "y")

    database = DB(db_path)
    try:
      err = capsys.readouterr().err
      assert "rclip v3 is incompatible with the existing vector cache" in err
      assert "Delete the vector cache and continue? [y/N]: " in err
      assert "Aborting without changing the vector cache." not in err

      count, version = _read_image_count_and_version(db_path)
    finally:
      database.close()

    assert count == 0
    assert version == 3
