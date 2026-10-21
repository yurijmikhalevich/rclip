from concurrent.futures import ThreadPoolExecutor
import io
import shutil
import sqlite3
import sys
import tempfile
from types import SimpleNamespace

from rclip.db import DB, NewImage


def _new_image(filepath: str, modified_at: float = 0.0) -> NewImage:
  return NewImage(filepath=filepath, modified_at=modified_at, size=1, vector=b"x", hash=None)


def _small_page_database(path, monkeypatch, filepath: str = "/photos/cat.jpg") -> None:
  with monkeypatch.context() as patch:
    patch.setattr(DB, "PAGE_SIZE", 4096)
    database = DB(path)
    database.upsert_image(_new_image(filepath))
    database.close()


def _page_size(database: DB) -> int:
  return database._con.execute("PRAGMA page_size").fetchone()[0]


def test_new_database_uses_large_pages(tmp_path):
  database = DB(tmp_path / "db.sqlite3")
  try:
    assert _page_size(database) == DB.PAGE_SIZE
  finally:
    database.close()


def test_rebuilds_a_small_page_database_with_large_pages(tmp_path, monkeypatch, capsys):
  _small_page_database(tmp_path / "db.sqlite3", monkeypatch)

  database = DB(tmp_path / "db.sqlite3")
  try:
    assert _page_size(database) == DB.PAGE_SIZE
    assert database.get_image_states(["/photos/cat.jpg"]).keys() == {"/photos/cat.jpg"}
    assert "rebuilding the index" in capsys.readouterr().err
  finally:
    database.close()


def test_switches_a_wal_database_to_a_rollback_journal_and_rebuilds_it_with_large_pages(tmp_path, monkeypatch):
  _small_page_database(tmp_path / "db.sqlite3", monkeypatch)
  connection = sqlite3.connect(tmp_path / "db.sqlite3")
  connection.execute("PRAGMA journal_mode=WAL")
  connection.close()

  database = DB(tmp_path / "db.sqlite3")
  try:
    assert _page_size(database) == DB.PAGE_SIZE
    assert database._con.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
  finally:
    database.close()


def test_skips_the_rebuild_while_another_process_keeps_the_database_in_wal_mode(tmp_path, monkeypatch, capsys):
  _small_page_database(tmp_path / "db.sqlite3", monkeypatch)
  other = sqlite3.connect(tmp_path / "db.sqlite3")
  other.execute("PRAGMA journal_mode=WAL")
  other.execute("SELECT * FROM images").fetchall()
  connect = sqlite3.connect
  monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: connect(*args, **kwargs, timeout=0))

  database = DB(tmp_path / "db.sqlite3")
  try:
    assert database._con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert _page_size(database) == 4096
    assert "rebuilding the index" not in capsys.readouterr().err
  finally:
    database.close()
    other.close()


class _Terminal(io.StringIO):
  def isatty(self) -> bool:
    return True


def test_keeps_small_pages_when_the_rebuild_of_a_large_index_is_declined(tmp_path, monkeypatch):
  _small_page_database(tmp_path / "db.sqlite3", monkeypatch)
  monkeypatch.setattr(DB, "REBUILD_CONFIRMATION_IMAGES", 1)
  monkeypatch.setattr(sys, "stdin", _Terminal("n\n"))
  stderr = _Terminal()
  monkeypatch.setattr(sys, "stderr", stderr)

  database = DB(tmp_path / "db.sqlite3")
  try:
    assert _page_size(database) == 4096
    assert "Rebuild it now? [Y/n]" in stderr.getvalue()
  finally:
    database.close()


def test_keeps_small_pages_without_enough_free_disk_space(tmp_path, monkeypatch):
  _small_page_database(tmp_path / "db.sqlite3", monkeypatch)
  monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=0))

  database = DB(tmp_path / "db.sqlite3")
  try:
    assert _page_size(database) == 4096
  finally:
    database.close()


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


def test_get_image_vectors_by_dir_path_matches_wildcard_characters_literally():
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


def test_get_image_states_returns_the_stored_states(tmp_path):
  database = DB(tmp_path / "db.sqlite3")
  try:
    database.upsert_image(_new_image("/photos/cat.jpg", modified_at=2.0))
    database.upsert_image(_new_image("/photos/dog.jpg", modified_at=3.0))
    database.mark_images_as_deleted(["/photos/dog.jpg"])

    states = database.get_image_states(["/photos/cat.jpg", "/photos/dog.jpg", "/photos/missing.jpg"])

    assert states.keys() == {"/photos/cat.jpg", "/photos/dog.jpg"}
    assert states["/photos/cat.jpg"]["modified_at"] == 2.0
    assert states["/photos/cat.jpg"]["size"] == 1
    assert states["/photos/cat.jpg"]["deleted"] is None
    assert states["/photos/dog.jpg"]["deleted"]
  finally:
    database.close()


def test_get_image_vectors_by_dir_path_filters_with_the_filepath_index(tmp_path):
  database = DB(tmp_path / "db.sqlite3")
  try:
    database.upsert_image(_new_image(str(tmp_path / "photos" / "cat.jpg")))
    statements: list[str] = []
    database._con.set_trace_callback(statements.append)
    list(database.get_image_vectors_by_dir_path(str(tmp_path / "photos")))
    database._con.set_trace_callback(None)

    query = next(statement for statement in statements if statement.startswith("SELECT filepath, vector"))
    plan = " ".join(row["detail"] for row in database._con.execute(f"EXPLAIN QUERY PLAN {query}"))

    assert "existing_images" in plan
    assert "filepath>?" in plan
    assert "filepath<?" in plan
  finally:
    database.close()


def test_database_reads_the_index_through_a_memory_map(tmp_path):
  database = DB(tmp_path / "db.sqlite3")
  try:
    assert database._con.execute("PRAGMA mmap_size").fetchone()[0] > 0
  finally:
    database.close()


def test_database_allows_configured_cross_thread_reads(tmp_path):
  database = DB(tmp_path / "db.sqlite3", allow_cross_thread=True)
  try:
    photos = tmp_path / "photos"
    cat = str(photos / "cat.jpg")
    database.upsert_image(_new_image(cat))

    with ThreadPoolExecutor(max_workers=1) as executor:
      rows = executor.submit(lambda: list(database.get_image_vectors_by_dir_path(str(photos)))).result()

    assert [row["filepath"] for row in rows] == [cat]
  finally:
    database.close()


def test_lists_recent_image_filepaths(tmp_path):
  database = DB(tmp_path / "db.sqlite3")
  try:
    photos = tmp_path / "photos"
    old = str(photos / "old.jpg")
    new_a = str(photos / "new-a.jpg")
    new_b = str(photos / "new-b.jpg")
    database.upsert_image(_new_image(old, modified_at=1))
    database.upsert_image(_new_image(new_b, modified_at=2))
    database.upsert_image(_new_image(new_a, modified_at=2))
    database.upsert_image(_new_image(str(tmp_path / "elsewhere" / "image.jpg"), modified_at=3))

    assert [row["filepath"] for row in database.get_image_filepaths_by_dir_path(str(photos))] == [
      new_a,
      new_b,
      old,
    ]
    assert [row["filepath"] for row in database.get_image_filepaths_by_dir_path(str(photos), (2, new_a))] == [
      new_b,
      old,
    ]
  finally:
    database.close()


def test_lists_recent_image_filepaths_without_sorting(tmp_path):
  database = DB(tmp_path / "db.sqlite3")
  try:
    photos = tmp_path / "photos"
    database.upsert_image(_new_image(str(photos / "cat.jpg")))
    statements: list[str] = []
    database._con.set_trace_callback(statements.append)
    list(database.get_image_filepaths_by_dir_path(str(photos), (2, str(photos / "dog.jpg"))))
    database._con.set_trace_callback(None)

    query = next(statement for statement in statements if statement.startswith("SELECT filepath, modified_at"))
    plan = " ".join(row["detail"] for row in database._con.execute(f"EXPLAIN QUERY PLAN {query}"))

    assert "recent_images" in plan
    assert "TEMP B-TREE" not in plan
  finally:
    database.close()


def test_mark_images_as_deleted_hides_only_the_given_images(tmp_path):
  database = DB(tmp_path / "db.sqlite3")
  try:
    photos = tmp_path / "photos"
    cat, dog = str(photos / "cat.jpg"), str(photos / "dog.jpg")
    database.upsert_image(_new_image(cat))
    database.upsert_image(_new_image(dog))

    database.mark_images_as_deleted([cat])

    assert [row["filepath"] for row in database.get_filepaths_by_dir_path(str(photos))] == [dog]
  finally:
    database.close()
