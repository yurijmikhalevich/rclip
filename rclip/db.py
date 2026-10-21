import os.path
import pathlib
import shutil
import sqlite3
import sys
from typing import Iterable, Optional, TypedDict, Union, cast


class ImageOmittable(TypedDict, total=False):
  deleted: bool


class NewImage(ImageOmittable):
  filepath: str
  modified_at: float
  size: int
  vector: bytes
  hash: Optional[str]


class Image(NewImage):
  id: int


class ImageState(TypedDict):
  deleted: Optional[bool]
  modified_at: float
  size: int


class DB:
  VERSION = 4
  MMAP_SIZE = 2**31
  # Each image takes about 2.2 KB, so a 4 KB page holds only one; a 16 KB page holds seven, which makes the index
  # about 40% smaller and faster to read.
  PAGE_SIZE = 16384
  # Rebuilding takes about half a minute per million images on an SSD, so larger indexes ask before it
  REBUILD_CONFIRMATION_IMAGES = 100_000

  def __init__(
    self,
    filename: Union[str, pathlib.Path],
    allow_vector_cache_reset: bool = True,
    allow_cross_thread: bool = False,
  ):
    self._con = sqlite3.connect(filename, check_same_thread=not allow_cross_thread)
    self._con.row_factory = sqlite3.Row
    self._filename = filename
    # applies to a new database right away, and to an existing one once it's rebuilt
    self._con.execute(f"PRAGMA page_size={self.PAGE_SIZE}")
    # A search reads every vector in the index, and reading them through a memory map skips copying each page
    # into SQLite's page cache. SQLite caps the map at its compile-time maximum, which is 2 GB by default.
    self._con.execute(f"PRAGMA mmap_size={self.MMAP_SIZE}")
    self.ensure_tables()
    self.ensure_version(allow_vector_cache_reset=allow_vector_cache_reset)
    self.ensure_page_size()

  def close(self):
    self._con.commit()
    self._con.close()

  def ensure_tables(self):
    self._con.execute("""
      CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        deleted BOOLEAN,
        filepath TEXT NOT NULL UNIQUE,
        modified_at DATETIME NOT NULL,
        size INTEGER NOT NULL,
        vector BLOB NOT NULL
      )
    """)
    # Query for images
    self._con.execute("CREATE UNIQUE INDEX IF NOT EXISTS existing_images ON images(filepath) WHERE deleted IS NULL")
    self._con.execute(
      "CREATE INDEX IF NOT EXISTS recent_images ON images(modified_at DESC, filepath) WHERE deleted IS NULL"
    )
    self._con.execute("CREATE TABLE IF NOT EXISTS db_version (version INTEGER)")
    self._con.commit()

  def ensure_version(self, *, allow_vector_cache_reset: bool = True):
    db_version_entry = self._con.execute("SELECT version FROM db_version").fetchone()
    db_version = db_version_entry["version"] if db_version_entry else 1
    if db_version == self.VERSION:
      return
    if db_version > self.VERSION:
      print(
        "found index version newer than this version of rclip can support;"
        " please, update rclip: https://github.com/yurijmikhalevich/rclip/blob/main/README.md#installation",
        file=sys.stderr,
      )
      sys.exit(1)
    if db_version < 3 and self._has_cached_vectors():
      if not allow_vector_cache_reset:
        raise Exception(
          'the existing vector cache is incompatible with this rclip version; rerun without "--no-indexing" to rebuild it.'
        )
      print(
        "The existing vector cache is incompatible with this rclip version; deleting cached vectors and rebuilding the index.",
        file=sys.stderr,
      )
    if db_version < 2:
      # Unused since rclip stopped flagging images during indexing; kept so older versions can still open the index.
      self._con.execute("ALTER TABLE images ADD COLUMN indexing BOOLEAN")
      db_version = 2
    if db_version < 3:
      # Model changed from ViT-B-32-quickgelu/openai to ViT-B-32-256/datacomp_s34b_b86k;
      # all embeddings must be recomputed.
      self._con.execute("DELETE FROM images")
      db_version = 3
    if db_version < 4:
      # Add hash column for rename/duplicate detection
      self._con.execute("ALTER TABLE images ADD COLUMN hash TEXT")
      self._con.execute("CREATE INDEX IF NOT EXISTS hash_index ON images(hash) WHERE deleted IS NULL")
      db_version = 4
    if db_version < self.VERSION:
      raise Exception("migration to a newer index version isn't implemented")
    if db_version_entry:
      self._con.execute("UPDATE db_version SET version=?", (self.VERSION,))
    else:
      self._con.execute("INSERT INTO db_version(version) VALUES (?)", (self.VERSION,))
    self._con.commit()

  def ensure_page_size(self):
    if self._con.execute("PRAGMA page_size").fetchone()[0] >= self.PAGE_SIZE:
      return
    images = self._con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    if images:
      # SQLite writes the rebuilt database to a temporary file and then copies it back through a journal
      db_size = os.path.getsize(self._filename)
      needed_space = f"{2 * db_size / 1e9:.1f} GB"
      if shutil.disk_usage(os.path.dirname(os.path.abspath(self._filename))).free < 2 * db_size:
        print(
          f"skipping the index rebuild that makes it smaller and faster to search: it needs {needed_space} of free disk space",
          file=sys.stderr,
        )
        return
      if images >= self.REBUILD_CONFIRMATION_IMAGES and sys.stdin.isatty() and sys.stderr.isatty():
        print(
          f"rclip can rebuild its index of {images:,} images to make it about 40% smaller and faster to search;"
          f" this runs once, may take a few minutes, and needs {needed_space} of free disk space. Rebuild it now? [Y/n] ",
          end="",
          file=sys.stderr,
          flush=True,
        )
        try:
          answer = input()
        except EOFError:
          answer = "n"
        if answer.strip().lower() not in ("", "y", "yes"):
          print("skipping the index rebuild; rclip will ask again next time", file=sys.stderr)
          return
      print("rebuilding the index once to make it smaller and faster to search", file=sys.stderr)
    # SQLite keeps the page size of a database in WAL mode, so the rebuild runs with a rollback journal
    journal_mode = self._con.execute("PRAGMA journal_mode").fetchone()[0]
    if journal_mode == "wal":
      self._con.execute("PRAGMA journal_mode=DELETE")
    self._con.execute("VACUUM")
    if journal_mode == "wal":
      self._con.execute("PRAGMA journal_mode=WAL")

  def _has_cached_vectors(self) -> bool:
    return self._con.execute("SELECT 1 FROM images LIMIT 1").fetchone() is not None

  def commit(self):
    self._con.commit()

  @staticmethod
  def _get_dirpath_range(path: str) -> tuple[str, str]:
    """Returns the [start, end) filepath range covering every file under `path`.

    A range lets SQLite use the filepath index, while a LIKE prefix has to be
    evaluated against every row."""
    # Normalize the directory prefix first so drive roots like "Y:\\" don't become "Y:\\\\".
    if not path.endswith(("/", "\\")):
      separator = "\\" if "\\" in path and "/" not in path else os.path.sep
      path += separator
    # The trailing separator is ASCII, so incrementing it yields the smallest
    # string that is greater than every filepath under `path`.
    return path, path[:-1] + chr(ord(path[-1]) + 1)

  def upsert_image(self, image: NewImage, commit: bool = True):
    self._con.execute(
      """
      INSERT INTO images(deleted, indexing, filepath, modified_at, size, vector, hash)
      VALUES (:deleted, :indexing, :filepath, :modified_at, :size, :vector, :hash)
      ON CONFLICT(filepath) DO UPDATE SET
        deleted=:deleted, indexing=:indexing, modified_at=:modified_at, size=:size, vector=:vector, hash=:hash
    """,
      {"deleted": None, "indexing": None, **image},
    )
    if commit:
      self._con.commit()

  def get_image_states(self, filepaths: list[str]) -> dict[str, ImageState]:
    """Fetches the fields used to detect changes by filepath, without reading the image vectors."""
    cur = self._con.execute(
      f"SELECT filepath, modified_at, size, deleted FROM images WHERE filepath IN ({','.join('?' * len(filepaths))})",
      filepaths,
    )
    return {row["filepath"]: row for row in cur}

  def mark_images_as_deleted(self, filepaths: Iterable[str]):
    self._con.executemany(
      "UPDATE images SET deleted = 1, indexing = NULL WHERE filepath = ?", ((f,) for f in filepaths)
    )
    self._con.commit()

  def restore_image(self, filepath: str, commit: bool = True):
    self._con.execute("UPDATE images SET deleted = NULL, indexing = NULL WHERE filepath = ?", (filepath,))
    if commit:
      self._con.commit()

  def get_images_by_hash(self, hash_value: str, file_size: int) -> list[Image]:
    cur = self._con.execute(
      "SELECT * FROM images WHERE hash = ? AND size = ? AND deleted IS NULL",
      (hash_value, file_size),
    )
    return cast(list[Image], [dict(row) for row in cur.fetchall()])

  def get_image_vectors_by_dir_path(self, path: str) -> sqlite3.Cursor:
    start, end = self._get_dirpath_range(path)
    return self._con.execute(
      "SELECT filepath, vector FROM images WHERE filepath >= ? AND filepath < ? AND deleted IS NULL",
      (start, end),
    )

  def get_filepaths_by_dir_path(self, path: str) -> sqlite3.Cursor:
    start, end = self._get_dirpath_range(path)
    return self._con.execute(
      "SELECT filepath FROM images WHERE filepath >= ? AND filepath < ? AND deleted IS NULL", (start, end)
    )

  def get_image_filepaths_by_dir_path(
    self, path: str, after: tuple[float, str] | None = None
  ) -> sqlite3.Cursor:
    start, end = self._get_dirpath_range(path)
    # SQLite would otherwise pick the filepath index and sort every image under `path` for each page; walking
    # the recency index returns a page as soon as it finds enough images under `path`.
    query = (
      "SELECT filepath, modified_at FROM images INDEXED BY recent_images"
      " WHERE filepath >= ? AND filepath < ? AND deleted IS NULL"
    )
    parameters: list[object] = [start, end]
    if after is not None:
      query += " AND modified_at <= ? AND (modified_at < ? OR filepath > ?)"
      parameters.extend((after[0], after[0], after[1]))
    return self._con.execute(query + " ORDER BY modified_at DESC, filepath", parameters)
