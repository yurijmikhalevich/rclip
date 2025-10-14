import os.path
import pathlib
import sqlite3
from typing import Any, Optional, TypedDict, Union


class ImageOmittable(TypedDict, total=False):
  deleted: bool


class NewImage(ImageOmittable):
  filepath: str
  modified_at: float
  size: int
  vector: bytes
  hash: Optional[str] = None


class Image(NewImage):
  id: int


class DB:
  def __init__(self, filename: Union[str, pathlib.Path]):
    self._con = sqlite3.connect(filename)
    self._con.row_factory = sqlite3.Row
    self.ensure_tables()

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
        vector BLOB NOT NULL,
        hash TEXT
      )
    """)
    # Query for images
    self._con.execute("CREATE UNIQUE INDEX IF NOT EXISTS existing_images ON images(filepath) WHERE deleted IS NULL")
    self._con.execute("CREATE INDEX IF NOT EXISTS hash_index ON images(hash) WHERE deleted IS NULL")

    # Add hash column to existing databases if it doesn't exist
    columns = self._con.execute("PRAGMA table_info(images)").fetchall()
    column_names = [col["name"] for col in columns]
    if "hash" not in column_names:
      self._con.execute("ALTER TABLE images ADD COLUMN hash TEXT")

    self._con.commit()


  def commit(self):
    self._con.commit()

  def upsert_image(self, image: NewImage, commit: bool = True):
    self._con.execute(
      """
      INSERT INTO images(deleted, filepath, modified_at, size, vector, hash)
      VALUES (:deleted, :filepath, :modified_at, :size, :vector, :hash)
      ON CONFLICT(filepath) DO UPDATE SET
        deleted=:deleted, modified_at=:modified_at, size=:size, vector=:vector, hash=COALESCE(:hash, hash)
    """,
      {"deleted": None, **image},
    )
    if commit:
      self._con.commit()


  def get_image(self, **kwargs: Any) -> Optional[Image]:
    query_parts = [f"{key}=:{key}" for key in kwargs]
    query_parts.append("deleted IS NULL")
    query = " AND ".join(query_parts)
    cur = self._con.execute(f"SELECT * FROM images WHERE {query} LIMIT 1", kwargs)
    return cur.fetchone()

  def get_images_by_hash(self, hash_value: str) -> list[Image]:
    cur = self._con.execute("SELECT * FROM images WHERE hash = ? AND deleted IS NULL", (hash_value,))
    return cur.fetchall()

  def get_image_vectors_by_dir_path(self, path: str) -> sqlite3.Cursor:
    return self._con.execute(
      "SELECT filepath, vector FROM images WHERE filepath LIKE ? AND deleted IS NULL", (path + f"{os.path.sep}%",)
    )
