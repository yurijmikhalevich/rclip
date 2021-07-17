import pathlib
import sqlite3
from typing import Any, Dict, List, Optional, TypedDict, Union


class ImageOmittable(TypedDict, total=False):
  deleted: bool


class NewImage(ImageOmittable):
  filepath: str
  modified_at: float
  size: int
  vector: bytes


class Image(NewImage):
  id: int


def dict_factory(cur: sqlite3.Cursor, row: List[Any]) -> Dict[str, Any]:
  dict_row: Dict[str, Any] = {}
  for idx, col in enumerate(cur.description):
    dict_row[col[0]] = row[idx]
  return dict_row


class DB:
  def __init__(self, filename: Union[str, pathlib.Path]):
    self._con = sqlite3.connect(filename)
    self._con.row_factory = dict_factory
    self.ensure_tables()

  def ensure_tables(self):
    self._con.execute('''
      CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        deleted BOOLEAN,
        filepath TEXT NOT NULL UNIQUE,
        modified_at DATETIME NOT NULL,
        size INTEGER NOT NULL,
        vector BLOB NOT NULL
      )
    ''')
    self._con.commit()

  def upsert_image(self, image: NewImage):
    self._con.execute('''
      INSERT INTO images(deleted, filepath, modified_at, size, vector)
      VALUES (:deleted, :filepath, :modified_at, :size, :vector)
      ON CONFLICT(filepath) DO UPDATE SET
        deleted=:deleted, modified_at=:modified_at, size=:size, vector=:vector
    ''', {'deleted': None, **image})
    self._con.commit()

  def delete_image(self, id: int):
    self._con.execute('UPDATE images SET deleted = 1 WHERE id = ?', (id,))
    self._con.commit()

  def get_image(self, **kwargs: Any) -> Optional[Image]:
    query = ' AND '.join(f'{key}=:{key}' for key in kwargs)
    cur = self._con.execute(f'SELECT * FROM images WHERE {query} LIMIT 1', kwargs)
    return cur.fetchone()

  def get_images_by_dir_path(self, path: str) -> sqlite3.Cursor:
    return self._con.execute(f'SELECT filepath, vector FROM images WHERE filepath LIKE ?', (path + '/%',))
