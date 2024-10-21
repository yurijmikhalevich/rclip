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
  hash: str


class Image(NewImage):
  id: int


class DB:
  VERSION = 3

  def __init__(self, filename: Union[str, pathlib.Path]):
    self._con = sqlite3.connect(filename)
    self._con.row_factory = sqlite3.Row
    self.ensure_version()
    self.ensure_tables()
    
  def close(self):
    self._con.commit()
    self._con.close()

  def ensure_tables(self):
    self._con.execute('''
      CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        deleted BOOLEAN,
        filepath TEXT NOT NULL UNIQUE,
        modified_at DATETIME NOT NULL,
        size INTEGER NOT NULL,
        vector BLOB NOT NULL,
        hash TEXT,
        indexing BOOLEAN
      )
    ''')
    # Query for images
    self._con.execute('CREATE UNIQUE INDEX IF NOT EXISTS existing_images ON images(filepath) WHERE deleted IS NULL')
    self._con.execute('CREATE TABLE IF NOT EXISTS db_version (version INTEGER)')
    # Check if 'hash' column exists before creating the index
    cursor = self._con.execute("PRAGMA table_info(images)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'hash' in columns:
        self._con.execute('CREATE INDEX IF NOT EXISTS image_hash_index ON images(hash)')
    self._con.commit()

  def ensure_version(self):
    db_version_entry = self._con.execute('SELECT version FROM db_version').fetchone()
    db_version = db_version_entry['version'] if db_version_entry else 1
    if db_version == self.VERSION:
        return
    if db_version > self.VERSION:
      raise Exception(
        'found index version newer than this version of rclip can support;'
        ' please, update rclip: https://github.com/yurijmikhalevich/rclip/blob/main/README.md#installation',
      )
    if db_version < 2:
      self._con.execute('ALTER TABLE images ADD COLUMN indexing BOOLEAN')
      db_version = 2
    if db_version < 3:
      self._con.execute('ALTER TABLE images ADD COLUMN hash TEXT')
      db_version = 3
    if db_version < self.VERSION:
      raise Exception('migration to a newer index version isn\'t implemented')
    if db_version_entry:
      self._con.execute('UPDATE db_version SET version=?', (self.VERSION,))
    else:
      self._con.execute('INSERT INTO db_version(version) VALUES (?)', (self.VERSION,))
    self._con.commit()

  def commit(self):
    self._con.commit()

  def upsert_image(self, image: NewImage, commit: bool = True):
    self._con.execute('''
      INSERT INTO images(deleted, indexing, filepath, modified_at, size, vector, hash)
      VALUES (:deleted, :indexing, :filepath, :modified_at, :size, :vector, :hash)
      ON CONFLICT(filepath) DO UPDATE SET
        deleted=:deleted, indexing=:indexing, modified_at=:modified_at, size=:size, vector=:vector, hash=:hash
    ''', {'deleted': None, 'indexing': None, **image})
    if commit:
      self._con.commit()

  def remove_indexing_flag_from_all_images(self, commit: bool = True):
    self._con.execute('UPDATE images SET indexing = NULL')
    if commit:
      self._con.commit()

  def flag_images_in_a_dir_as_indexing(self, path: str, commit: bool = True):
    self._con.execute('UPDATE images SET indexing = 1 WHERE filepath LIKE ?', (path + f'{os.path.sep}%',))
    if commit:
      self._con.commit()

  def flag_indexing_images_in_a_dir_as_deleted(self, path: str):
    self._con.execute(
      'UPDATE images SET deleted = 1, indexing = NULL WHERE filepath LIKE ? AND indexing = 1',
      (path + f'{os.path.sep}%',),
    )
    self._con.commit()

  def remove_indexing_flag(self, filepath: str, commit: bool = True):
    self._con.execute('UPDATE images SET indexing = NULL WHERE filepath = ?', (filepath,))
    if commit:
      self._con.commit()

  def get_image(self, **kwargs: Any) -> Optional[Image]:
    query = ' AND '.join(f'{key}=:{key}' for key in kwargs)
    cur = self._con.execute(f'SELECT * FROM images WHERE {query} LIMIT 1', kwargs)
    return cur.fetchone()

  def get_image_vectors_by_dir_path(self, path: str) -> sqlite3.Cursor:
    return self._con.execute(
      f'SELECT filepath, vector FROM images WHERE filepath LIKE ? AND deleted IS NULL', (path + f'{os.path.sep}%',)
    )

  def get_image_by_hash(self, hash: str) -> Optional[Image]:
    cur = self._con.execute('SELECT * FROM images WHERE hash = ? AND deleted IS NULL LIMIT 1', (hash,))
    return cur.fetchone()

  def update_image_filepath(self, old_filepath: str, new_filepath: str, commit: bool = True):
    try:
        self._con.execute('UPDATE images SET filepath = ? WHERE filepath = ?', (new_filepath, old_filepath))
    except sqlite3.IntegrityError:
        # If the new filepath already exists, we need to merge the entries
        existing_image = self.get_image(filepath=new_filepath)
        if existing_image:
            # Delete the old entry
            self._con.execute('DELETE FROM images WHERE filepath = ?', (old_filepath,))
        else:
            # If there's no existing image with the new filepath, re-raise the exception
            raise
    if commit:
        self._con.commit()