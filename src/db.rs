//! The SQLite index, schema-compatible with the Python version of rclip.
//!
//! Writes follow Python's `sqlite3` transaction semantics: the first write after a commit opens a
//! transaction, and nothing becomes durable until [`Db::commit`]. Callers rely on this to batch
//! thousands of updates into one transaction while indexing.

use std::io::IsTerminal;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};

pub const VERSION: i64 = 4;
/// Each image takes about 2.2 KB, so a 4 KB page holds only one; a 16 KB page holds seven, which makes the
/// index about 40% smaller and faster to read.
const PAGE_SIZE: i64 = 16384;
/// A search reads every vector in the index, and reading them through a memory map skips copying each page
/// into SQLite's page cache. SQLite caps the map at its compile-time maximum, which is 2 GB by default.
const MMAP_SIZE: i64 = 1 << 31;
/// Rebuilding takes about half a minute per million images on an SSD, so larger indexes ask before it.
const REBUILD_CONFIRMATION_IMAGES: i64 = 100_000;

/// Fields used to detect whether a file changed since it was indexed.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageState {
  pub deleted: bool,
  pub modified_at: f64,
  pub size: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NewImage<'a> {
  pub filepath: &'a str,
  pub modified_at: f64,
  pub size: i64,
  pub vector: &'a [u8],
  pub hash: &'a str,
}

/// Position in the "recently modified first" listing, used to fetch the next page.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageCursor {
  pub modified_at: f64,
  pub filepath: String,
}

pub struct Db {
  connection: Connection,
  /// The database file, used to open extra read-only connections for parallel scans.
  path: Option<std::path::PathBuf>,
}

/// Scans covering fewer rows than this run on a single connection.
const PARALLEL_SCAN_MIN_ROWS: i64 = 50_000;
const PARALLEL_SCAN_THREADS: usize = 4;

/// Whether a vector cache built by an incompatible model version may be discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorCacheReset {
  Allow,
  Forbid,
}

impl Db {
  pub fn open(path: &Path, reset: VectorCacheReset) -> Result<Self> {
    let connection = Connection::open(path).with_context(|| format!("cannot open the index at {}", path.display()))?;
    // The page size applies to a new database right away, and to an existing one once it's rebuilt.
    connection.pragma_update(None, "page_size", PAGE_SIZE)?;
    connection.pragma_update(None, "mmap_size", MMAP_SIZE)?;
    // Other tools may switch the index to WAL mode, which makes indexing slower and keeps SQLite from changing
    // the page size. An index another process has open can't leave WAL mode, so it's left for a later run.
    connection.pragma_update(None, "journal_mode", "DELETE").ok();
    let db = Db { connection, path: Some(path.to_path_buf()) };
    db.ensure_tables()?;
    db.ensure_version(reset)?;
    db.ensure_page_size(path)?;
    Ok(db)
  }

  #[cfg(test)]
  pub fn open_in_memory() -> Result<Self> {
    let db = Db { connection: Connection::open_in_memory()?, path: None };
    db.ensure_tables()?;
    db.ensure_version(VectorCacheReset::Allow)?;
    Ok(db)
  }

  fn ensure_tables(&self) -> Result<()> {
    self.connection.execute_batch(
      "CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        deleted BOOLEAN,
        filepath TEXT NOT NULL UNIQUE,
        modified_at DATETIME NOT NULL,
        size INTEGER NOT NULL,
        vector BLOB NOT NULL
      );
      CREATE UNIQUE INDEX IF NOT EXISTS existing_images ON images(filepath) WHERE deleted IS NULL;
      CREATE INDEX IF NOT EXISTS recent_images ON images(modified_at DESC, filepath) WHERE deleted IS NULL;
      CREATE TABLE IF NOT EXISTS db_version (version INTEGER);",
    )?;
    Ok(())
  }

  fn ensure_version(&self, reset: VectorCacheReset) -> Result<()> {
    let stored: Option<i64> =
      self.connection.query_row("SELECT version FROM db_version", [], |row| row.get(0)).optional()?;
    let mut version = stored.unwrap_or(1);
    if version == VERSION {
      return Ok(());
    }
    if version > VERSION {
      bail!(
        "found index version newer than this version of rclip can support; \
         please, update rclip: https://github.com/yurijmikhalevich/rclip/blob/main/README.md#installation"
      );
    }
    if version < 3 && self.has_cached_vectors()? {
      if reset == VectorCacheReset::Forbid {
        bail!(
          "the existing vector cache is incompatible with this rclip version; \
           rerun without \"--no-indexing\" to rebuild it."
        );
      }
      eprintln!(
        "The existing vector cache is incompatible with this rclip version; \
         deleting cached vectors and rebuilding the index."
      );
    }
    // The upgrade runs in one transaction with the version update, so an interrupted one is retried from the start.
    if version < 2 {
      self.write("ALTER TABLE images ADD COLUMN indexing BOOLEAN", [])?;
      version = 2;
    }
    if version < 3 {
      // The model changed from ViT-B-32-quickgelu/openai to ViT-B-32-256/datacomp_s34b_b86k, so
      // every embedding must be recomputed.
      self.write("DELETE FROM images", [])?;
      version = 3;
    }
    if version < 4 {
      // The hash column enables rename and duplicate detection.
      self.write("ALTER TABLE images ADD COLUMN hash TEXT", [])?;
      self.write("CREATE INDEX IF NOT EXISTS hash_index ON images(hash) WHERE deleted IS NULL", [])?;
    }
    if stored.is_some() {
      self.write("UPDATE db_version SET version=?", [VERSION])?;
    } else {
      self.write("INSERT INTO db_version(version) VALUES (?)", [VERSION])?;
    }
    self.commit()
  }

  /// Rebuilds an index stored in small pages, which makes it smaller and faster to search.
  fn ensure_page_size(&self, path: &Path) -> Result<()> {
    let page_size: i64 = self.connection.pragma_query_value(None, "page_size", |row| row.get(0))?;
    let journal_mode: String = self.connection.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
    // The rebuild can't change the page size of an index that is still in WAL mode.
    if page_size >= PAGE_SIZE || journal_mode == "wal" {
      return Ok(());
    }
    let images: i64 = self.connection.query_row("SELECT COUNT(*) FROM images", [], |row| row.get(0))?;
    if images > 0 {
      // SQLite writes the rebuilt database to a temporary file and then copies it back through a journal.
      let needed_bytes = 2 * std::fs::metadata(path)?.len();
      let needed_space = format!("{:.1} GB", needed_bytes as f64 / 1e9);
      if fs4::available_space(path.parent().unwrap_or(path))? < needed_bytes {
        eprintln!(
          "skipping the index rebuild that makes it smaller and faster to search: \
           it needs {needed_space} of free disk space"
        );
        return Ok(());
      }
      if images >= REBUILD_CONFIRMATION_IMAGES && std::io::stdin().is_terminal() && std::io::stderr().is_terminal() {
        eprint!(
          "rclip can rebuild its index of {images} images to make it about 40% smaller and faster to search; \
           this runs once, may take a few minutes, and needs {needed_space} of free disk space. \
           Rebuild it now? [Y/n] "
        );
        let mut answer = String::new();
        // A closed or unreadable stdin declines.
        let read = std::io::stdin().read_line(&mut answer).unwrap_or(0);
        if read == 0 || !["", "y", "yes"].contains(&answer.trim().to_lowercase().as_str()) {
          eprintln!("skipping the index rebuild; rclip will ask again next time");
          return Ok(());
        }
      }
      eprintln!("rebuilding the index once to make it smaller and faster to search");
    }
    self.connection.execute_batch("VACUUM")?;
    Ok(())
  }

  fn has_cached_vectors(&self) -> Result<bool> {
    Ok(self.connection.query_row("SELECT 1 FROM images LIMIT 1", [], |_| Ok(())).optional()?.is_some())
  }

  /// Executes a write statement, opening a transaction first if none is active.
  fn write(&self, sql: &str, params: impl rusqlite::Params) -> Result<()> {
    if self.connection.is_autocommit() {
      self.connection.execute_batch("BEGIN")?;
    }
    self.connection.prepare_cached(sql)?.execute(params)?;
    Ok(())
  }

  pub fn commit(&self) -> Result<()> {
    if !self.connection.is_autocommit() {
      self.connection.execute_batch("COMMIT")?;
    }
    Ok(())
  }

  pub fn upsert_image(&self, image: &NewImage<'_>) -> Result<()> {
    self.write(
      "INSERT INTO images(deleted, indexing, filepath, modified_at, size, vector, hash)
      VALUES (NULL, NULL, ?1, ?2, ?3, ?4, ?5)
      ON CONFLICT(filepath) DO UPDATE SET
        deleted=NULL, indexing=NULL, modified_at=?2, size=?3, vector=?4, hash=?5",
      params![image.filepath, image.modified_at, image.size, image.vector, image.hash],
    )
  }

  pub fn mark_deleted(&self, filepath: &str) -> Result<()> {
    self.write("UPDATE images SET deleted = 1, indexing = NULL WHERE filepath = ?", [filepath])
  }

  pub fn restore_image(&self, filepath: &str) -> Result<()> {
    self.write("UPDATE images SET deleted = NULL, indexing = NULL WHERE filepath = ?", [filepath])
  }

  pub fn image_state(&self, filepath: &str) -> Result<Option<ImageState>> {
    let mut statement =
      self.connection.prepare_cached("SELECT modified_at, size, deleted FROM images WHERE filepath = ? LIMIT 1")?;
    Ok(
      statement
        .query_row([filepath], |row| {
          Ok(ImageState {
            modified_at: row.get(0)?,
            size: row.get(1)?,
            deleted: row.get::<_, Option<bool>>(2)?.unwrap_or(false),
          })
        })
        .optional()?,
    )
  }

  /// Returns the vector of an existing image with the same content fingerprint, if any.
  pub fn vector_by_hash(&self, hash: &str, size: i64) -> Result<Option<Vec<u8>>> {
    let mut statement =
      self.connection.prepare_cached("SELECT vector FROM images WHERE hash = ? AND size = ? AND deleted IS NULL")?;
    Ok(statement.query_row(params![hash, size], |row| row.get(0)).optional()?)
  }

  /// Calls `visit` with the path and raw vector of every indexed image under `path`, stopping at
  /// the first error. Each scanning thread gets its own state from `new_state`; the states are
  /// returned for the caller to merge.
  ///
  /// When the directory covers a large share of the index, rows are read in table order by several
  /// threads, which is much faster than following the filepath index through a large file.
  pub fn scan_vectors<S: Send>(
    &self,
    path: &str,
    new_state: impl Fn() -> S + Sync,
    visit: impl Fn(&mut S, &str, &[u8]) -> Result<()> + Sync,
  ) -> Result<Vec<S>> {
    let (start, end) = dirpath_range(path);
    let (count, max_id): (i64, i64) = self.connection.query_row(
      "SELECT (SELECT count(*) FROM images WHERE filepath >= ?1 AND filepath < ?2 AND deleted IS NULL),
        (SELECT coalesce(max(id), 0) FROM images)",
      [&start, &end],
      |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let threads = PARALLEL_SCAN_THREADS.min(std::thread::available_parallelism().map_or(1, usize::from));
    let dense = count >= PARALLEL_SCAN_MIN_ROWS && count * 4 >= max_id;
    let (Some(db_path), true, true) = (&self.path, dense, threads > 1) else {
      let mut state = new_state();
      let mut statement = self.connection.prepare_cached(
        "SELECT filepath, vector FROM images WHERE filepath >= ? AND filepath < ? AND deleted IS NULL",
      )?;
      let mut rows = statement.query([&start, &end])?;
      while let Some(row) = rows.next()? {
        visit(&mut state, row.get_ref(0)?.as_str()?, row.get_ref(1)?.as_blob()?)?;
      }
      return Ok(vec![state]);
    };

    let chunk = max_id / threads as i64 + 1;
    std::thread::scope(|scope| {
      let workers: Vec<_> = (0..threads as i64)
        .map(|index| {
          let (start, end, new_state, visit) = (&start, &end, &new_state, &visit);
          scope.spawn(move || -> Result<S> {
            let connection = Connection::open_with_flags(
              db_path,
              rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )?;
            connection.pragma_update(None, "mmap_size", MMAP_SIZE)?;
            let mut statement = connection.prepare(
              "SELECT filepath, vector FROM images NOT INDEXED
              WHERE id >= ?1 AND id < ?2 AND filepath >= ?3 AND filepath < ?4 AND deleted IS NULL",
            )?;
            let mut rows = statement.query(params![index * chunk, (index + 1) * chunk, start, end])?;
            let mut state = new_state();
            while let Some(row) = rows.next()? {
              visit(&mut state, row.get_ref(0)?.as_str()?, row.get_ref(1)?.as_blob()?)?;
            }
            Ok(state)
          })
        })
        .collect();
      workers.into_iter().map(|worker| worker.join().expect("scan threads do not panic")).collect()
    })
  }

  /// Returns the path of every indexed image under `path`.
  pub fn filepaths(&self, path: &str) -> Result<Vec<String>> {
    let (start, end) = dirpath_range(path);
    let mut statement = self
      .connection
      .prepare_cached("SELECT filepath FROM images WHERE filepath >= ? AND filepath < ? AND deleted IS NULL")?;
    let filepaths = statement.query_map([start, end], |row| row.get(0))?.collect::<rusqlite::Result<_>>()?;
    Ok(filepaths)
  }

  /// Calls `visit` for every indexed image under `path`, most recently modified first, starting
  /// after `after`. Stops when `visit` returns `false`.
  pub fn for_each_filepath(
    &self,
    path: &str,
    after: Option<&ImageCursor>,
    mut visit: impl FnMut(&str, f64) -> Result<bool>,
  ) -> Result<()> {
    let (start, end) = dirpath_range(path);
    // SQLite would otherwise pick the filepath index and sort every image under `path` for each page; walking
    // the recency index returns a page as soon as it finds enough images under `path`.
    let mut sql = "SELECT filepath, modified_at FROM images INDEXED BY recent_images
      WHERE filepath >= ?1 AND filepath < ?2 AND deleted IS NULL"
      .to_string();
    if after.is_some() {
      sql += " AND modified_at <= ?3 AND (modified_at < ?3 OR filepath > ?4)";
    }
    sql += " ORDER BY modified_at DESC, filepath";
    let mut statement = self.connection.prepare_cached(&sql)?;
    let mut rows = match after {
      Some(cursor) => statement.query(params![start, end, cursor.modified_at, cursor.filepath])?,
      None => statement.query(params![start, end])?,
    };
    while let Some(row) = rows.next()? {
      if !visit(row.get_ref(0)?.as_str()?, row.get(1)?)? {
        break;
      }
    }
    Ok(())
  }
}

/// Returns the `[start, end)` filepath range covering every file under `path`.
///
/// A range lets SQLite use the filepath index, while a LIKE prefix has to be evaluated against
/// every row.
pub fn dirpath_range(path: &str) -> (String, String) {
  let mut start = path.to_string();
  // Normalize the directory prefix first so drive roots like "Y:\" don't become "Y:\\".
  if !start.ends_with(['/', '\\']) {
    let separator = if start.contains('\\') && !start.contains('/') { '\\' } else { std::path::MAIN_SEPARATOR };
    start.push(separator);
  }
  // The trailing separator is ASCII, so incrementing it yields the smallest string that is greater
  // than every filepath under `path`.
  let mut end = start.clone();
  let last = end.pop().expect("the range start ends with a separator");
  end.push((last as u8 + 1) as char);
  (start, end)
}

#[cfg(test)]
mod tests {
  use super::*;

  // The tests use Windows paths because `dirpath_range` treats them the same way on every platform.

  fn insert(db: &Db, filepath: &str, modified_at: f64) {
    db.upsert_image(&NewImage { filepath, modified_at, size: 1, vector: &[1, 2, 3, 4], hash: "hash" }).unwrap();
  }

  fn vectors(db: &Db, path: &str) -> Vec<String> {
    let states = db
      .scan_vectors(path, Vec::new, |paths: &mut Vec<String>, filepath, _| {
        paths.push(filepath.to_string());
        Ok(())
      })
      .unwrap();
    let mut paths: Vec<String> = states.into_iter().flatten().collect();
    paths.sort();
    paths
  }

  #[test]
  fn dirpath_range_handles_separators() {
    let separator = std::path::MAIN_SEPARATOR;
    let after_separator = (separator as u8 + 1) as char;
    assert_eq!(dirpath_range("/photos"), (format!("/photos{separator}"), format!("/photos{after_separator}")));
    assert_eq!(dirpath_range("/photos/"), ("/photos/".into(), "/photos0".into()));
    assert_eq!(dirpath_range("Y:\\"), ("Y:\\".into(), "Y:]".into()));
    assert_eq!(dirpath_range("C:\\photos"), ("C:\\photos\\".into(), "C:\\photos]".into()));
  }

  #[test]
  fn vectors_are_scoped_to_the_directory() {
    let db = Db::open_in_memory().unwrap();
    insert(&db, r"Y:\photos\a.jpg", 1.0);
    insert(&db, r"Y:\photos\nested\b.jpg", 2.0);
    insert(&db, r"Y:\photos2\c.jpg", 3.0);
    insert(&db, r"Y:\photo%\d.jpg", 4.0);
    assert_eq!(vectors(&db, r"Y:\photos"), [r"Y:\photos\a.jpg", r"Y:\photos\nested\b.jpg"]);
    assert_eq!(vectors(&db, r"Y:\photo%"), [r"Y:\photo%\d.jpg"]);
  }

  #[test]
  fn scans_large_directories_in_parallel() {
    let directory = tempfile::tempdir().unwrap();
    let db = Db::open(&directory.path().join("db.sqlite3"), VectorCacheReset::Allow).unwrap();
    let count = PARALLEL_SCAN_MIN_ROWS as usize + 10;
    for index in 0..count {
      insert(&db, &format!(r"Y:\photos\{index}.jpg"), 1.0);
    }
    insert(&db, r"Y:\other\a.jpg", 1.0);
    db.commit().unwrap();
    let states = db
      .scan_vectors(
        r"Y:\photos",
        || 0usize,
        |count, _, _| {
          let _: () = *count += 1;
          Ok(())
        },
      )
      .unwrap();
    assert!(states.len() > 1 || std::thread::available_parallelism().map_or(1, usize::from) == 1);
    assert_eq!(states.iter().sum::<usize>(), count);
  }

  #[test]
  fn image_state_round_trips() {
    let db = Db::open_in_memory().unwrap();
    insert(&db, "/a.jpg", 1_700_000_000.0);
    insert(&db, "/b.jpg", 1_700_000_000.25);
    db.commit().unwrap();
    assert_eq!(
      db.image_state("/a.jpg").unwrap(),
      Some(ImageState { deleted: false, modified_at: 1_700_000_000.0, size: 1 })
    );
    assert_eq!(db.image_state("/b.jpg").unwrap().unwrap().modified_at, 1_700_000_000.25);
    assert_eq!(db.image_state("/missing.jpg").unwrap(), None);
  }

  #[test]
  fn deletion_flow_marks_and_restores_images() {
    let db = Db::open_in_memory().unwrap();
    insert(&db, r"Y:\photos\a.jpg", 1.0);
    insert(&db, r"Y:\photos\b.jpg", 1.0);
    db.commit().unwrap();
    db.mark_deleted(r"Y:\photos\b.jpg").unwrap();
    assert!(db.image_state(r"Y:\photos\b.jpg").unwrap().unwrap().deleted);
    assert_eq!(vectors(&db, r"Y:\photos"), [r"Y:\photos\a.jpg"]);
    db.restore_image(r"Y:\photos\b.jpg").unwrap();
    assert_eq!(vectors(&db, r"Y:\photos"), [r"Y:\photos\a.jpg", r"Y:\photos\b.jpg"]);
  }

  #[test]
  fn finds_vectors_by_hash_and_size() {
    let db = Db::open_in_memory().unwrap();
    insert(&db, "/a.jpg", 1.0);
    assert_eq!(db.vector_by_hash("hash", 1).unwrap(), Some(vec![1, 2, 3, 4]));
    assert_eq!(db.vector_by_hash("hash", 2).unwrap(), None);
  }

  #[test]
  fn lists_recent_filepaths_with_a_cursor() {
    let db = Db::open_in_memory().unwrap();
    insert(&db, r"Y:\p\old.jpg", 1.0);
    insert(&db, r"Y:\p\new-a.jpg", 2.0);
    insert(&db, r"Y:\p\new-b.jpg", 2.0);
    let list = |after: Option<&ImageCursor>| {
      let mut paths = Vec::new();
      db.for_each_filepath(r"Y:\p", after, |filepath, _| {
        paths.push(filepath.to_string());
        Ok(true)
      })
      .unwrap();
      paths
    };
    assert_eq!(list(None), [r"Y:\p\new-a.jpg", r"Y:\p\new-b.jpg", r"Y:\p\old.jpg"]);
    let cursor = ImageCursor { modified_at: 2.0, filepath: r"Y:\p\new-a.jpg".into() };
    assert_eq!(list(Some(&cursor)), [r"Y:\p\new-b.jpg", r"Y:\p\old.jpg"]);
  }

  /// Creates an index holding one image in the 4 KB pages older versions used.
  fn create_small_page_index(path: &Path) {
    let db = Db { connection: Connection::open(path).unwrap(), path: None };
    db.ensure_tables().unwrap();
    db.ensure_version(VectorCacheReset::Allow).unwrap();
    insert(&db, r"Y:\photos\cat.jpg", 1.0);
    db.commit().unwrap();
  }

  fn pragma<T: rusqlite::types::FromSql>(db: &Db, name: &str) -> T {
    db.connection.pragma_query_value(None, name, |row| row.get(0)).unwrap()
  }

  #[test]
  fn new_indexes_use_large_pages() {
    let directory = tempfile::tempdir().unwrap();
    let db = Db::open(&directory.path().join("db.sqlite3"), VectorCacheReset::Allow).unwrap();
    assert_eq!(pragma::<i64>(&db, "page_size"), PAGE_SIZE);
  }

  #[test]
  fn rebuilds_small_page_indexes_with_large_pages() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("db.sqlite3");
    create_small_page_index(&path);
    Connection::open(&path).unwrap().pragma_update(None, "journal_mode", "WAL").unwrap();
    let db = Db::open(&path, VectorCacheReset::Allow).unwrap();
    assert_eq!(pragma::<i64>(&db, "page_size"), PAGE_SIZE);
    assert_eq!(pragma::<String>(&db, "journal_mode"), "delete");
    assert_eq!(db.filepaths(r"Y:\photos").unwrap(), [r"Y:\photos\cat.jpg"]);
  }

  #[test]
  fn keeps_small_pages_while_another_connection_holds_the_index_in_wal_mode() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("db.sqlite3");
    create_small_page_index(&path);
    let other = Connection::open(&path).unwrap();
    other.pragma_update(None, "journal_mode", "WAL").unwrap();
    let _: i64 = other.query_row("SELECT COUNT(*) FROM images", [], |row| row.get(0)).unwrap();
    let db = Db::open(&path, VectorCacheReset::Allow).unwrap();
    assert_eq!(pragma::<String>(&db, "journal_mode"), "wal");
    assert_eq!(pragma::<i64>(&db, "page_size"), 4096);
    assert_eq!(db.filepaths(r"Y:\photos").unwrap(), [r"Y:\photos\cat.jpg"]);
  }

  #[test]
  fn migrates_old_indexes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("db.sqlite3");
    let connection = Connection::open(&path).unwrap();
    connection
      .execute_batch(
        "CREATE TABLE images (id INTEGER PRIMARY KEY, deleted BOOLEAN, filepath TEXT NOT NULL UNIQUE,
          modified_at DATETIME NOT NULL, size INTEGER NOT NULL, vector BLOB NOT NULL);
        INSERT INTO images(filepath, modified_at, size, vector) VALUES ('/a.jpg', 1, 1, x'00');",
      )
      .unwrap();
    drop(connection);
    assert!(Db::open(&path, VectorCacheReset::Forbid).is_err());
    let db = Db::open(&path, VectorCacheReset::Allow).unwrap();
    assert_eq!(db.image_state("/a.jpg").unwrap(), None);
    let version: i64 = db.connection.query_row("SELECT version FROM db_version", [], |row| row.get(0)).unwrap();
    assert_eq!(version, VERSION);
  }

  #[test]
  fn retries_an_interrupted_upgrade() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("db.sqlite3");
    let connection = Connection::open(&path).unwrap();
    connection
      .execute_batch(
        "CREATE TABLE images (id INTEGER PRIMARY KEY, deleted BOOLEAN, filepath TEXT NOT NULL UNIQUE,
          modified_at DATETIME NOT NULL, size INTEGER NOT NULL, vector BLOB NOT NULL, indexing BOOLEAN);
        CREATE TABLE db_version (version INTEGER);
        INSERT INTO db_version(version) VALUES (3);
        CREATE TRIGGER interrupt BEFORE UPDATE ON db_version BEGIN SELECT RAISE(ABORT, 'interrupted'); END;",
      )
      .unwrap();
    assert!(Db::open(&path, VectorCacheReset::Allow).is_err());
    connection.execute_batch("DROP TRIGGER interrupt").unwrap();
    drop(connection);
    let db = Db::open(&path, VectorCacheReset::Allow).unwrap();
    insert(&db, "/a.jpg", 1.0);
    assert_eq!(db.vector_by_hash("hash", 1).unwrap(), Some(vec![1, 2, 3, 4]));
  }
}
