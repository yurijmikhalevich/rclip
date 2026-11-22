//! Directory traversal and the path filters shared by indexing, search, and browsing.

use std::collections::HashSet;
use std::fs::{self, Metadata};
use std::path::{MAIN_SEPARATOR, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::imaging::{IMAGE_EXT, IMAGE_RAW_EXT};

pub const DEFAULT_EXCLUDE_DIRS: &[&str] = &["@eaDir", "node_modules", ".git", "System Volume Information"];

/// Decides which files and directories rclip looks at.
#[derive(Clone, Debug)]
pub struct PathFilter {
  exclude_dirs: Vec<String>,
  skip_hidden: bool,
  extensions: HashSet<&'static str>,
}

impl PathFilter {
  pub fn new(exclude_dirs: Option<Vec<String>>, include_hidden: bool, raw_support: bool) -> Self {
    let exclude_dirs = exclude_dirs.unwrap_or_else(|| DEFAULT_EXCLUDE_DIRS.iter().map(|dir| dir.to_string()).collect());
    let mut extensions: HashSet<&'static str> = IMAGE_EXT.iter().copied().collect();
    if raw_support {
      extensions.extend(IMAGE_RAW_EXT);
    }
    PathFilter { exclude_dirs, skip_hidden: !include_hidden, extensions }
  }

  /// Whether a file name has a supported image extension (and something before it).
  pub fn is_image_name(&self, name: &str) -> bool {
    match name.rsplit_once('.') {
      Some((stem, extension)) if !stem.is_empty() => {
        // Extensions are ASCII, so lowercasing only ASCII keeps the comparison exact.
        self.extensions.contains(extension.to_ascii_lowercase().as_str())
      }
      _ => false,
    }
  }

  fn is_hidden(&self, name: &str) -> bool {
    self.skip_hidden && name.starts_with('.')
  }

  /// Whether a directory, given relative to the walk root, is excluded.
  ///
  /// Equivalent to the Python version's `^.+/(dir1|dir2)(/.+)?$` match against `./<relative>`:
  /// an excluded name must appear right after a separator and end at a separator or the end.
  fn is_excluded_dir(&self, relative_dir: &str) -> bool {
    let path = format!(".{MAIN_SEPARATOR}{relative_dir}");
    self.exclude_dirs.iter().any(|name| {
      let needle = format!("{MAIN_SEPARATOR}{name}");
      path.match_indices(&needle).any(|(position, _)| {
        let rest = &path[position + needle.len()..];
        rest.is_empty() || rest.starts_with(MAIN_SEPARATOR)
      })
    })
  }

  /// Whether an indexed file under `directory` should be hidden from search and browsing, so that
  /// the exclusion flags behave the same as during indexing.
  pub fn is_excluded(&self, filepath: &str, directory: &str) -> bool {
    let relative = relative_path(filepath, directory);
    if self.skip_hidden && relative.split(MAIN_SEPARATOR).any(|part| part.starts_with('.')) {
      return true;
    }
    let parent = relative.rsplit_once(MAIN_SEPARATOR).map_or("", |(parent, _)| parent);
    self.is_excluded_dir(parent)
  }
}

fn relative_path<'a>(filepath: &'a str, directory: &str) -> &'a str {
  let directory = directory.trim_end_matches(MAIN_SEPARATOR);
  filepath.strip_prefix(directory).and_then(|rest| rest.strip_prefix(MAIN_SEPARATOR)).unwrap_or(filepath)
}

/// A file found during the walk, with the metadata used to detect changes.
#[derive(Clone, Debug, PartialEq)]
pub struct FileEntry {
  pub path: String,
  pub modified_at: f64,
  pub size: i64,
}

/// Errors that do not stop the walk.
#[derive(Debug)]
pub enum WalkIssue {
  /// A directory could not be listed, or an entry such as a symlink to an inaccessible target could not be
  /// inspected; its previously indexed images must be kept as they are.
  UnreadableDir(String, std::io::Error),
  /// A file's metadata could not be read.
  Metadata(String, std::io::Error),
  /// A path is not valid UTF-8 and cannot be stored in the index.
  NonUtf8(PathBuf),
}

/// Walks `root` depth-first and calls `visit` for every image file that passes `filter`, until
/// `visit` returns `false`. Symlinks are followed, with cycle protection for symlinked directories.
pub fn walk(root: &str, filter: &PathFilter, mut visit: impl FnMut(Result<FileEntry, WalkIssue>) -> bool) {
  walk_paths(root, filter, |found| {
    let entry = match found {
      Found::File { path, metadata } => match metadata.map_or_else(|| fs::metadata(&path), Ok) {
        Ok(metadata) if metadata.is_file() => {
          Ok(FileEntry { modified_at: modified_at(&metadata), size: metadata.len() as i64, path })
        }
        Ok(_) => return true,
        Err(error) => Err(WalkIssue::Metadata(path, error)),
      },
      Found::Issue(issue) => Err(issue),
    };
    visit(entry)
  });
}

/// Counts the image files [`walk`] would visit without reading their metadata, which is much
/// faster. Stops early when `stop` is set.
pub fn count(root: &str, filter: &PathFilter, stop: &AtomicBool) -> u64 {
  let mut count = 0;
  walk_paths(root, filter, |found| {
    if let Found::File { .. } = found {
      count += 1;
    }
    !stop.load(Ordering::Relaxed)
  });
  count
}

enum Found {
  /// An image file; `metadata` is known already when the file was reached through a symlink.
  File {
    path: String,
    metadata: Option<Metadata>,
  },
  Issue(WalkIssue),
}

fn walk_paths(root: &str, filter: &PathFilter, mut visit: impl FnMut(Found) -> bool) {
  let mut stack = vec![root.to_string()];
  let mut visited_links = HashSet::new();
  while let Some(dir) = stack.pop() {
    let entries = match fs::read_dir(&dir) {
      Ok(entries) => entries,
      Err(error) => {
        if !visit(Found::Issue(WalkIssue::UnreadableDir(dir, error))) {
          return;
        }
        continue;
      }
    };
    for entry in entries {
      let entry = match entry {
        Ok(entry) => entry,
        // the rest of the listing is lost, so the directory's images are kept as they are
        Err(error) => {
          if !visit(Found::Issue(WalkIssue::UnreadableDir(dir, error))) {
            return;
          }
          break;
        }
      };
      let file_name = entry.file_name();
      let Some(name) = file_name.to_str() else {
        if !visit(Found::Issue(WalkIssue::NonUtf8(entry.path()))) {
          return;
        }
        continue;
      };
      if filter.is_hidden(name) {
        continue;
      }
      let path = join(&dir, name);
      let file_type = match entry.file_type() {
        Ok(file_type) => file_type,
        Err(error) => {
          if !visit(Found::Issue(WalkIssue::UnreadableDir(path, error))) {
            return;
          }
          continue;
        }
      };
      let (is_dir, metadata) = if file_type.is_symlink() {
        match fs::metadata(&path) {
          Ok(metadata) => (metadata.is_dir(), Some(metadata)),
          // a dangling symlink
          Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
          Err(error) => {
            if !visit(Found::Issue(WalkIssue::UnreadableDir(path, error))) {
              return;
            }
            continue;
          }
        }
      } else {
        (file_type.is_dir(), None)
      };
      if is_dir {
        if filter.is_excluded_dir(relative_path(&path, root)) {
          continue;
        }
        if file_type.is_symlink() {
          let target = match fs::canonicalize(&path) {
            Ok(target) => target,
            Err(error) => {
              if !visit(Found::Issue(WalkIssue::UnreadableDir(path, error))) {
                return;
              }
              continue;
            }
          };
          if !visited_links.insert(target) {
            continue;
          }
        }
        stack.push(path);
      } else if filter.is_image_name(name) && !visit(Found::File { path, metadata }) {
        return;
      }
    }
  }
}

fn join(dir: &str, name: &str) -> String {
  if dir.ends_with(MAIN_SEPARATOR) { format!("{dir}{name}") } else { format!("{dir}{MAIN_SEPARATOR}{name}") }
}

/// The modification time as CPython's `os.stat().st_mtime` computes it, bit for bit, so that an
/// index built by the Python version is recognized as up to date.
#[cfg(unix)]
pub fn modified_at(metadata: &Metadata) -> f64 {
  use std::os::unix::fs::MetadataExt;
  metadata.mtime() as f64 + metadata.mtime_nsec() as f64 * 1e-9
}

#[cfg(windows)]
pub fn modified_at(metadata: &Metadata) -> f64 {
  use std::os::windows::fs::MetadataExt;
  const SECS_BETWEEN_EPOCHS: i64 = 11_644_473_600;
  let filetime = metadata.last_write_time() as i64;
  let (seconds, nanoseconds) = (filetime / 10_000_000 - SECS_BETWEEN_EPOCHS, (filetime % 10_000_000) * 100);
  seconds as f64 + nanoseconds as f64 * 1e-9
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Converts a path written with `/` to the platform's separators.
  fn native(path: &str) -> String {
    path.replace('/', std::path::MAIN_SEPARATOR_STR)
  }

  fn walked(root: &std::path::Path, filter: &PathFilter) -> Vec<String> {
    let root = root.to_str().unwrap();
    let mut paths = Vec::new();
    walk(root, filter, |entry| {
      if let Ok(entry) = entry {
        paths.push(entry.path[root.len() + 1..].replace(MAIN_SEPARATOR, "/"));
      }
      true
    });
    assert_eq!(count(root, filter, &AtomicBool::new(false)), paths.len() as u64);
    paths.sort();
    paths
  }

  fn touch(root: &std::path::Path, relative: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, b"x").unwrap();
  }

  #[test]
  fn matches_image_names_like_the_python_regex() {
    let filter = PathFilter::new(None, false, false);
    assert!(filter.is_image_name("a.jpg"));
    assert!(filter.is_image_name("a.b.JPEG"));
    assert!(!filter.is_image_name(".jpg"));
    assert!(!filter.is_image_name("a.txt"));
    assert!(!filter.is_image_name("a.arw"));
    assert!(PathFilter::new(None, false, true).is_image_name("a.ARW"));
  }

  #[test]
  fn skips_hidden_files_and_excluded_dirs_by_default() {
    let root = tempfile::tempdir().unwrap();
    for path in [
      "a.jpg",
      ".b.jpg",
      ".hidden/c.jpg",
      "node_modules/d.jpg",
      "x/@eaDir/e.jpg",
      "x/f.png",
      "System Volume Information/g.jpg",
      "x/not-image.txt",
    ] {
      touch(root.path(), path);
    }
    assert_eq!(walked(root.path(), &PathFilter::new(None, false, false)), ["a.jpg", "x/f.png"]);
    assert_eq!(
      walked(root.path(), &PathFilter::new(None, true, false)),
      [".b.jpg", ".hidden/c.jpg", "a.jpg", "x/f.png"]
    );
  }

  #[test]
  fn exclusions_are_relative_to_the_root() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().join("node_modules").join("photos");
    touch(&base, "a.jpg");
    touch(&base, "node_modules/b.jpg");
    assert_eq!(walked(&base, &PathFilter::new(None, false, false)), ["a.jpg"]);
  }

  #[test]
  fn excluded_dir_names_must_match_whole_components() {
    let filter = PathFilter::new(Some(vec!["skip".into(), native("a/b")]), false, false);
    assert!(filter.is_excluded_dir("skip"));
    assert!(filter.is_excluded_dir(&native("x/skip/y")));
    assert!(!filter.is_excluded_dir("skipper"));
    assert!(!filter.is_excluded_dir(&native("x/noskip")));
    assert!(filter.is_excluded_dir(&native("x/a/b")));
    assert!(!filter.is_excluded_dir(""));
  }

  #[test]
  fn excludes_indexed_files_the_same_way() {
    let filter = PathFilter::new(None, false, false);
    let is_excluded = |filepath, directory| filter.is_excluded(&native(filepath), &native(directory));
    assert!(is_excluded("/p/.hidden/a.jpg", "/p"));
    assert!(is_excluded("/p/x/.a.jpg", "/p"));
    assert!(is_excluded("/p/x/node_modules/a.jpg", "/p"));
    assert!(!is_excluded("/p/x/a.jpg", "/p"));
    assert!(!is_excluded("/node_modules/p/a.jpg", "/node_modules/p"));
  }

  #[cfg(unix)]
  #[test]
  fn survives_symlink_cycles() {
    let root = tempfile::tempdir().unwrap();
    touch(root.path(), "a/b.jpg");
    std::os::unix::fs::symlink(root.path().join("a"), root.path().join("a/loop")).unwrap();
    let paths = walked(root.path(), &PathFilter::new(None, false, false));
    assert!(paths.len() <= 2, "{paths:?}");
    assert!(paths.contains(&"a/b.jpg".to_string()));
  }

  #[cfg(unix)]
  #[test]
  fn reports_symlinks_to_inaccessible_targets_and_skips_dangling_ones() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let (root, outside) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let locked = outside.path().join("locked");
    touch(&locked, "a.jpg");
    symlink(locked.join("a.jpg"), root.path().join("inaccessible.jpg")).unwrap();
    symlink(outside.path().join("missing.jpg"), root.path().join("dangling.jpg")).unwrap();
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    let mut unreadable = Vec::new();
    walk(root.path().to_str().unwrap(), &PathFilter::new(None, false, false), |entry| {
      if let Err(WalkIssue::UnreadableDir(path, _)) = entry {
        unreadable.push(path);
      }
      true
    });
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(unreadable, [root.path().join("inaccessible.jpg").to_str().unwrap()]);
  }

  #[cfg(unix)]
  #[test]
  fn modification_time_matches_python() {
    let root = tempfile::tempdir().unwrap();
    touch(root.path(), "a.jpg");
    let metadata = fs::metadata(root.path().join("a.jpg")).unwrap();
    let modified_at = super::modified_at(&metadata);
    assert!(modified_at > 1_000_000_000.0);
  }
}
