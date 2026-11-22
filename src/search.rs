//! Ranking indexed images against a query and listing them for browsing.

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow};

use crate::db::{Db, ImageCursor};
use crate::model::{FeatureVector, Model, QueryError, VECTOR_SIZE, is_file_path};
use crate::walk::PathFilter;

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
  pub filepath: String,
  pub score: f64,
}

/// Results rank by descending score, then by path.
impl Ord for SearchResult {
  fn cmp(&self, other: &Self) -> CmpOrdering {
    other.score.total_cmp(&self.score).then_with(|| self.filepath.cmp(&other.filepath))
  }
}

impl PartialOrd for SearchResult {
  fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
    Some(self.cmp(other))
  }
}

impl Eq for SearchResult {}

/// Raised when a search is cancelled through its cancellation flag.
#[derive(Debug, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

#[derive(Debug, thiserror::Error)]
pub enum SearchError {
  #[error(transparent)]
  Query(#[from] QueryError),
  #[error(transparent)]
  Cancelled(#[from] Cancelled),
  #[error(transparent)]
  Other(#[from] anyhow::Error),
}

pub struct Search<'a> {
  pub db: &'a Db,
  pub model: &'a Model,
  pub filter: &'a PathFilter,
  pub directory: &'a str,
}

pub struct Query<'a> {
  pub query: &'a str,
  pub add: &'a [String],
  pub subtract: &'a [String],
}

fn check(cancel: Option<&AtomicBool>) -> Result<(), Cancelled> {
  match cancel {
    Some(flag) if flag.load(Ordering::Relaxed) => Err(Cancelled),
    _ => Ok(()),
  }
}

impl Search<'_> {
  /// Returns the `top_k` best matches, best first.
  pub fn search(
    &self,
    query: &Query<'_>,
    top_k: usize,
    cancel: Option<&AtomicBool>,
  ) -> Result<Vec<SearchResult>, SearchError> {
    let positive: Vec<String> = std::iter::once(query.query.to_string()).chain(query.add.iter().cloned()).collect();
    check(cancel)?;
    let positive_features = self.model.query_features(&positive)?;
    check(cancel)?;
    let negative_features = self.model.query_features(query.subtract)?;
    check(cancel)?;
    let features: FeatureVector = positive_features.iter().zip(&negative_features).map(|(p, n)| p - n).collect();

    // Images used as queries are excluded from the results.
    let cwd = std::env::current_dir().map_err(|error| anyhow!("cannot get the current directory: {error}"))?;
    let excluded: HashSet<String> = positive
      .iter()
      .chain(query.subtract)
      .filter(|query| is_file_path(query))
      .map(|query| absolute_path(&cwd, query).to_string_lossy().into_owned())
      .collect();

    let rankings = self
      .db
      .scan_vectors(
        self.directory,
        || (Ranking::new(top_k), 0usize),
        |(ranking, rows), filepath, vector| {
          *rows += 1;
          if rows.is_multiple_of(4096) {
            check(cancel)?;
          }
          if !self.filter.is_excluded(filepath, self.directory) && !excluded.contains(filepath) {
            ranking.push(dot(&features, vector), filepath);
          }
          Ok(())
        },
      )
      .map_err(|error| match error.downcast::<Cancelled>() {
        Ok(cancelled) => SearchError::Cancelled(cancelled),
        Err(error) => SearchError::Other(error),
      })?;
    let mut ranked = Ranking::new(top_k);
    for (ranking, _) in rankings {
      for result in ranking.into_sorted() {
        ranked.push(result.score, &result.filepath);
      }
    }
    check(cancel)?;
    Ok(ranked.into_sorted())
  }

  /// Lists up to `limit` images, most recently modified first, continuing after `after`. The
  /// returned cursor is set when more images follow.
  pub fn list_images(
    &self,
    limit: usize,
    after: Option<&ImageCursor>,
    cancel: Option<&AtomicBool>,
  ) -> Result<(Vec<String>, Option<ImageCursor>), SearchError> {
    let mut results = Vec::new();
    let mut last: Option<ImageCursor> = None;
    let mut has_more = false;
    self
      .db
      .for_each_filepath(self.directory, after, |filepath, modified_at| {
        check(cancel)?;
        if self.filter.is_excluded(filepath, self.directory) {
          return Ok(true);
        }
        if results.len() == limit {
          has_more = true;
          return Ok(false);
        }
        results.push(filepath.to_string());
        last = Some(ImageCursor { modified_at, filepath: filepath.to_string() });
        Ok(true)
      })
      .map_err(|error| match error.downcast::<Cancelled>() {
        Ok(cancelled) => SearchError::Cancelled(cancelled),
        Err(error) => SearchError::Other(error),
      })?;
    Ok((results, if has_more { last } else { None }))
  }
}

/// Keeps the best `limit` results seen so far.
struct Ranking {
  limit: usize,
  // A max-heap under `SearchResult`'s ordering keeps the worst retained result on top.
  heap: BinaryHeap<SearchResult>,
}

impl Ranking {
  fn new(limit: usize) -> Self {
    Ranking { limit, heap: BinaryHeap::new() }
  }

  fn push(&mut self, score: f64, filepath: &str) {
    if self.limit == 0 {
      return;
    }
    if self.heap.len() == self.limit {
      let worst = self.heap.peek().expect("the heap is full");
      let better = score > worst.score || (score == worst.score && filepath < worst.filepath.as_str());
      if !better {
        return;
      }
      self.heap.pop();
    }
    self.heap.push(SearchResult { filepath: filepath.to_string(), score });
  }

  fn into_sorted(self) -> Vec<SearchResult> {
    self.heap.into_sorted_vec()
  }
}

/// Dot product of a double-precision query with a stored little-endian float32 vector.
fn dot(query: &[f64], vector: &[u8]) -> f64 {
  debug_assert_eq!(vector.len(), VECTOR_SIZE * 4);
  // Eight independent accumulators let the compiler keep the multiply-adds in SIMD registers.
  let mut sums = [0.0f64; 8];
  for (query, values) in query.as_chunks::<8>().0.iter().zip(vector.as_chunks::<32>().0) {
    for lane in 0..8 {
      let value = f32::from_le_bytes(values[lane * 4..lane * 4 + 4].try_into().expect("four bytes"));
      sums[lane] += query[lane] * f64::from(value);
    }
  }
  sums.iter().sum()
}

/// `os.path.abspath`: joins `path` to `cwd` and normalizes `.` and `..` lexically.
pub fn absolute_path(cwd: &Path, path: &str) -> PathBuf {
  let joined = cwd.join(path);
  let mut normalized = PathBuf::new();
  for component in joined.components() {
    match component {
      Component::CurDir => {}
      Component::ParentDir => {
        normalized.pop();
      }
      component => normalized.push(component),
    }
  }
  normalized
}

#[cfg(test)]
mod tests {
  use super::*;

  fn vector(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
  }

  #[test]
  fn dot_product_matches_a_naive_sum() {
    let query: Vec<f64> = (0..VECTOR_SIZE).map(|i| (i as f64).sin()).collect();
    let values: Vec<f32> = (0..VECTOR_SIZE).map(|i| (i as f32 * 0.37).cos()).collect();
    let naive: f64 = query.iter().zip(&values).map(|(q, v)| q * f64::from(*v)).sum();
    assert!((dot(&query, &vector(values)) - naive).abs() < 1e-9);
  }

  #[test]
  fn ranking_keeps_the_global_top_results_with_path_tiebreaks() {
    let mut ranking = Ranking::new(3);
    for (score, path) in [(0.1, "e"), (0.9, "b"), (0.5, "c"), (0.9, "a"), (0.2, "d"), (0.5, "a2")] {
      ranking.push(score, path);
    }
    let results: Vec<_> = ranking.into_sorted().into_iter().map(|result| result.filepath).collect();
    assert_eq!(results, ["a", "b", "a2"]);
  }

  #[cfg(unix)]
  #[test]
  fn absolute_path_normalizes_like_python() {
    let cwd = Path::new("/home/user/photos");
    assert_eq!(absolute_path(cwd, "./cat.jpg"), Path::new("/home/user/photos/cat.jpg"));
    assert_eq!(absolute_path(cwd, "../cat.jpg"), Path::new("/home/user/cat.jpg"));
    assert_eq!(absolute_path(cwd, "/tmp//a/./b.jpg"), Path::new("/tmp/a/b.jpg"));
  }

  #[test]
  fn lists_images_with_exclusions_applied_before_the_limit() {
    let db = Db::open_in_memory().unwrap();
    // `list_images` splits paths at the platform's separator
    let native = |path: &str| path.replace('/', std::path::MAIN_SEPARATOR_STR);
    for (path, modified_at) in [("/p/a.jpg", 5.0), ("/p/.hidden.jpg", 4.0), ("/p/b.jpg", 3.0), ("/p/c.jpg", 2.0)] {
      let image = crate::db::NewImage { filepath: &native(path), modified_at, size: 1, vector: &[0; 4], hash: "h" };
      db.upsert_image(&image).unwrap();
    }
    let model = Model::new(crate::model::download::ModelFiles::new(Path::new("/nonexistent")));
    let filter = PathFilter::new(None, false, false);
    let search = Search { db: &db, model: &model, filter: &filter, directory: &native("/p") };
    let (page, cursor) = search.list_images(2, None, None).unwrap();
    assert_eq!(page, [native("/p/a.jpg"), native("/p/b.jpg")]);
    let (page, cursor) = search.list_images(2, cursor.as_ref(), None).unwrap();
    assert_eq!(page, [native("/p/c.jpg")]);
    assert!(cursor.is_none());
  }
}
