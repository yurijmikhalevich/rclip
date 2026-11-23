//! Keeps the index in sync with the images on disk.
//!
//! The work is pipelined so that every stage stays busy:
//!
//! ```text
//! walker thread ──files──▶ coordinator ──jobs──▶ decode workers (hash, decode, preprocess)
//!                          (owns the DB) ◀─images──┘
//!                               │  ▲
//!                         batches  embeddings
//!                               ▼  │
//!                          inference thread (Core ML / ONNX Runtime)
//! ```
//!
//! The coordinator is the only thread touching the database, so all writes share one
//! transaction that is committed periodically.

use std::collections::HashSet;
use std::path::MAIN_SEPARATOR;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded, select, unbounded};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use sha2::{Digest, Sha256};

use crate::db::{Db, NewImage};
use crate::imaging::{self, IMAGE_EXT, IMAGE_RAW_EXT, ImageError, preprocess};
use crate::model::{IndexEncoder, VECTOR_SIZE};
use crate::walk::{self, FileEntry, PathFilter, WalkIssue};

const COMMIT_EVERY: u64 = 50_000;
/// How many indexing batches to keep loading ahead of the model. Preprocessed images are small
/// (768 KiB each), so a few batches of look-ahead cost little memory.
const LOOKAHEAD_BATCHES: usize = 3;
pub const MAX_LOADING_WORKERS: usize = 16;

pub struct IndexOptions {
  pub batch_size: usize,
  pub raw_support: bool,
  pub max_pixels: Option<u64>,
  pub workers: usize,
}

pub fn loading_workers() -> usize {
  std::thread::available_parallelism().map_or(1, usize::from).clamp(1, MAX_LOADING_WORKERS)
}

/// A file that needs (re)indexing, with its decoded and preprocessed pixels.
struct LoadedImage {
  entry: FileEntry,
  hash: String,
  tensor: Vec<f32>,
}

struct Batch {
  images: Vec<(FileEntry, String)>,
  tensors: Vec<f32>,
}

/// Brings the index for `directory` up to date: indexes new and changed images, reuses vectors of
/// renamed or duplicated files, and marks images that disappeared as deleted.
pub fn ensure_index(
  db: &Db,
  encoder: impl FnOnce() -> Result<IndexEncoder>,
  filter: &PathFilter,
  directory: &str,
  options: &IndexOptions,
) -> Result<()> {
  eprintln!(
    "checking images in the current directory for changes; \
     use \"--no-indexing\" to skip this if no images were added, changed, or removed"
  );
  let progress = progress_bar();
  let mut indexer =
    Indexer { db, options, progress: &progress, seen: HashSet::new(), unreadable_dirs: Vec::new(), scanned: 0 };
  let result = indexer.run(encoder, filter, directory);
  progress.finish();
  result?;
  indexer.mark_missing_images_as_deleted(filter, directory)?;
  eprintln!();
  Ok(())
}

struct Indexer<'a> {
  db: &'a Db,
  options: &'a IndexOptions,
  progress: &'a ProgressBar,
  /// Paths whose index entries are confirmed up to date in this run.
  seen: HashSet<String>,
  /// Directories that could not be listed and entries that could not be inspected; their images are left as
  /// they are.
  unreadable_dirs: Vec<String>,
  scanned: u64,
}

impl Indexer<'_> {
  fn run(
    &mut self,
    encoder: impl FnOnce() -> Result<IndexEncoder>,
    filter: &PathFilter,
    directory: &str,
  ) -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let (entries_tx, entries_rx) = bounded::<Result<FileEntry, WalkIssue>>(4096);
    let walker = {
      let (filter, directory, stop) = (filter.clone(), directory.to_string(), Arc::clone(&stop));
      std::thread::spawn(move || {
        walk::walk(&directory, &filter, |entry| !stop.load(Ordering::Relaxed) && entries_tx.send(entry).is_ok());
      })
    };
    // Counting without reading file metadata is much faster than the scan itself, so the progress
    // bar learns its total early.
    let counter = {
      let (filter, directory, stop, progress) =
        (filter.clone(), directory.to_string(), Arc::clone(&stop), self.progress.clone());
      std::thread::spawn(move || {
        let total = walk::count(&directory, &filter, &stop);
        if !stop.load(Ordering::Relaxed) {
          progress.set_length(total);
          progress.set_style(progress_style(true));
        }
      })
    };

    let workers = self.options.workers;
    let (jobs_tx, jobs_rx) = unbounded::<FileEntry>();
    let (loaded_tx, loaded_rx) = unbounded::<(FileEntry, Result<LoadedImage, ImageError>)>();
    let loaders: Vec<_> = (0..workers)
      .map(|_| {
        let (jobs_rx, loaded_tx, max_pixels) = (jobs_rx.clone(), loaded_tx.clone(), self.options.max_pixels);
        std::thread::spawn(move || {
          for entry in jobs_rx {
            let result = load_image(&entry, max_pixels);
            if loaded_tx.send((entry, result)).is_err() {
              break;
            }
          }
        })
      })
      .collect();
    drop((jobs_rx, loaded_tx));

    let result = self.coordinate(encoder, &entries_rx, &jobs_tx, &loaded_rx);

    stop.store(true, Ordering::Relaxed);
    drop((entries_rx, jobs_tx, loaded_rx));
    for thread in loaders.into_iter().chain([walker, counter]) {
      thread.join().ok();
    }
    self.db.commit()?;
    result
  }

  fn coordinate(
    &mut self,
    encoder: impl FnOnce() -> Result<IndexEncoder>,
    entries: &Receiver<Result<FileEntry, WalkIssue>>,
    jobs: &Sender<FileEntry>,
    loaded: &Receiver<(FileEntry, Result<LoadedImage, ImageError>)>,
  ) -> Result<()> {
    // The encoder is loaded lazily: an up-to-date index needs no model at all.
    let mut encoder = Some(encoder);
    let mut inference: Option<Inference> = None;
    let batch_size = self.options.batch_size;
    // Decoded images waiting for the model count against the look-ahead too, so memory stays
    // bounded when the model is slower than decoding.
    let max_pending = (LOOKAHEAD_BATCHES * batch_size).max(self.options.workers * 2);
    let mut in_flight = 0usize;
    let mut walking = true;
    let mut batch = Batch { images: Vec::new(), tensors: Vec::new() };

    loop {
      let busy = inference.as_ref().map_or(0, |inference| inference.busy);
      let ready = batch.images.len() >= batch_size || (!walking && in_flight == 0 && !batch.images.is_empty());
      // Keep one batch running and one queued.
      if ready && busy < 2 {
        let inference = match &mut inference {
          Some(inference) => inference,
          None => inference.insert(Inference::spawn(encoder.take().expect("the encoder is loaded once")()?)),
        };
        let count = batch.images.len().min(batch_size);
        let images = batch.images.drain(..count).collect();
        let tensors = batch.tensors.drain(..count * preprocess::TENSOR_LEN).collect();
        inference.send(Batch { images, tensors });
        continue;
      }
      if !walking && in_flight == 0 && batch.images.is_empty() && busy == 0 {
        return Ok(());
      }

      let (no_entries, no_embeddings) = (crossbeam_channel::never(), crossbeam_channel::never());
      let accepting = walking && in_flight + batch.images.len() < max_pending;
      let embeddings = inference.as_ref().map_or(&no_embeddings, |inference| &inference.embeddings_rx);
      select! {
        recv(if accepting { entries } else { &no_entries }) -> message => match message {
          Ok(Ok(entry)) => self.scan(entry, jobs, &mut in_flight)?,
          Ok(Err(issue)) => self.report(issue),
          Err(_) => walking = false,
        },
        recv(loaded) -> message => {
          let (entry, image) = message.expect("loaders outlive in-flight jobs");
          in_flight -= 1;
          match image {
            Ok(image) => self.add_loaded(image, &mut batch)?,
            Err(error) => self.warn(skip_message(&entry.path, &error)),
          }
        },
        recv(embeddings) -> message => {
          let (images, embeddings) = message.expect("the inference thread outlives busy batches");
          inference.as_mut().expect("embeddings come from a running inference").busy -= 1;
          match embeddings {
            Ok(embeddings) => self.store(&images, &embeddings)?,
            Err(error) => self.warn(format!("error computing features: {error:#}")),
          }
        },
      }
    }
  }

  /// Handles one file found by the walker: unchanged files are confirmed, the rest are queued.
  fn scan(&mut self, entry: FileEntry, jobs: &Sender<FileEntry>, in_flight: &mut usize) -> Result<()> {
    self.progress.inc(1);
    if self.options.raw_support
      && IMAGE_RAW_EXT.contains(&imaging::file_extension(&entry.path).as_str())
      && has_processed_sibling(&entry.path)
    {
      return Ok(());
    }
    if self.scanned.is_multiple_of(COMMIT_EVERY) {
      self.db.commit()?;
    }
    self.scanned += 1;
    if let Some(state) = self.db.image_state(&entry.path)?
      && state.modified_at == entry.modified_at
      && state.size == entry.size
    {
      if state.deleted {
        self.db.restore_image(&entry.path)?;
      }
      self.seen.insert(entry.path);
      return Ok(());
    }
    jobs.send(entry).expect("loaders outlive the scan");
    *in_flight += 1;
    Ok(())
  }

  fn add_loaded(&mut self, image: LoadedImage, batch: &mut Batch) -> Result<()> {
    if let Some(vector) = self.db.vector_by_hash(&image.hash, image.entry.size)? {
      // A renamed or duplicated file: reuse the vector instead of recomputing it.
      self.upsert(&image.entry, &image.hash, &vector)?;
      return Ok(());
    }
    batch.tensors.extend_from_slice(&image.tensor);
    batch.images.push((image.entry, image.hash));
    Ok(())
  }

  fn store(&mut self, images: &[(FileEntry, String)], embeddings: &[f32]) -> Result<()> {
    for ((entry, hash), embedding) in images.iter().zip(embeddings.as_chunks::<VECTOR_SIZE>().0) {
      let vector: Vec<u8> = embedding.iter().flat_map(|value| value.to_le_bytes()).collect();
      self.upsert(entry, hash, &vector)?;
    }
    Ok(())
  }

  fn upsert(&mut self, entry: &FileEntry, hash: &str, vector: &[u8]) -> Result<()> {
    self.db.upsert_image(&NewImage {
      filepath: &entry.path,
      modified_at: entry.modified_at,
      size: entry.size,
      vector,
      hash,
    })?;
    self.seen.insert(entry.path.clone());
    Ok(())
  }

  fn report(&mut self, issue: WalkIssue) {
    match issue {
      WalkIssue::UnreadableDir(dir, error) => {
        // The confined snap cannot follow symlinks out of the directories it was granted.
        if crate::snap::is_snap()
          && error.kind() == std::io::ErrorKind::PermissionDenied
          && std::path::Path::new(&dir).is_symlink()
        {
          self.progress.finish_and_clear();
          eprintln!();
          crate::snap::exit_with_permission_error(&dir, false);
        }
        self.warn(format!("cannot read {dir}: {error}; keeping its images in the index as they are"));
        self.unreadable_dirs.push(dir);
      }
      WalkIssue::Metadata(path, error) => {
        self.warn(format!("error getting fs metadata for {path}: {error}"));
        self.unreadable_dirs.push(path);
      }
      WalkIssue::NonUtf8(path) => self.warn(format!("skipping {}: the path is not valid UTF-8", path.display())),
    }
  }

  fn warn(&self, message: String) {
    self.progress.suspend(|| eprintln!("{message}"));
  }

  /// Marks indexed images under `directory` that were not found during this run as deleted.
  /// Excluded paths and unreadable directories and entries were not inspected, so their state is kept.
  fn mark_missing_images_as_deleted(&self, filter: &PathFilter, directory: &str) -> Result<()> {
    let unreadable: Vec<String> = self
      .unreadable_dirs
      .iter()
      .map(|dir| format!("{}{MAIN_SEPARATOR}", dir.trim_end_matches(MAIN_SEPARATOR)))
      .collect();
    for filepath in self.db.filepaths(directory)? {
      if !self.seen.contains(&filepath)
        && !filter.is_excluded(&filepath, directory)
        // an unreadable entry may be a symlink to a file, too
        && !self.unreadable_dirs.contains(&filepath)
        && !unreadable.iter().any(|dir| filepath.starts_with(dir.as_str()))
      {
        self.db.mark_deleted(&filepath)?;
      }
    }
    self.db.commit()
  }
}

/// The images of a batch with their embeddings, or the error that prevented computing them.
type Embedded = (Vec<(FileEntry, String)>, Result<Vec<f32>>);

/// Runs the encoder on its own thread so that decoding continues while the model works.
struct Inference {
  batches_tx: Option<Sender<Batch>>,
  embeddings_rx: Receiver<Embedded>,
  thread: Option<std::thread::JoinHandle<()>>,
  busy: usize,
}

impl Inference {
  fn spawn(encoder: IndexEncoder) -> Self {
    let (batches_tx, batches_rx) = unbounded::<Batch>();
    let (embeddings_tx, embeddings_rx) = unbounded();
    let thread = std::thread::spawn(move || {
      for batch in batches_rx {
        let embeddings = encoder.encode(&batch.tensors);
        if embeddings_tx.send((batch.images, embeddings)).is_err() {
          break;
        }
      }
    });
    Inference { batches_tx: Some(batches_tx), embeddings_rx, thread: Some(thread), busy: 0 }
  }

  fn send(&mut self, batch: Batch) {
    self.busy += 1;
    self.batches_tx.as_ref().expect("the inference thread is running").send(batch).ok();
  }
}

impl Drop for Inference {
  fn drop(&mut self) {
    self.batches_tx = None;
    if let Some(thread) = self.thread.take() {
      thread.join().ok();
    }
  }
}

fn load_image(entry: &FileEntry, max_pixels: Option<u64>) -> Result<LoadedImage, ImageError> {
  let data = std::fs::read(&entry.path)?;
  let hash = file_hash(&data);
  let extension = imaging::file_extension(&entry.path);
  let image = imaging::decode(&data, &extension, max_pixels, Some(preprocess::IMAGE_SIZE))?;
  drop(data);
  Ok(LoadedImage { entry: entry.clone(), hash, tensor: preprocess::preprocess(&image) })
}

/// SHA-256 of the first MiB and the last KiB of a file, used to recognize renamed and duplicated
/// files without hashing them in full.
pub fn file_hash(data: &[u8]) -> String {
  let mut hasher = Sha256::new();
  hasher.update(&data[..data.len().min(1024 * 1024)]);
  if data.len() > 1024 {
    hasher.update(&data[data.len() - 1024..]);
  }
  hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Whether a processed image (e.g. a JPEG) with the same name sits next to a RAW file. Like the
/// Python version, this checks the lowercase and uppercase extensions but not mixed-case ones.
fn has_processed_sibling(raw_path: &str) -> bool {
  let stem = match raw_path.rsplit_once('.') {
    Some((stem, _)) => stem,
    None => raw_path,
  };
  IMAGE_EXT.iter().any(|extension| {
    std::path::Path::new(&format!("{stem}.{extension}")).is_file()
      || std::path::Path::new(&format!("{stem}.{}", extension.to_ascii_uppercase())).is_file()
  })
}

fn skip_message(path: &str, error: &ImageError) -> String {
  match error {
    ImageError::TooLarge { pixels, limit } => too_large_message(path, *pixels, *limit),
    ImageError::Unidentified => format!("skipping {path}: it is not a readable image"),
    ImageError::Io(error) if error.kind() == std::io::ErrorKind::OutOfMemory => {
      format!("skipping {path}: ran out of memory while processing it")
    }
    error => format!("skipping {path}: {error}"),
  }
}

fn too_large_message(path: &str, pixels: u64, limit: u64) -> String {
  let size = if pixels > 0 { format!(" ({:.0} MP)", pixels as f64 / 1_000_000.0) } else { String::new() };
  format!(
    "skipping {path}: it is too large to process{size}; the limit is {:.0} MP. \
    Raise or disable it with \"--max-image-megapixels\" if you want to index this image",
    limit as f64 / 1_000_000.0
  )
}

fn progress_style(known_total: bool) -> ProgressStyle {
  let template = if known_total {
    "{percent:>3}%|{wide_bar}| {pos}/{len} [{elapsed}<{eta}, {per_sec}]"
  } else {
    "{pos} images [{elapsed}, {per_sec}]"
  };
  ProgressStyle::with_template(template).expect("the progress template is valid")
}

fn progress_bar() -> ProgressBar {
  let progress = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr());
  progress.set_style(progress_style(false));
  progress
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn hashes_the_head_and_tail_of_files() {
    let small = file_hash(b"abc");
    assert_eq!(small, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    let large: Vec<u8> = (0..3 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let mut expected = Sha256::new();
    expected.update(&large[..1024 * 1024]);
    expected.update(&large[large.len() - 1024..]);
    let expected: String = expected.finalize().iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(file_hash(&large), expected);
  }

  #[test]
  fn formats_too_large_messages_like_python() {
    assert_eq!(
      too_large_message("/a.jpg", 120_000_000, 89_478_485),
      "skipping /a.jpg: it is too large to process (120 MP); the limit is 89 MP. \
      Raise or disable it with \"--max-image-megapixels\" if you want to index this image"
    );
  }

  #[test]
  fn detects_processed_siblings_of_raw_files() {
    let directory = tempfile::tempdir().unwrap();
    let raw = directory.path().join("a.ARW");
    std::fs::write(&raw, b"x").unwrap();
    assert!(!has_processed_sibling(raw.to_str().unwrap()));
    std::fs::write(directory.path().join("a.JPG"), b"x").unwrap();
    assert!(has_processed_sibling(raw.to_str().unwrap()));
  }

  #[cfg(unix)]
  #[test]
  fn keeps_images_whose_metadata_cannot_be_read() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let image = directory.path().join("a.jpg");
    std::fs::write(&image, b"x").unwrap();
    let image = image.to_str().unwrap();
    let db = Db::open_in_memory().unwrap();
    db.upsert_image(&NewImage { filepath: image, modified_at: 1.0, size: 1, vector: &[0; 4], hash: "h" }).unwrap();
    // A directory that can be listed but not searched hides its files' metadata.
    std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o444)).unwrap();
    let options = IndexOptions { batch_size: 1, raw_support: false, max_pixels: None, workers: 1 };
    let filter = PathFilter::new(None, false, false);
    let result = ensure_index(&db, || anyhow::bail!("no model"), &filter, directory.path().to_str().unwrap(), &options);
    std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    result.unwrap();
    assert!(!db.image_state(image).unwrap().unwrap().deleted);
  }
}
