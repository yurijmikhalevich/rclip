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
use crate::model::VECTOR_SIZE;
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

/// Computes the normalized embeddings of preprocessed images stored back to back.
pub type Encoder = Box<dyn Fn(&[f32]) -> Result<Vec<f32>> + Send>;

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
  encoder: impl FnOnce() -> Result<Encoder>,
  filter: &PathFilter,
  directory: &str,
  options: &IndexOptions,
) -> Result<()> {
  eprintln!(
    "checking images in the current directory for changes; \
     use \"--no-indexing\" to skip this if no images were added, changed, or removed"
  );
  // Read before the walk, so that an image another rclip run indexes meanwhile is not taken for a missing one.
  let indexed = db.filepaths(directory)?;
  let progress = progress_bar();
  let mut indexer =
    Indexer { db, options, progress: &progress, seen: HashSet::new(), unreadable_dirs: Vec::new(), scanned: 0 };
  let result = indexer.run(encoder, filter, directory);
  progress.finish();
  result?;
  indexer.mark_missing_images_as_deleted(indexed, filter, directory)?;
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
  fn run(&mut self, encoder: impl FnOnce() -> Result<Encoder>, filter: &PathFilter, directory: &str) -> Result<()> {
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
            let result = imaging::catch_panic(|| load_image(&entry, max_pixels))
              .unwrap_or_else(|message| Err(ImageError::Decode(message)));
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
    encoder: impl FnOnce() -> Result<Encoder>,
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
    // Saturating, so that a huge batch size cannot wrap around and stall the walk before a batch fills.
    let max_pending = LOOKAHEAD_BATCHES.saturating_mul(batch_size).max(self.options.workers * 2);
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

  /// Marks the `indexed` images under `directory` that were not found during this run as deleted.
  /// Excluded paths and unreadable directories and entries were not inspected, so their state is kept.
  fn mark_missing_images_as_deleted(&self, indexed: Vec<String>, filter: &PathFilter, directory: &str) -> Result<()> {
    let unreadable: Vec<String> = self
      .unreadable_dirs
      .iter()
      .map(|dir| format!("{}{MAIN_SEPARATOR}", dir.trim_end_matches(MAIN_SEPARATOR)))
      .collect();
    for filepath in indexed {
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
  fn spawn(encoder: Encoder) -> Self {
    let (batches_tx, batches_rx) = unbounded::<Batch>();
    let (embeddings_tx, embeddings_rx) = unbounded();
    let thread = std::thread::spawn(move || {
      for batch in batches_rx {
        let embeddings = encoder(&batch.tensors);
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
  // The progress of the many indexing runs in the tests would clutter the test output.
  let target = if cfg!(test) { ProgressDrawTarget::hidden() } else { ProgressDrawTarget::stderr() };
  let progress = ProgressBar::with_draw_target(None, target);
  progress.set_style(progress_style(false));
  progress
}

#[cfg(test)]
mod tests {
  use std::cell::Cell;
  use std::path::{MAIN_SEPARATOR_STR, Path};
  use std::sync::Mutex;
  use std::time::{Duration, SystemTime};

  use super::*;
  use crate::db::{ImageState, VectorCacheReset};
  use crate::model::Model;
  use crate::model::download::ModelFiles;
  use crate::search::Search;

  /// A directory of images and an index that a fake model keeps in sync with it. The model embeds an image as
  /// the value of its first pixel, so images of different shades get different vectors.
  struct Library {
    /// Declared before the directory to be closed before the directory is removed.
    db: Db,
    directory: tempfile::TempDir,
    /// The number of images in every batch the model has encoded.
    batches: Arc<Mutex<Vec<usize>>>,
    /// How long the model takes to encode a batch.
    encoding_time: Cell<Duration>,
    /// The modification time of the last written file, in seconds. Every write gets a later one, as files
    /// written in quick succession can otherwise share a modification time and a size.
    clock: Cell<u64>,
  }

  impl Library {
    fn new() -> Self {
      let directory = tempfile::tempdir().unwrap();
      std::fs::create_dir(directory.path().join("photos")).unwrap();
      let db = Db::open(&directory.path().join("db.sqlite3"), VectorCacheReset::Allow).unwrap();
      Library {
        db,
        directory,
        batches: Arc::default(),
        encoding_time: Cell::default(),
        clock: Cell::new(1_000_000_000),
      }
    }

    /// The indexed directory.
    fn root(&self) -> String {
      self.directory.path().join("photos").into_os_string().into_string().unwrap()
    }

    /// The path of a file given relative to the indexed directory with `/` separators.
    fn path(&self, name: &str) -> String {
      format!("{}{MAIN_SEPARATOR}{}", self.root(), name.replace('/', MAIN_SEPARATOR_STR))
    }

    fn write(&self, name: &str, data: &[u8]) {
      let path = self.path(name);
      std::fs::create_dir_all(Path::new(&path).parent().unwrap()).unwrap();
      std::fs::write(&path, data).unwrap();
      self.clock.set(self.clock.get() + 10);
      set_modified(&path, self.clock.get());
    }

    fn index(&self) {
      self.index_with(&PathFilter::new(None, false, false), &self.root(), None);
    }

    fn index_with(&self, filter: &PathFilter, directory: &str, max_pixels: Option<u64>) {
      let (batches, encoding_time) = (Arc::clone(&self.batches), self.encoding_time.get());
      let encoder = || -> Result<Encoder> {
        Ok(Box::new(move |tensors: &[f32]| {
          std::thread::sleep(encoding_time);
          let embeddings: Vec<f32> =
            tensors.chunks(preprocess::TENSOR_LEN).flat_map(|tensor| [tensor[0]; VECTOR_SIZE]).collect();
          batches.lock().unwrap().push(embeddings.len() / VECTOR_SIZE);
          Ok(embeddings)
        }))
      };
      let options = IndexOptions { batch_size: 2, raw_support: false, max_pixels, workers: 2 };
      ensure_index(&self.db, encoder, filter, directory, &options).unwrap();
    }

    /// How many images the model has encoded.
    fn encoded(&self) -> usize {
      self.batches.lock().unwrap().iter().sum()
    }

    /// The images in the index that are not marked as deleted, relative to the indexed directory.
    fn images(&self) -> Vec<String> {
      let root = self.root();
      let mut images: Vec<String> =
        self.db.filepaths(&root).unwrap().iter().map(|path| relative(path, &root)).collect();
      images.sort();
      images
    }

    /// The images that browsing `directory` with `filter` shows, fetched one per page, relative to `directory`.
    fn browse(&self, filter: &PathFilter, directory: &str) -> Vec<String> {
      let model = Model::new(ModelFiles::new(Path::new("/nonexistent")));
      let search = Search { db: &self.db, model: &model, filter, directory };
      let (mut images, mut cursor) = (Vec::new(), None);
      loop {
        let (page, next) = search.list_images(1, cursor.as_ref(), None).unwrap();
        images.extend(page.iter().map(|path| relative(path, directory)));
        cursor = next;
        if cursor.is_none() {
          break;
        }
      }
      images.sort();
      images
    }

    fn state(&self, name: &str) -> ImageState {
      self.db.image_state(&self.path(name)).unwrap().unwrap()
    }

    /// The first value of the vector stored for an image.
    fn vector(&self, name: &str) -> f32 {
      let path = self.path(name);
      let visit = |vectors: &mut Vec<f32>, filepath: &str, vector: &[u8]| {
        if filepath == path {
          vectors.push(f32::from_le_bytes(vector[..4].try_into().unwrap()));
        }
        Ok(())
      };
      self.db.scan_vectors(&self.root(), Vec::new, visit).unwrap().concat()[0]
    }
  }

  /// A path relative to `directory`, with `/` separators.
  fn relative(path: &str, directory: &str) -> String {
    path[directory.len() + 1..].replace(MAIN_SEPARATOR, "/")
  }

  fn set_modified(path: &str, seconds: u64) {
    let time = SystemTime::UNIX_EPOCH + Duration::from_secs(seconds);
    std::fs::File::options().write(true).open(path).unwrap().set_modified(time).unwrap();
  }

  /// An encoded square image of one shade of gray.
  fn png(side: u32, shade: u8) -> Vec<u8> {
    let mut data = std::io::Cursor::new(Vec::new());
    let image = image::RgbImage::from_pixel(side, side, image::Rgb([shade; 3]));
    image.write_to(&mut data, image::ImageFormat::Png).unwrap();
    data.into_inner()
  }

  #[test]
  fn indexes_new_and_changed_images_and_skips_unchanged_ones() {
    let library = Library::new();
    library.write("a.png", &png(2, 10));
    library.index();
    assert_eq!((library.images(), library.encoded()), (vec!["a.png".to_string()], 1));
    let vector = library.vector("a.png");
    library.index();
    assert_eq!(library.encoded(), 1);
    library.write("a.png", &png(2, 200));
    library.index();
    assert_eq!(library.encoded(), 2);
    assert!(library.vector("a.png") > vector);
  }

  #[test]
  fn leaves_an_up_to_date_index_untouched() {
    let library = Library::new();
    library.write("a.png", &png(2, 10));
    library.write("b.png", &png(2, 20));
    library.index();
    // the data version changes when another connection commits a change to the index
    let observer = rusqlite::Connection::open(library.directory.path().join("db.sqlite3")).unwrap();
    let data_version = || observer.pragma_query_value(None, "data_version", |row| row.get::<_, i64>(0)).unwrap();
    let up_to_date = data_version();
    library.index();
    assert_eq!(data_version(), up_to_date);
    library.write("c.png", &png(2, 30));
    library.index();
    assert_ne!(data_version(), up_to_date);
  }

  #[test]
  fn encodes_images_in_batches_of_the_configured_size() {
    let library = Library::new();
    // gives images the time to pile up while the model is busy; they must not make the next batch larger
    library.encoding_time.set(Duration::from_millis(20));
    for shade in 0..7 {
      library.write(&format!("{shade}.png"), &png(2, shade));
    }
    library.index();
    let mut batches = library.batches.lock().unwrap().clone();
    batches.sort();
    assert_eq!(batches, [1, 2, 2, 2]);
    assert_eq!(library.images().len(), 7);
  }

  #[test]
  fn keeps_each_image_with_its_own_metadata_and_vector_when_another_one_fails_to_load() {
    let library = Library::new();
    library.write("a.png", &png(2, 10));
    library.write("b.png", b"not an image");
    library.write("c.png", &png(30, 200));
    library.index();
    assert_eq!(library.images(), ["a.png", "c.png"]);
    for name in ["a.png", "c.png"] {
      let (state, metadata) = (library.state(name), std::fs::metadata(library.path(name)).unwrap());
      assert_eq!((state.modified_at, state.size.try_into()), (walk::modified_at(&metadata), Ok(metadata.len())));
    }
    assert!(library.vector("a.png") < library.vector("c.png"));
  }

  #[test]
  fn skips_images_over_the_pixel_limit() {
    let library = Library::new();
    library.write("small.png", &png(10, 10));
    library.write("large.png", &png(50, 10));
    let filter = PathFilter::new(None, false, false);
    library.index_with(&filter, &library.root(), Some(100));
    assert_eq!(library.images(), ["small.png"]);
    library.index_with(&filter, &library.root(), None);
    assert_eq!(library.images(), ["large.png", "small.png"]);
  }

  #[test]
  fn reuses_the_vectors_of_renamed_and_duplicated_images() {
    let library = Library::new();
    library.write("old.png", &png(2, 10));
    library.index();
    std::fs::rename(library.path("old.png"), library.path("new.png")).unwrap();
    std::fs::copy(library.path("new.png"), library.path("copy.png")).unwrap();
    library.index();
    assert_eq!(library.images(), ["copy.png", "new.png"]);
    assert!(library.state("old.png").deleted);
    assert_eq!(library.encoded(), 1);
  }

  #[test]
  fn recomputes_the_vector_of_an_image_that_shares_only_its_hash_with_an_indexed_one() {
    let library = Library::new();
    let data = png(2, 10);
    library.write("a.png", &data);
    let other = NewImage { filepath: "other.png", modified_at: 1.0, size: 1, vector: &[0; 4], hash: &file_hash(&data) };
    library.db.upsert_image(&other).unwrap();
    library.index();
    assert_eq!(library.encoded(), 1);
  }

  #[test]
  fn restores_a_deleted_image_that_comes_back_unchanged() {
    let library = Library::new();
    let data = png(2, 10);
    library.write("a.png", &data);
    library.index();
    std::fs::remove_file(library.path("a.png")).unwrap();
    library.index();
    assert!(library.images().is_empty());
    std::fs::write(library.path("a.png"), &data).unwrap();
    set_modified(&library.path("a.png"), library.clock.get());
    library.index();
    assert_eq!((library.images(), library.encoded()), (vec!["a.png".to_string()], 1));
  }

  #[test]
  fn marks_missing_and_unreadable_images_as_deleted() {
    let library = Library::new();
    for name in ["kept.png", "removed.png", "broken.png"] {
      library.write(name, &png(2, 10));
    }
    library.index();
    std::fs::remove_file(library.path("removed.png")).unwrap();
    library.write("broken.png", b"not an image");
    library.index();
    assert_eq!(library.images(), ["kept.png"]);
    assert!(library.state("removed.png").deleted && library.state("broken.png").deleted);
  }

  #[cfg(unix)]
  #[test]
  fn keeps_images_in_unreadable_directories_and_behind_inaccessible_symlinks() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let library = Library::new();
    for name in ["kept.png", "removed.png", "locked/a.png", "../outside/b.png", "../outside/nested/c.png"] {
      library.write(name, &png(2, 10));
    }
    symlink(library.path("../outside/b.png"), library.path("file_link.png")).unwrap();
    symlink(library.path("../outside/nested"), library.path("dir_link")).unwrap();
    library.index();
    let set_mode = |mode| {
      for directory in ["locked", "../outside"] {
        std::fs::set_permissions(library.path(directory), std::fs::Permissions::from_mode(mode)).unwrap();
      }
    };
    set_mode(0o000);
    std::fs::remove_file(library.path("removed.png")).unwrap();
    library.index();
    set_mode(0o755);
    assert_eq!(library.images(), ["dir_link/c.png", "file_link.png", "kept.png", "locked/a.png"]);
    assert!(library.state("removed.png").deleted);
  }

  #[test]
  fn exclusions_hide_images_without_removing_them_from_the_index() {
    for (excluded, include_hidden, exclude_dirs) in
      [(".worktrees", false, None), (".git", true, None), ("private", true, Some(vec!["private".to_string()]))]
    {
      let library = Library::new();
      // the excluded name above the indexed directory must not affect the images in it
      let root = library.path(&format!("{excluded}/checkout"));
      let in_excluded = format!("{excluded}/photo.png");
      let mut all = vec![".photo.png", &in_excluded, "nested/photo.png"];
      all.sort();
      for (shade, name) in (0..).zip(&all) {
        library.write(&format!("{excluded}/checkout/{name}"), &png(2, shade));
      }
      let visible = if include_hidden { vec![".photo.png", "nested/photo.png"] } else { vec!["nested/photo.png"] };
      let everything = PathFilter::new(Some(vec!["unused".into()]), true, false);
      let restricted = PathFilter::new(exclude_dirs, include_hidden, false);

      library.index_with(&everything, &root, None);
      assert_eq!(library.browse(&everything, &root), all);
      // what is in the index already is filtered when indexing is skipped
      assert_eq!(library.browse(&restricted, &root), visible);
      // indexing with exclusions keeps the excluded images, also when a scan of a parent excludes them all
      library.index_with(&restricted, &root, None);
      library.index_with(&restricted, &library.root(), None);
      assert_eq!(library.browse(&restricted, &root), visible);
      library.index_with(&everything, &root, None);
      assert_eq!(library.browse(&everything, &root), all);
      assert_eq!(library.encoded(), 3);

      // the removal of an excluded image goes unnoticed until the image is no longer excluded
      std::fs::remove_file(library.path(&format!("{excluded}/checkout/{in_excluded}"))).unwrap();
      library.index_with(&restricted, &root, None);
      assert_eq!(library.browse(&everything, &root), all);
      library.index_with(&everything, &root, None);
      all.retain(|name| *name != in_excluded);
      assert_eq!(library.browse(&everything, &root), all);
    }
  }

  /// Applies random changes to a directory and checks after each indexing run that exactly the readable images
  /// on disk are in the index, and that every other image in the index is marked as deleted.
  #[test]
  fn tracks_images_through_random_changes() {
    let library = Library::new();
    let (photos, removed) = (library.root(), library.directory.path().join("removed"));
    std::fs::create_dir(&removed).unwrap();
    // a linear congruential generator, which makes the same changes on every run
    let mut seed = 372u64;
    let mut random = |bound: usize| {
      seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
      usize::try_from(seed >> 33).unwrap() % bound
    };
    let names = |directory: &Path| {
      let entries = std::fs::read_dir(directory).unwrap();
      let mut names: Vec<String> = entries.map(|entry| entry.unwrap().file_name().into_string().unwrap()).collect();
      names.sort();
      names
    };
    for name in 0..8 {
      library.write(&format!("{name}.png"), &png(2, name));
    }
    for round in 0..40 {
      for change in 0..1 + random(3) {
        let (files, removed_files) = (names(Path::new(&photos)), names(&removed));
        let image = png(2, u8::try_from(random(256)).unwrap());
        let new = format!("{}.png", 8 + 3 * round + change);
        let operation = ["add", "change", "corrupt", "delete", "rename", "duplicate", "restore"][random(7)];
        match (operation, files.get(random(files.len().max(1)))) {
          ("change", Some(file)) => library.write(file, &image),
          ("corrupt", Some(file)) => library.write(file, b"not an image"),
          ("delete", Some(file)) => std::fs::rename(library.path(file), removed.join(file)).unwrap(),
          ("rename", Some(file)) => std::fs::rename(library.path(file), library.path(&new)).unwrap(),
          ("duplicate", Some(file)) => {
            std::fs::write(library.path(&new), std::fs::read(library.path(file)).unwrap()).unwrap()
          }
          // the image comes back unchanged, with the modification time it has in the index
          ("restore", _) if !removed_files.is_empty() => {
            let file = &removed_files[random(removed_files.len())];
            std::fs::rename(removed.join(file), library.path(file)).unwrap()
          }
          _ => library.write(&new, &image),
        }
      }
      library.index();
      let readable = |name: &String| std::fs::read(library.path(name)).unwrap() != b"not an image";
      let expected: Vec<String> = names(Path::new(&photos)).into_iter().filter(readable).collect();
      assert_eq!(library.images(), expected);
    }
  }

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
    if !std::fs::metadata(image).is_err_and(|error| error.kind() == std::io::ErrorKind::PermissionDenied) {
      // A privileged user, such as root in a container, can read it anyway, so there is nothing to check.
      std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
      return;
    }
    let options = IndexOptions { batch_size: 1, raw_support: false, max_pixels: None, workers: 1 };
    let filter = PathFilter::new(None, false, false);
    let result = ensure_index(&db, || anyhow::bail!("no model"), &filter, directory.path().to_str().unwrap(), &options);
    std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    result.unwrap();
    assert!(!db.image_state(image).unwrap().unwrap().deleted);
  }
}
