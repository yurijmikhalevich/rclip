//! Loads images in the background and keeps them transmitted to the terminal.
//!
//! Decoding, scaling, and PNG encoding happen on loader threads. The UI thread only writes the
//! finished transmissions to the terminal between frames, so it never stalls on image work.
//! Transmitted images stay in the terminal until a byte budget forces the least recently used
//! ones out, so scrolling back or reopening an image needs no new transmission.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::sync::{Arc, Condvar, Mutex};

use crossbeam_channel::Sender;

use super::kitty;
use crate::imaging::resample::{Filter, resize};
use crate::imaging::{self, RgbImage};

/// Kitty silently evicts the oldest images once a per-buffer quota of decoded pixels is reached,
/// so keep the transmitted images well below it.
const TRANSMITTED_BUDGET: usize = 64 * 1024 * 1024;
/// Budget for decoded images kept around to re-encode at another size without decoding again.
const PREPARED_BUDGET: usize = 192 * 1024 * 1024;
/// At most this many images decode at once; decoding is memory hungry.
const LOADERS: usize = 4;

/// The two prepared sizes: grid and filmstrip previews, and the detail view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
  Preview,
  Detail,
}

impl Variant {
  fn max_size(self) -> (u32, u32) {
    match self {
      Variant::Preview => (640, 480),
      Variant::Detail => (1920, 1920),
    }
  }
}

/// How an image is laid out in its box of cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fit {
  /// At its natural size, scaled down to fit if needed, and centered on whole cells.
  Natural,
  /// Scaled to fill the box while keeping its aspect ratio, centered with pixel precision.
  Fill,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageKey {
  pub path: String,
  pub variant: Variant,
  pub fit: Fit,
  /// The box the image must fit in, in cells.
  pub cells: (u16, u16),
  /// The terminal's cell size in pixels.
  pub cell_pixels: (u16, u16),
}

/// A transmission prepared by a loader thread.
pub struct Prepared {
  png: Vec<u8>,
  /// The cells the image covers and their offset inside its box.
  cells: (u16, u16),
  offset: (u16, u16),
  decoded_bytes: usize,
}

pub enum ImageState<'a> {
  Loading,
  Ready { id: u32, cells: (u16, u16), offset: (u16, u16) },
  Failed(&'a str),
}

enum Slot {
  Loading,
  Ready { id: u32, cells: (u16, u16), offset: (u16, u16), decoded_bytes: usize, last_used: u64 },
  Failed(String),
}

pub struct Images {
  slots: HashMap<ImageKey, Slot>,
  /// Keys used while drawing the current frame; they are never evicted.
  in_use: HashSet<ImageKey>,
  frame: u64,
  transmitted_bytes: usize,
  /// Escape sequences waiting to be written before the next frame.
  pending: Vec<u8>,
  ids: kitty::IdAllocator,
  tmux: bool,
  loader: Arc<LoaderShared>,
  threads: Vec<std::thread::JoinHandle<()>>,
}

type Notify = Box<dyn Fn(ImageKey, Result<Prepared, String>) + Send + Sync>;

struct LoaderShared {
  state: Mutex<LoaderState>,
  wake: Condvar,
  notify: Notify,
}

struct LoaderState {
  jobs: VecDeque<ImageKey>,
  stop: bool,
  prepared: PreparedCache,
}

impl Images {
  /// `notify` is called on a loader thread when an image is ready; the result must be handed back
  /// through [`Images::loaded`] on the UI thread.
  pub fn new<E: Send + 'static>(events: Sender<E>, wrap: fn(ImageKey, Result<Prepared, String>) -> E) -> Self {
    let notify: Notify = Box::new(move |key, result| {
      events.send(wrap(key, result)).ok();
    });
    let loader = Arc::new(LoaderShared {
      state: Mutex::new(LoaderState { jobs: VecDeque::new(), stop: false, prepared: PreparedCache::default() }),
      wake: Condvar::new(),
      notify,
    });
    let threads = (0..LOADERS)
      .map(|_| {
        let loader = Arc::clone(&loader);
        std::thread::spawn(move || run_loader(&loader))
      })
      .collect();
    let tmux = std::env::var_os("TMUX").is_some() || std::env::var("TERM").is_ok_and(|term| term.starts_with("tmux"));
    Images {
      slots: HashMap::new(),
      in_use: HashSet::new(),
      frame: 0,
      transmitted_bytes: 0,
      pending: Vec::new(),
      ids: kitty::IdAllocator::new(),
      tmux,
      loader,
      threads,
    }
  }

  pub fn begin_frame(&mut self) {
    self.frame += 1;
    self.in_use.clear();
  }

  /// Returns the state of an image, starting to load it if needed. Images requested while drawing
  /// a frame stay loaded and transmitted.
  pub fn get(&mut self, key: &ImageKey) -> ImageState<'_> {
    self.in_use.insert(key.clone());
    let frame = self.frame;
    let slot = self.slots.entry(key.clone()).or_insert_with(|| {
      let mut state = self.loader.state.lock().expect("the loader lock is never poisoned");
      state.jobs.push_back(key.clone());
      self.loader.wake.notify_one();
      Slot::Loading
    });
    match slot {
      Slot::Loading => ImageState::Loading,
      Slot::Ready { id, cells, offset, last_used, .. } => {
        *last_used = frame;
        ImageState::Ready { id: *id, cells: *cells, offset: *offset }
      }
      Slot::Failed(error) => ImageState::Failed(error),
    }
  }

  /// Drops queued loads nobody asked for in this frame and evicts transmitted images over budget.
  pub fn end_frame(&mut self) {
    {
      let mut state = self.loader.state.lock().expect("the loader lock is never poisoned");
      let in_use = &self.in_use;
      let slots = &mut self.slots;
      state.jobs.retain(|key| {
        let wanted = in_use.contains(key);
        if !wanted {
          slots.remove(key);
        }
        wanted
      });
    }
    // Failed images are retried when requested again after they leave the screen.
    self.slots.retain(|key, slot| !matches!(slot, Slot::Failed(_)) || self.in_use.contains(key));
    if self.transmitted_bytes <= TRANSMITTED_BUDGET {
      return;
    }
    let mut evictable: Vec<(u64, ImageKey)> = self
      .slots
      .iter()
      .filter(|(key, _)| !self.in_use.contains(*key))
      .filter_map(|(key, slot)| match slot {
        Slot::Ready { last_used, .. } => Some((*last_used, key.clone())),
        _ => None,
      })
      .collect();
    evictable.sort_by_key(|(last_used, _)| *last_used);
    for (_, key) in evictable {
      if self.transmitted_bytes <= TRANSMITTED_BUDGET {
        break;
      }
      if let Some(Slot::Ready { id, decoded_bytes, .. }) = self.slots.remove(&key) {
        self.transmitted_bytes -= decoded_bytes;
        self.pending.extend(kitty::delete(id, self.tmux));
      }
    }
  }

  /// Records a finished load. Returns whether the image is still wanted (and needs a redraw).
  pub fn loaded(&mut self, key: ImageKey, result: Result<Prepared, String>) -> bool {
    let Some(slot) = self.slots.get_mut(&key) else { return false };
    *slot = match result {
      Ok(prepared) => {
        let id = self.ids.next();
        self.pending.extend(kitty::transmit(id, &prepared.png, prepared.cells.0, prepared.cells.1, self.tmux));
        self.transmitted_bytes += prepared.decoded_bytes;
        Slot::Ready {
          id,
          cells: prepared.cells,
          offset: prepared.offset,
          decoded_bytes: prepared.decoded_bytes,
          last_used: self.frame,
        }
      }
      Err(error) => Slot::Failed(error),
    };
    true
  }

  /// Writes pending transmissions and deletions; call before drawing a frame.
  pub fn flush(&mut self, output: &mut impl Write) -> std::io::Result<()> {
    if !self.pending.is_empty() {
      output.write_all(&self.pending)?;
      output.flush()?;
      self.pending.clear();
    }
    Ok(())
  }

  /// Frees every transmitted image, e.g. before exiting.
  pub fn clear(&mut self, output: &mut impl Write) -> std::io::Result<()> {
    for slot in self.slots.values() {
      if let Slot::Ready { id, .. } = slot {
        self.pending.extend(kitty::delete(*id, self.tmux));
      }
    }
    self.slots.clear();
    self.transmitted_bytes = 0;
    self.flush(output)
  }
}

impl Drop for Images {
  fn drop(&mut self) {
    self.loader.state.lock().expect("the loader lock is never poisoned").stop = true;
    self.loader.wake.notify_all();
    for thread in self.threads.drain(..) {
      thread.join().ok();
    }
  }
}

fn run_loader(loader: &LoaderShared) {
  loop {
    let key = {
      let mut state = loader.state.lock().expect("the loader lock is never poisoned");
      loop {
        if state.stop {
          return;
        }
        if let Some(key) = state.jobs.pop_front() {
          break key;
        }
        state = loader.wake.wait(state).expect("the loader lock is never poisoned");
      }
    };
    let cached = loader.state.lock().expect("the loader lock is never poisoned").prepared.get(&key.path, key.variant);
    let image = match cached {
      Some(image) => Ok(image),
      None => prepare(&key.path, key.variant).map(|image| {
        let image = Arc::new(image);
        let mut state = loader.state.lock().expect("the loader lock is never poisoned");
        state.prepared.insert(key.path.clone(), key.variant, Arc::clone(&image));
        image
      }),
    };
    let result = image.and_then(|image| encode(&image, &key));
    (loader.notify)(key, result);
  }
}

/// An upright, flattened image no larger than the variant's size, like the Python version's
/// `prepare_image`.
pub fn prepare(path: &str, variant: Variant) -> Result<RgbImage, String> {
  let (max_width, max_height) = variant.max_size();
  // Twice the shown size leaves the resampling below enough pixels to keep the image sharp.
  let image =
    imaging::read_display_image(path, Some((max_width * 2, max_height * 2))).map_err(|error| error.to_string())?;
  if image.width <= max_width && image.height <= max_height {
    return Ok(image);
  }
  let scale = (f64::from(max_width) / f64::from(image.width)).min(f64::from(max_height) / f64::from(image.height));
  let width = ((f64::from(image.width) * scale).round() as u32).clamp(1, max_width);
  let height = ((f64::from(image.height) * scale).round() as u32).clamp(1, max_height);
  Ok(resize(&image, width, height, Filter::Lanczos))
}

/// Scales `image` into the key's box and encodes it as PNG, padding it with transparent pixels to
/// whole cells so that the terminal never stretches it.
fn encode(image: &RgbImage, key: &ImageKey) -> Result<Prepared, String> {
  let (cell_width, cell_height) = (u32::from(key.cell_pixels.0.max(1)), u32::from(key.cell_pixels.1.max(1)));
  let (box_columns, box_rows) = (key.cells.0.clamp(1, kitty::MAX_CELLS), key.cells.1.clamp(1, kitty::MAX_CELLS));
  let (box_width, box_height) = (u32::from(box_columns) * cell_width, u32::from(box_rows) * cell_height);
  let mut scale = (f64::from(box_width) / f64::from(image.width)).min(f64::from(box_height) / f64::from(image.height));
  if key.fit == Fit::Natural {
    scale = scale.min(1.0);
  }
  let width = ((f64::from(image.width) * scale).round() as u32).clamp(1, box_width);
  let height = ((f64::from(image.height) * scale).round() as u32).clamp(1, box_height);
  let (columns, rows) = match key.fit {
    Fit::Fill => (box_columns, box_rows),
    Fit::Natural => (width.div_ceil(cell_width) as u16, height.div_ceil(cell_height) as u16),
  };
  let scaled = if (width, height) == (image.width, image.height) {
    image.clone()
  } else {
    resize(image, width, height, Filter::Bicubic)
  };
  let (canvas_width, canvas_height) = (u32::from(columns) * cell_width, u32::from(rows) * cell_height);
  let (left, top) = ((canvas_width - width) / 2, (canvas_height - height) / 2);
  let mut rgba = vec![0u8; canvas_width as usize * canvas_height as usize * 4];
  for y in 0..height as usize {
    let source = &scaled.pixels[y * width as usize * 3..(y + 1) * width as usize * 3];
    let start = ((top as usize + y) * canvas_width as usize + left as usize) * 4;
    for (target, pixel) in
      rgba[start..start + width as usize * 4].as_chunks_mut::<4>().0.iter_mut().zip(source.as_chunks::<3>().0)
    {
      target.copy_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
    }
  }
  let mut png = Vec::new();
  image::ImageEncoder::write_image(
    image::codecs::png::PngEncoder::new(&mut png),
    &rgba,
    canvas_width,
    canvas_height,
    image::ExtendedColorType::Rgba8,
  )
  .map_err(|error| error.to_string())?;
  Ok(Prepared {
    png,
    cells: (columns, rows),
    offset: ((box_columns - columns) / 2, (box_rows - rows) / 2),
    decoded_bytes: rgba.len(),
  })
}

/// Decoded images by path and variant, least recently used first out.
#[derive(Default)]
struct PreparedCache {
  images: HashMap<(String, Variant), (Arc<RgbImage>, u64)>,
  bytes: usize,
  clock: u64,
}

impl PreparedCache {
  fn get(&mut self, path: &str, variant: Variant) -> Option<Arc<RgbImage>> {
    self.clock += 1;
    let (image, last_used) = self.images.get_mut(&(path.to_string(), variant))?;
    *last_used = self.clock;
    Some(Arc::clone(image))
  }

  fn insert(&mut self, path: String, variant: Variant, image: Arc<RgbImage>) {
    self.clock += 1;
    self.bytes += image.pixels.len();
    if let Some((old, _)) = self.images.insert((path, variant), (image, self.clock)) {
      self.bytes -= old.pixels.len();
    }
    while self.bytes > PREPARED_BUDGET && self.images.len() > 1 {
      let oldest = self.images.iter().min_by_key(|(_, (_, last_used))| *last_used).map(|(key, _)| key.clone());
      if let Some((image, _)) = oldest.and_then(|key| self.images.remove(&key)) {
        self.bytes -= image.pixels.len();
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn key(fit: Fit, cells: (u16, u16)) -> ImageKey {
    ImageKey { path: String::new(), variant: Variant::Preview, fit, cells, cell_pixels: (10, 20) }
  }

  #[test]
  fn natural_fit_keeps_small_images_small_and_centers_them_on_cells() {
    let image = RgbImage { width: 30, height: 30, pixels: vec![255; 30 * 30 * 3] };
    let prepared = encode(&image, &key(Fit::Natural, (10, 10))).unwrap();
    assert_eq!(prepared.cells, (3, 2));
    assert_eq!(prepared.offset, (3, 4));
    assert_eq!(prepared.decoded_bytes, 30 * 40 * 4);
  }

  #[test]
  fn fill_scales_to_the_whole_box() {
    let image = RgbImage { width: 30, height: 30, pixels: vec![255; 30 * 30 * 3] };
    let prepared = encode(&image, &key(Fit::Fill, (10, 10))).unwrap();
    assert_eq!(prepared.cells, (10, 10));
    assert_eq!(prepared.offset, (0, 0));
    let decoded = image::load_from_memory(&prepared.png).unwrap().to_rgba8();
    assert_eq!((decoded.width(), decoded.height()), (100, 200));
    // The square image is centered vertically, leaving transparent bands above and below.
    assert_eq!(decoded.get_pixel(50, 10)[3], 0);
    assert_eq!(decoded.get_pixel(50, 100)[3], 255);
  }

  #[test]
  fn natural_fit_scales_large_images_down() {
    let image = RgbImage { width: 400, height: 100, pixels: vec![0; 400 * 100 * 3] };
    let prepared = encode(&image, &key(Fit::Natural, (20, 10))).unwrap();
    assert_eq!(prepared.cells, (20, 3));
  }

  #[test]
  fn prepared_cache_evicts_the_least_recently_used_images() {
    let mut cache = PreparedCache::default();
    let big = || Arc::new(RgbImage { width: 1, height: 1, pixels: vec![0; PREPARED_BUDGET / 2 + 1] });
    cache.insert("a".into(), Variant::Preview, big());
    cache.insert("b".into(), Variant::Preview, big());
    assert!(cache.get("a", Variant::Preview).is_none());
    assert!(cache.get("b", Variant::Preview).is_some());
  }
}
