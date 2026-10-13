//! Image decoding with Pillow-compatible pixel output.
//!
//! Every decoder here returns plain 8-bit RGB pixels converted the way Pillow's
//! `Image.convert("RGB")` would, so that the Rust and Python versions of rclip compute the same
//! embeddings for the same files and can share one index.

mod decode;
mod exif;
mod heif;
mod jp2;
mod jpeg;
pub mod preprocess;
mod raw;
pub mod resample;

use std::path::Path;

pub use decode::decode;

/// Formats that are always indexed.
pub const IMAGE_EXT: &[&str] =
  &["jpg", "jpeg", "png", "webp", "heic", "tiff", "tif", "bmp", "gif", "jp2", "pnm", "pbm", "pgm", "ppm"];
/// RAW formats are indexed only with `--experimental-raw-support` and only when there is no
/// processed image with the same name alongside them.
pub const IMAGE_RAW_EXT: &[&str] = &["arw", "cr2", "dng"];

/// Pillow's historic decompression-bomb limit; the automatic cap never goes below it so that
/// ordinary large photos (high-MP panoramas, scans) always index.
pub const MIN_MAX_IMAGE_PIXELS: u64 = 89_478_485;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbImage {
  pub width: u32,
  pub height: u32,
  /// Interleaved RGB rows without padding.
  pub pixels: Vec<u8>,
}

impl RgbImage {
  pub fn from_luma(width: u32, height: u32, luma: &[u8]) -> Self {
    Self { width, height, pixels: luma.iter().flat_map(|&value| [value; 3]).collect() }
  }

  /// Drops the alpha channel without compositing, like Pillow's RGBA to RGB conversion.
  pub fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Self {
    Self {
      width,
      height,
      pixels: rgba.as_chunks::<4>().0.iter().flat_map(|pixel| [pixel[0], pixel[1], pixel[2]]).collect(),
    }
  }

  /// Applies an EXIF orientation (1-8), returning the image as it should be displayed.
  pub fn oriented(self, orientation: u32) -> RgbImage {
    if !(2..=8).contains(&orientation) {
      return self;
    }
    let (width, height) = (self.width as usize, self.height as usize);
    let transposed = orientation >= 5;
    let (out_width, out_height) = if transposed { (height, width) } else { (width, height) };
    let mut pixels = vec![0u8; self.pixels.len()];
    for y in 0..out_height {
      for x in 0..out_width {
        // Map each output coordinate back to the stored pixel it shows.
        let (sx, sy) = match orientation {
          2 => (width - 1 - x, y),
          3 => (width - 1 - x, height - 1 - y),
          4 => (x, height - 1 - y),
          5 => (y, x),
          6 => (y, height - 1 - x),
          7 => (width - 1 - y, height - 1 - x),
          _ => (width - 1 - y, x),
        };
        let source = (sy * width + sx) * 3;
        let target = (y * out_width + x) * 3;
        pixels[target..target + 3].copy_from_slice(&self.pixels[source..source + 3]);
      }
    }
    RgbImage { width: out_width as u32, height: out_height as u32, pixels }
  }
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
  #[error("{0}")]
  Io(#[from] std::io::Error),
  #[error("image has {pixels} pixels, which exceeds the limit of {limit} pixels")]
  TooLarge { pixels: u64, limit: u64 },
  #[error("cannot identify image file")]
  Unidentified,
  #[error("{0}")]
  Decode(String),
}

impl ImageError {
  pub fn decode(error: impl std::fmt::Display) -> Self {
    ImageError::Decode(error.to_string())
  }
}

/// Fails with [`ImageError::TooLarge`] when an image of `width`x`height` exceeds `max_pixels`.
pub fn check_pixel_limit(width: u32, height: u32, max_pixels: Option<u64>) -> Result<(), ImageError> {
  let pixels = u64::from(width) * u64::from(height);
  match max_pixels {
    Some(limit) if pixels > limit => Err(ImageError::TooLarge { pixels, limit }),
    _ => Ok(()),
  }
}

/// Lowercase extension without the dot, or an empty string.
pub fn file_extension(path: &str) -> String {
  Path::new(path).extension().and_then(|ext| ext.to_str()).map(str::to_ascii_lowercase).unwrap_or_default()
}

/// Reads and decodes an image file. `max_pixels` of `None` disables the decompression-bomb cap.
pub fn read_image(path: &str, max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
  let path = path.strip_prefix("file://").unwrap_or(path);
  let data = std::fs::read(path)?;
  decode(&data, &file_extension(path), max_pixels)
}

/// The interactive UI's background, shown behind transparent pixels.
const DISPLAY_BACKGROUND: u8 = 0x12;

/// Reads an image for display: upright according to its EXIF orientation and with transparency
/// flattened onto the UI background, like the Python version's interactive UI.
pub fn read_display_image(path: &str) -> Result<RgbImage, ImageError> {
  let data = std::fs::read(path)?;
  let extension = file_extension(path);
  let image = decode::decode_flattened(&data, &extension, DISPLAY_BACKGROUND)?;
  // RAW files show their embedded preview or LibRaw's output, which are already upright, and the
  // native HEIF decoders apply the orientation themselves.
  if IMAGE_RAW_EXT.contains(&extension.as_str()) || extension == "heic" {
    return Ok(image);
  }
  Ok(image.oriented(exif::orientation(&data).map_or(1, u32::from)))
}

/// Picks the pixel cap from the machine's RAM so that only images we can reasonably decode are
/// attempted, never going below [`MIN_MAX_IMAGE_PIXELS`].
pub fn default_max_image_pixels(workers: usize) -> u64 {
  // A decoded pixel costs ~3 bytes, but decoding and preprocessing keep extra copies around.
  const DECODED_BYTES_PER_PIXEL: u64 = 4;
  match total_memory_bytes() {
    Some(total) if total > 0 => {
      let estimate = total / 2 / (workers.max(1) as u64 * DECODED_BYTES_PER_PIXEL);
      estimate.max(MIN_MAX_IMAGE_PIXELS)
    }
    _ => MIN_MAX_IMAGE_PIXELS,
  }
}

#[cfg(unix)]
fn total_memory_bytes() -> Option<u64> {
  // SAFETY: sysconf has no preconditions.
  let (pages, page_size) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
  (pages > 0 && page_size > 0).then(|| pages as u64 * page_size as u64)
}

#[cfg(windows)]
fn total_memory_bytes() -> Option<u64> {
  use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
  let mut status = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
  // SAFETY: `status` is a properly initialized MEMORYSTATUSEX with its length set.
  unsafe { GlobalMemoryStatusEx(&mut status) }.ok()?;
  Some(status.ullTotalPhys)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn numbered(width: u32, height: u32) -> RgbImage {
    let pixels = (0..width * height).flat_map(|value| [value as u8; 3]).collect();
    RgbImage { width, height, pixels }
  }

  fn values(image: &RgbImage) -> Vec<u8> {
    image.pixels.as_chunks::<3>().0.iter().map(|pixel| pixel[0]).collect()
  }

  #[test]
  fn orientation_matches_pillow_transposes() {
    // 0 1 2
    // 3 4 5
    let image = numbered(3, 2);
    assert_eq!(values(&image.clone().oriented(1)), [0, 1, 2, 3, 4, 5]);
    assert_eq!(values(&image.clone().oriented(2)), [2, 1, 0, 5, 4, 3]);
    assert_eq!(values(&image.clone().oriented(3)), [5, 4, 3, 2, 1, 0]);
    assert_eq!(values(&image.clone().oriented(4)), [3, 4, 5, 0, 1, 2]);
    assert_eq!(values(&image.clone().oriented(5)), [0, 3, 1, 4, 2, 5]);
    assert_eq!(values(&image.clone().oriented(6)), [3, 0, 4, 1, 5, 2]);
    assert_eq!(values(&image.clone().oriented(7)), [5, 2, 4, 1, 3, 0]);
    assert_eq!(values(&image.oriented(8)), [2, 5, 1, 4, 0, 3]);
  }

  #[test]
  fn pixel_limit_is_inclusive() {
    assert!(check_pixel_limit(10, 10, Some(100)).is_ok());
    assert!(matches!(check_pixel_limit(10, 11, Some(100)), Err(ImageError::TooLarge { pixels: 110, limit: 100 })));
    assert!(check_pixel_limit(u32::MAX, u32::MAX, None).is_ok());
  }

  #[test]
  fn default_cap_never_drops_below_the_floor() {
    assert!(default_max_image_pixels(1_000_000) >= MIN_MAX_IMAGE_PIXELS);
  }
}
