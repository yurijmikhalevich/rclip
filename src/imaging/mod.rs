//! Image decoding with Pillow-compatible pixel output.
//!
//! Every decoder here returns plain 8-bit RGB pixels converted the way Pillow's
//! `Image.convert("RGB")` would, so that the Rust and Python versions of rclip compute the same
//! embeddings for the same files and can share one index.

mod decode;
mod heif;
mod jp2;
mod jpeg;
pub mod preprocess;
mod raw;
pub mod resample;

use std::io::Cursor;
use std::path::Path;

use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader};

pub use decode::decode;

/// Formats that are always indexed.
pub const IMAGE_EXT: &[&str] =
  &["jpg", "jpeg", "png", "webp", "heic", "tiff", "tif", "bmp", "gif", "jp2", "pnm", "pbm", "pgm", "ppm"];
/// RAW formats are indexed only with `--experimental-raw-support` and only when there is no
/// processed image with the same name alongside them.
pub const IMAGE_RAW_EXT: &[&str] =
  &["3fr", "arw", "cr2", "cr3", "dng", "erf", "iiq", "mos", "mrw", "nef", "nrw", "orf", "pef", "raf", "rw2", "srw"];

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

  /// The image as it should be displayed, given its EXIF orientation.
  pub fn oriented(self, orientation: Orientation) -> RgbImage {
    let pixels = image::RgbImage::from_raw(self.width, self.height, self.pixels).expect("the pixels fill the image");
    let mut image = DynamicImage::ImageRgb8(pixels);
    image.apply_orientation(orientation);
    let image = image.into_rgb8();
    RgbImage { width: image.width(), height: image.height(), pixels: image.into_raw() }
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

/// Reads and decodes an image file like [`decode`].
pub fn read_image(path: &str, max_pixels: Option<u64>, min_side: Option<u32>) -> Result<RgbImage, ImageError> {
  let path = path.strip_prefix("file://").unwrap_or(path);
  let data = std::fs::read(path)?;
  decode(&data, &file_extension(path), max_pixels, min_side)
}

/// The interactive UI's background, shown behind transparent pixels.
const DISPLAY_BACKGROUND: u8 = 0x12;

/// Reads an image for display: upright according to its EXIF orientation and with transparency
/// flattened onto the UI background, like the Python version's interactive UI. `draft` lets JPEGs
/// decode at a reduced scale that keeps the stored image at least that large.
pub fn read_display_image(path: &str, draft: Option<(u32, u32)>) -> Result<RgbImage, ImageError> {
  let data = std::fs::read(path)?;
  let extension = file_extension(path);
  let image = decode::decode_flattened(&data, &extension, DISPLAY_BACKGROUND, draft)?;
  // RAW files show their embedded preview or LibRaw's output, which are already upright, and the
  // native HEIF decoders apply the orientation themselves.
  if IMAGE_RAW_EXT.contains(&extension.as_str()) || extension == "heic" {
    return Ok(image);
  }
  Ok(image.oriented(orientation(&data).unwrap_or(Orientation::NoTransforms)))
}

/// The EXIF orientation of a JPEG, PNG, WebP, or TIFF image.
fn orientation(data: &[u8]) -> Option<Orientation> {
  ImageReader::new(Cursor::new(data)).with_guessed_format().ok()?.into_decoder().ok()?.orientation().ok()
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
    assert_eq!(values(&image.clone().oriented(Orientation::from_exif(1).unwrap())), [0, 1, 2, 3, 4, 5]);
    assert_eq!(values(&image.clone().oriented(Orientation::from_exif(2).unwrap())), [2, 1, 0, 5, 4, 3]);
    assert_eq!(values(&image.clone().oriented(Orientation::from_exif(3).unwrap())), [5, 4, 3, 2, 1, 0]);
    assert_eq!(values(&image.clone().oriented(Orientation::from_exif(4).unwrap())), [3, 4, 5, 0, 1, 2]);
    assert_eq!(values(&image.clone().oriented(Orientation::from_exif(5).unwrap())), [0, 3, 1, 4, 2, 5]);
    assert_eq!(values(&image.clone().oriented(Orientation::from_exif(6).unwrap())), [3, 0, 4, 1, 5, 2]);
    assert_eq!(values(&image.clone().oriented(Orientation::from_exif(7).unwrap())), [5, 2, 4, 1, 3, 0]);
    assert_eq!(values(&image.oriented(Orientation::from_exif(8).unwrap())), [2, 5, 1, 4, 0, 3]);
  }

  /// Writes a 4x2 image with an EXIF block whose orientation is 6.
  fn write_with_exif(mut encoder: impl image::ImageEncoder) {
    // A big-endian TIFF block with one entry: orientation 6, a SHORT stored inline.
    let exif = b"MM\0\x2a\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0\x06\0\0\0\0\0\0";
    encoder.set_exif_metadata(exif.to_vec()).unwrap();
    encoder.write_image(&[200; 4 * 2 * 3], 4, 2, image::ExtendedColorType::Rgb8).unwrap();
  }

  #[test]
  fn reads_the_orientation_from_exif() {
    let (mut jpeg, mut png, mut webp) = (Vec::new(), Vec::new(), Vec::new());
    write_with_exif(image::codecs::jpeg::JpegEncoder::new(&mut jpeg));
    write_with_exif(image::codecs::png::PngEncoder::new(&mut png));
    write_with_exif(image::codecs::webp::WebPEncoder::new_lossless(&mut webp));
    for data in [jpeg, png, webp] {
      assert_eq!(orientation(&data), Some(Orientation::Rotate90));
    }
    assert_eq!(orientation(b"GIF89a"), None);
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
