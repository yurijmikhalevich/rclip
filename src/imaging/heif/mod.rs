//! Platform-native HEIF decoding.
//!
//! Only codecs supplied by the operating system are used: where none is available, HEIF images
//! are not supported, exactly like in the Python version.

#[cfg(target_os = "macos")]
mod macos;
#[cfg(windows)]
mod windows;

#[cfg(target_os = "macos")]
pub use macos::decode;
#[cfg(windows)]
pub use windows::decode;

pub fn is_supported() -> bool {
  cfg!(any(target_os = "macos", windows))
}

#[cfg(not(any(target_os = "macos", windows)))]
pub fn decode(
  _data: &[u8],
  _max_pixels: Option<u64>,
  _min_side: Option<u32>,
) -> Result<super::RgbImage, super::ImageError> {
  Err(super::ImageError::decode("native HEIF decoding is available only on macOS and Windows"))
}

#[cfg(all(test, any(target_os = "macos", windows)))]
mod tests {
  use crate::imaging::preprocess::IMAGE_SIZE;
  use crate::imaging::{ImageError, decode};

  /// A 1600x1200 image, or `None` where Windows has no HEIF and HEVC extensions installed (CI
  /// runners lack them), in which case the test has nothing to check.
  fn large_heic() -> Option<Vec<u8>> {
    let data = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/large.heic")).unwrap();
    if cfg!(windows) && matches!(decode(&data, "heic", None, None), Err(ImageError::Decode(_))) {
      eprintln!("skipping: the operating system HEIF codec is unavailable");
      return None;
    }
    Some(data)
  }

  #[test]
  fn decodes_the_full_frame_by_default() {
    let Some(data) = large_heic() else { return };
    let image = decode(&data, "heic", None, None).unwrap();
    assert_eq!((image.width, image.height), (1600, 1200));
    assert_eq!(image.pixels.len(), 1600 * 1200 * 3);
  }

  #[test]
  fn downscales_to_the_minimum_side_while_decoding() {
    let Some(data) = large_heic() else { return };
    let image = decode(&data, "heic", None, Some(IMAGE_SIZE)).unwrap();
    assert!(image.width.min(image.height) >= IMAGE_SIZE);
    assert!(image.width * image.height < 1600 * 1200);
  }

  #[cfg(target_os = "macos")]
  #[test]
  fn downscaled_decode_matches_the_full_frame_decode() {
    use crate::imaging::resample::{Filter, resize};

    let Some(data) = large_heic() else { return };
    let downscaled = decode(&data, "heic", None, Some(IMAGE_SIZE)).unwrap();
    let full = decode(&data, "heic", None, None).unwrap();
    let expected = resize(&full, downscaled.width, downscaled.height, Filter::Bicubic);
    let difference: usize = downscaled
      .pixels
      .iter()
      .zip(&expected.pixels)
      .map(|(&actual, &expected)| usize::from(actual.abs_diff(expected)))
      .sum();
    assert!(difference < 10 * expected.pixels.len(), "the mean difference is {}", difference / expected.pixels.len());
  }

  #[test]
  fn enforces_the_pixel_limit_on_the_source_size() {
    let Some(data) = large_heic() else { return };
    let result = decode(&data, "heic", Some(1_000_000), Some(IMAGE_SIZE));
    assert!(matches!(result, Err(ImageError::TooLarge { pixels: 1_920_000, limit: 1_000_000 })));
  }
}
