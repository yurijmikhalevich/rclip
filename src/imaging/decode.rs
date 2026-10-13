use std::io::Cursor;

use image::{DynamicImage, ImageFormat, ImageReader};

use super::{IMAGE_RAW_EXT, ImageError, RgbImage, check_pixel_limit, heif, jp2, jpeg, raw};

/// Decodes an encoded image into RGB pixels.
///
/// Like Pillow, the format is detected from the file contents, with two exceptions inherited from
/// the Python version: RAW files are selected by `extension` and decoded with LibRaw, and `.heic`
/// files are decoded with the operating system's codec.
pub fn decode(data: &[u8], extension: &str, max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
  if IMAGE_RAW_EXT.contains(&extension) {
    return raw::decode(data, max_pixels);
  }
  if extension == "heic" && heif::is_supported() {
    return heif::decode(data, max_pixels);
  }
  if jp2::is_jp2(data) {
    return jp2::decode(data, max_pixels);
  }
  match image::guess_format(data) {
    Ok(ImageFormat::Jpeg) => jpeg::decode(data, max_pixels),
    Ok(format) => decode_generic(data, format, max_pixels),
    Err(_) => Err(ImageError::Unidentified),
  }
}

fn decode_generic(data: &[u8], format: ImageFormat, max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
  let reader = || ImageReader::with_format(Cursor::new(data), format);
  let (width, height) = reader().into_dimensions().map_err(|_| ImageError::Unidentified)?;
  check_pixel_limit(width, height, max_pixels)?;
  let mut reader = reader();
  reader.no_limits();
  let image = reader.decode().map_err(ImageError::decode)?;
  Ok(to_rgb(image))
}

/// Converts to 8-bit RGB the way Pillow does for the modes it opens these images in.
fn to_rgb(image: DynamicImage) -> RgbImage {
  let (width, height) = (image.width(), image.height());
  match image {
    DynamicImage::ImageRgb8(rgb) => RgbImage { width, height, pixels: rgb.into_raw() },
    DynamicImage::ImageRgba8(rgba) => RgbImage::from_rgba(width, height, rgba.as_raw()),
    DynamicImage::ImageLuma8(luma) => RgbImage::from_luma(width, height, luma.as_raw()),
    // Pillow opens 16-bit RGB(A) images in 8-bit modes by keeping the most significant byte.
    DynamicImage::ImageRgb16(rgb) => {
      RgbImage { width, height, pixels: rgb.as_raw().iter().map(|&value| (value >> 8) as u8).collect() }
    }
    DynamicImage::ImageRgba16(rgba) => RgbImage {
      width,
      height,
      pixels: rgba.as_raw().chunks_exact(4).flat_map(|pixel| [0, 1, 2].map(|c| (pixel[c] >> 8) as u8)).collect(),
    },
    other => RgbImage { width, height, pixels: other.into_rgb8().into_raw() },
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn unknown_data_is_unidentified() {
    assert!(matches!(decode(b"definitely not an image", "jpg", None), Err(ImageError::Unidentified)));
  }
}
