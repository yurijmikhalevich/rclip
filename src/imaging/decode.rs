use std::io::Cursor;

use image::{DynamicImage, ImageFormat, ImageReader};

use super::{IMAGE_RAW_EXT, ImageError, RgbImage, check_pixel_limit, heif, jp2, jpeg, raw};

/// Decodes an encoded image into RGB pixels.
///
/// Like Pillow, the format is detected from the file contents, with two exceptions inherited from
/// the Python version: RAW files are selected by `extension` and decoded with LibRaw, and `.heic`
/// files are decoded with the operating system's codec.
///
/// `max_pixels` of `None` disables the decompression-bomb cap. `min_side` lets the JPEG and HEIF
/// decoders downscale while decoding, which skips most of the work for camera-sized photos, as long
/// as the shortest side stays at least that long; leave it unset when the image is displayed.
pub fn decode(
  data: &[u8],
  extension: &str,
  max_pixels: Option<u64>,
  min_side: Option<u32>,
) -> Result<RgbImage, ImageError> {
  match generic_format(data, extension) {
    Some(format) => Ok(to_rgb(decode_generic(data, format, max_pixels)?)),
    None => decode_special(data, extension, max_pixels, min_side.map(|side| (side, side)), min_side),
  }
}

/// Decodes an image for display, flattening transparency onto `background` the way the Python
/// version's interactive UI does instead of dropping the alpha channel. `draft` lets JPEGs decode
/// at a reduced scale that keeps the image at least that large; HEIF images keep their full frame.
pub fn decode_flattened(
  data: &[u8],
  extension: &str,
  background: u8,
  draft: Option<(u32, u32)>,
) -> Result<RgbImage, ImageError> {
  match generic_format(data, extension) {
    Some(format) => {
      let image = decode_generic(data, format, None)?;
      if image.color().has_alpha() { Ok(flatten(&image.into_rgba8(), background)) } else { Ok(to_rgb(image)) }
    }
    None => decode_special(data, extension, None, draft, None),
  }
}

/// The format of images decoded by the `image` crate, or `None` for the ones with dedicated
/// decoders.
fn generic_format(data: &[u8], extension: &str) -> Option<ImageFormat> {
  if IMAGE_RAW_EXT.contains(&extension) || (extension == "heic" && heif::is_supported()) || jp2::is_jp2(data) {
    return None;
  }
  image::guess_format(data).ok().filter(|format| *format != ImageFormat::Jpeg)
}

fn decode_special(
  data: &[u8],
  extension: &str,
  max_pixels: Option<u64>,
  jpeg_draft: Option<(u32, u32)>,
  heif_min_side: Option<u32>,
) -> Result<RgbImage, ImageError> {
  if IMAGE_RAW_EXT.contains(&extension) {
    raw::decode(data, max_pixels, jpeg_draft)
  } else if extension == "heic" && heif::is_supported() {
    heif::decode(data, max_pixels, heif_min_side)
  } else if jp2::is_jp2(data) {
    jp2::decode(data, max_pixels)
  } else if image::guess_format(data).is_ok_and(|format| format == ImageFormat::Jpeg) {
    jpeg::decode(data, max_pixels, jpeg_draft)
  } else {
    Err(ImageError::Unidentified)
  }
}

fn decode_generic(data: &[u8], format: ImageFormat, max_pixels: Option<u64>) -> Result<DynamicImage, ImageError> {
  let reader = || ImageReader::with_format(Cursor::new(data), format);
  let (width, height) = reader().into_dimensions().map_err(|_| ImageError::Unidentified)?;
  check_pixel_limit(width, height, max_pixels)?;
  let mut reader = reader();
  reader.no_limits();
  reader.decode().map_err(ImageError::decode)
}

/// Composites RGBA pixels over an opaque gray background.
fn flatten(rgba: &image::RgbaImage, background: u8) -> RgbImage {
  let background = u32::from(background);
  let pixels = rgba
    .as_raw()
    .as_chunks::<4>()
    .0
    .iter()
    .flat_map(|pixel| {
      let alpha = u32::from(pixel[3]);
      [0, 1, 2].map(|channel| ((u32::from(pixel[channel]) * alpha + background * (255 - alpha) + 127) / 255) as u8)
    })
    .collect();
  RgbImage { width: rgba.width(), height: rgba.height(), pixels }
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
      pixels: rgba
        .as_raw()
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|pixel| [0, 1, 2].map(|c| (pixel[c] >> 8) as u8))
        .collect(),
    },
    other => RgbImage { width, height, pixels: other.into_rgb8().into_raw() },
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn flattens_transparency_onto_the_background() {
    let rgba = image::RgbaImage::from_raw(2, 1, vec![255, 0, 0, 255, 255, 255, 255, 0]).unwrap();
    assert_eq!(flatten(&rgba, 0x12).pixels, [255, 0, 0, 0x12, 0x12, 0x12]);
  }

  #[test]
  fn unknown_data_is_unidentified() {
    assert!(matches!(decode(b"definitely not an image", "jpg", None, None), Err(ImageError::Unidentified)));
  }
}
