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
/// as the shortest side stays at least that long.
pub fn decode(
  data: &[u8],
  extension: &str,
  max_pixels: Option<u64>,
  min_side: Option<u32>,
) -> Result<RgbImage, ImageError> {
  match generic_format(data, extension) {
    Some(format) => Ok(to_rgb(decode_generic(data, format, max_pixels)?)),
    None => decode_special(data, extension, max_pixels, min_side),
  }
}

/// Decodes an image for display, flattening transparency onto `background` the way the Python
/// version's interactive UI does instead of dropping the alpha channel. `min_side` works like in [`decode`].
pub fn decode_flattened(
  data: &[u8],
  extension: &str,
  background: u8,
  min_side: Option<u32>,
) -> Result<RgbImage, ImageError> {
  match generic_format(data, extension) {
    Some(format) => {
      let image = decode_generic(data, format, None)?;
      if image.color().has_alpha() { Ok(flatten(&image.into_rgba8(), background)) } else { Ok(to_rgb(image)) }
    }
    None => decode_special(data, extension, None, min_side),
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
  min_side: Option<u32>,
) -> Result<RgbImage, ImageError> {
  if IMAGE_RAW_EXT.contains(&extension) {
    raw::decode(data, max_pixels, min_side)
  } else if extension == "heic" && heif::is_supported() {
    heif::decode(data, max_pixels, min_side)
  } else if jp2::is_jp2(data) {
    jp2::decode(data, max_pixels)
  } else if image::guess_format(data).is_ok_and(|format| format == ImageFormat::Jpeg) {
    jpeg::decode(data, max_pixels, min_side)
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
    // Pillow opens 16-bit grayscale images in mode I;16, whose conversion to RGB clamps samples to 255.
    DynamicImage::ImageLuma16(luma) => {
      let luma: Vec<u8> = luma.as_raw().iter().map(|&value| value.min(255) as u8).collect();
      RgbImage::from_luma(width, height, &luma)
    }
    // Pillow opens other 16-bit images in 8-bit modes by keeping the most significant byte.
    DynamicImage::ImageLumaA16(luma_alpha) => {
      let luma: Vec<u8> = luma_alpha.as_raw().as_chunks::<2>().0.iter().map(|pixel| (pixel[0] >> 8) as u8).collect();
      RgbImage::from_luma(width, height, &luma)
    }
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
  use crate::imaging::preprocess::{IMAGE_SIZE, preprocess};

  #[test]
  fn flattens_transparency_onto_the_background() {
    let rgba = image::RgbaImage::from_raw(2, 1, vec![255, 0, 0, 255, 255, 255, 255, 0]).unwrap();
    assert_eq!(flatten(&rgba, 0x12).pixels, [255, 0, 0, 0x12, 0x12, 0x12]);
  }

  #[test]
  fn converts_16_bit_grayscale_like_pillow() {
    let samples = [0u16, 1, 255, 256, 257, 1000, 32768, 65535];
    let encode = |image: DynamicImage| {
      let mut data = Vec::new();
      image.write_to(&mut Cursor::new(&mut data), ImageFormat::Png).unwrap();
      decode(&data, "png", None, None)
        .unwrap()
        .pixels
        .as_chunks::<3>()
        .0
        .iter()
        .map(|pixel| pixel[0])
        .collect::<Vec<_>>()
    };
    let luma = image::ImageBuffer::from_raw(8, 1, samples.to_vec()).unwrap();
    assert_eq!(encode(DynamicImage::ImageLuma16(luma)), [0, 1, 255, 255, 255, 255, 255, 255]);
    let luma_alpha =
      image::ImageBuffer::from_raw(8, 1, samples.iter().flat_map(|&value| [value, 65535]).collect()).unwrap();
    assert_eq!(encode(DynamicImage::ImageLumaA16(luma_alpha)), [0, 0, 0, 1, 1, 3, 128, 255]);
  }

  #[test]
  fn unknown_data_is_unidentified() {
    assert!(matches!(decode(b"definitely not an image", "jpg", None, None), Err(ImageError::Unidentified)));
  }

  #[cfg(not(any(target_os = "macos", windows)))]
  #[test]
  fn heic_images_are_unidentified_without_an_operating_system_codec() {
    let data = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/e2e/images heic/bee.heic")).unwrap();
    assert!(matches!(decode(&data, "heic", None, None), Err(ImageError::Unidentified)));
  }

  /// A smooth gradient, which JPEG compresses with little loss, in color or in grayscale.
  fn gradient(width: u32, height: u32, color: bool, format: ImageFormat) -> Vec<u8> {
    let image = DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |x, y| {
      let value = ((x * 255 / width + y * 255 / height) / 2) as u8;
      image::Rgb([value, 255 - value, value / 2 + 64])
    }));
    let image = if color { image } else { image.grayscale() };
    let mut data = Vec::new();
    image.write_to(&mut Cursor::new(&mut data), format).unwrap();
    data
  }

  fn decoded_size(data: &[u8], min_side: Option<u32>) -> (u32, u32) {
    let image = decode(data, "", None, min_side).unwrap();
    assert_eq!(image.pixels.len(), (image.width * image.height * 3) as usize);
    (image.width, image.height)
  }

  #[test]
  fn decodes_large_jpegs_at_the_largest_reduction_that_covers_the_minimum_side() {
    let color = gradient(4096, 3072, true, ImageFormat::Jpeg);
    assert_eq!(decoded_size(&color, Some(IMAGE_SIZE)), (512, 384));
    assert_eq!(decoded_size(&color, None), (4096, 3072));
    let gray = gradient(2048, 1536, false, ImageFormat::Jpeg);
    assert_eq!(decoded_size(&gray, Some(IMAGE_SIZE)), (512, 384));
  }

  #[test]
  fn decodes_small_jpegs_and_other_formats_at_full_size() {
    assert_eq!(decoded_size(&gradient(400, 300, true, ImageFormat::Jpeg), Some(IMAGE_SIZE)), (400, 300));
    assert_eq!(decoded_size(&gradient(1024, 768, true, ImageFormat::Png), Some(IMAGE_SIZE)), (1024, 768));
  }

  #[test]
  fn reduced_jpeg_decoding_stays_close_to_the_full_resolution_result() {
    let data = gradient(4096, 3072, true, ImageFormat::Jpeg);
    let reduced = preprocess(&decode(&data, "", None, Some(IMAGE_SIZE)).unwrap());
    let full = preprocess(&decode(&data, "", None, None).unwrap());
    assert!(reduced.iter().zip(&full).all(|(reduced, full)| (reduced - full).abs() <= 0.2));
  }

  #[test]
  fn rejects_images_over_the_pixel_limit() {
    for format in [ImageFormat::Png, ImageFormat::Jpeg] {
      let data = gradient(50, 50, true, format);
      assert!(matches!(decode(&data, "", Some(100), None), Err(ImageError::TooLarge { pixels: 2500, limit: 100 })));
      assert!(decode(&data, "", Some(2500), None).is_ok());
      assert!(decode(&data, "", None, None).is_ok());
    }
  }
}
