//! HEIF decoding through macOS Image I/O and Core Graphics.

use std::ffi::c_void;

use objc2_core_foundation::{
  CFBoolean, CFData, CFDictionary, CFNumber, CFRetained, CFString, CFType, CGPoint, CGRect, CGSize,
};
use objc2_core_graphics::{
  CGBitmapContextCreate, CGColorSpace, CGColorSpaceModel, CGContext, CGImage, CGImageAlphaInfo,
};
use objc2_image_io::{
  CGImageSource, kCGImagePropertyOrientation, kCGImageSourceCreateThumbnailFromImageAlways,
  kCGImageSourceThumbnailMaxPixelSize,
};

use image::metadata::Orientation;

use crate::imaging::{ImageError, RgbImage, check_pixel_limit};

/// Decodes an HEIF image. `min_side` lets Image I/O downscale while decoding, as long as the shortest
/// side stays at least that long.
pub fn decode(data: &[u8], max_pixels: Option<u64>, min_side: Option<u32>) -> Result<RgbImage, ImageError> {
  let data = CFData::from_bytes(data);
  // SAFETY: no options dictionary is passed.
  let source = unsafe { CGImageSource::with_data(&data, None) }
    .ok_or_else(|| ImageError::decode("macOS Image I/O does not recognize this HEIF image"))?;
  // SAFETY: the source is valid and no options dictionary is passed.
  let image = unsafe { source.image_at_index(0, None) }
    .ok_or_else(|| ImageError::decode("macOS has no codec capable of decoding this HEIF image"))?;
  let (width, height) = (CGImage::width(Some(&image)), CGImage::height(Some(&image)));
  if width == 0 || height == 0 {
    return Err(ImageError::decode("macOS returned invalid HEIF image dimensions"));
  }
  check_pixel_limit(width as u32, height as u32, max_pixels)?;
  // A downscaled frame avoids decoding and converting the full image.
  let image = min_side.and_then(|min_side| thumbnail(&source, width, height, min_side as usize)).unwrap_or(image);
  let (width, height) = (CGImage::width(Some(&image)), CGImage::height(Some(&image)));
  let pixels = render_rgb(&image, width, height)?;
  Ok(RgbImage { width: width as u32, height: height as u32, pixels }.oriented(orientation(&source)))
}

/// Asks Image I/O for the smallest frame whose shortest side still covers `min_side`. Returns `None`
/// when Image I/O cannot produce one, so the caller falls back to the full frame.
fn thumbnail(source: &CGImageSource, width: usize, height: usize, min_side: usize) -> Option<CFRetained<CGImage>> {
  let max_pixel_size = CFNumber::new_i64((min_side * width.max(height) / width.min(height) + 1) as i64);
  // SAFETY: the keys are constant CFStrings provided by Image I/O.
  let keys = unsafe { [kCGImageSourceCreateThumbnailFromImageAlways, kCGImageSourceThumbnailMaxPixelSize] };
  let values: [&CFType; 2] = [CFBoolean::new(true), &max_pixel_size];
  let options = CFDictionary::<CFString, CFType>::from_slices(&keys, &values);
  // SAFETY: the source is valid and the options hold the value types Image I/O expects for these keys.
  unsafe { source.thumbnail_at_index(0, Some(options.as_opaque())) }
}

/// Draws the image into an 8-bit RGBX bitmap. The bitmap uses the image's own RGB color space so
/// that, like the Python version (which re-encodes the image as PNG), no color conversion happens.
fn render_rgb(image: &CGImage, width: usize, height: usize) -> Result<Vec<u8>, ImageError> {
  let image_space = CGImage::color_space(Some(image));
  let space = match image_space {
    Some(space) if CGColorSpace::model(Some(&space)) == CGColorSpaceModel::RGB => space,
    _ => CGColorSpace::new_device_rgb().ok_or_else(|| ImageError::decode("macOS could not create a color space"))?,
  };
  let mut rgbx = vec![0u8; width * height * 4];
  // SAFETY: `rgbx` holds `height` rows of `width * 4` bytes and outlives the context.
  let context: CFRetained<CGContext> = unsafe {
    CGBitmapContextCreate(
      rgbx.as_mut_ptr().cast::<c_void>(),
      width,
      height,
      8,
      width * 4,
      Some(&space),
      CGImageAlphaInfo::NoneSkipLast.0,
    )
  }
  .ok_or_else(|| ImageError::decode("macOS could not allocate an HEIF conversion buffer"))?;
  let rect = CGRect { origin: CGPoint { x: 0.0, y: 0.0 }, size: CGSize { width: width as f64, height: height as f64 } };
  CGContext::draw_image(Some(&context), rect, Some(image));
  drop(context);
  Ok(rgbx.as_chunks::<4>().0.iter().flat_map(|pixel| [pixel[0], pixel[1], pixel[2]]).collect())
}

fn orientation(source: &CGImageSource) -> Orientation {
  // SAFETY: the source is valid and no options dictionary is passed.
  let Some(properties) = (unsafe { source.properties_at_index(0, None) }) else {
    return Orientation::NoTransforms;
  };
  // SAFETY: kCGImagePropertyOrientation is a constant CFString key provided by Image I/O.
  let key: *const c_void = unsafe { (kCGImagePropertyOrientation as *const objc2_core_foundation::CFString).cast() };
  // SAFETY: the dictionary is valid; a missing key yields null.
  let value = unsafe { properties.value(key) };
  if value.is_null() {
    return Orientation::NoTransforms;
  }
  // SAFETY: Image I/O stores the orientation as a CFNumber.
  let number = unsafe { &*value.cast::<CFNumber>() };
  number
    .as_i32()
    .and_then(|value| Orientation::from_exif(u8::try_from(value).ok()?))
    .unwrap_or(Orientation::NoTransforms)
}
