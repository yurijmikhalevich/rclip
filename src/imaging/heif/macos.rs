//! HEIF decoding through macOS Image I/O and Core Graphics.

use std::ffi::c_void;

use objc2_core_foundation::{CFData, CFNumber, CFRetained, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{
  CGBitmapContextCreate, CGColorSpace, CGColorSpaceModel, CGContext, CGImage, CGImageAlphaInfo,
};
use objc2_image_io::{CGImageSource, kCGImagePropertyOrientation};

use crate::imaging::{ImageError, RgbImage, check_pixel_limit};

pub fn decode(data: &[u8], max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
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
  let pixels = render_rgb(&image, width, height)?;
  Ok(RgbImage { width: width as u32, height: height as u32, pixels }.oriented(orientation(&source)))
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

fn orientation(source: &CGImageSource) -> u32 {
  // SAFETY: the source is valid and no options dictionary is passed.
  let Some(properties) = (unsafe { source.properties_at_index(0, None) }) else {
    return 1;
  };
  // SAFETY: kCGImagePropertyOrientation is a constant CFString key provided by Image I/O.
  let key: *const c_void = unsafe { (kCGImagePropertyOrientation as *const objc2_core_foundation::CFString).cast() };
  // SAFETY: the dictionary is valid; a missing key yields null.
  let value = unsafe { properties.value(key) };
  if value.is_null() {
    return 1;
  }
  // SAFETY: Image I/O stores the orientation as a CFNumber.
  let number = unsafe { &*value.cast::<CFNumber>() };
  number.as_i32().map_or(1, |value| value as u32)
}
