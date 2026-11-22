//! JPEG decoding with libjpeg-turbo, the decoder Pillow uses, so pixels match the Python version.

use std::ffi::CStr;

use turbojpeg_sys as tj;

use super::{ImageError, RgbImage, check_pixel_limit};

struct Decompressor(tj::tjhandle);

impl Drop for Decompressor {
  fn drop(&mut self) {
    // SAFETY: the handle came from tj3Init and is destroyed exactly once.
    unsafe { tj::tj3Destroy(self.0) }
  }
}

impl Decompressor {
  fn new() -> Result<Self, ImageError> {
    // SAFETY: tj3Init has no preconditions; a null handle is reported as an error.
    let handle = unsafe { tj::tj3Init(tj::TJINIT_TJINIT_DECOMPRESS as libc::c_int) };
    if handle.is_null() {
      return Err(ImageError::decode("cannot initialize libjpeg-turbo"));
    }
    Ok(Decompressor(handle))
  }

  fn error(&self) -> String {
    // SAFETY: the handle is valid and tj3GetErrorStr returns a NUL-terminated string.
    unsafe { CStr::from_ptr(tj::tj3GetErrorStr(self.0)) }.to_string_lossy().into_owned()
  }

  fn get(&self, parameter: tj::TJPARAM) -> i32 {
    // SAFETY: the handle is valid.
    unsafe { tj::tj3Get(self.0, parameter as libc::c_int) }
  }

  /// Decompresses into `pixels`. Like Pillow, recoverable problems (libjpeg warnings such as a
  /// truncated file) still produce an image; only fatal errors fail.
  fn decompress(&self, data: &[u8], pixels: &mut [u8], format: tj::TJPF) -> Result<(), ImageError> {
    // SAFETY: the caller sizes `pixels` for the whole image in `format`; a pitch of 0 means
    // tightly packed rows.
    let result =
      unsafe { tj::tj3Decompress8(self.0, data.as_ptr(), data.len() as tj::size_t, pixels.as_mut_ptr(), 0, format) };
    // SAFETY: the handle is valid.
    if result != 0 && unsafe { tj::tj3GetErrorCode(self.0) } != tj::TJERR_TJERR_WARNING as libc::c_int {
      return Err(ImageError::decode(self.error()));
    }
    Ok(())
  }
}

/// Decodes a JPEG. Like Pillow's `draft`, `draft` picks the largest DCT reduction (down to 1/8) that
/// keeps the image at least that large, which skips most of the decoding work.
pub fn decode(data: &[u8], max_pixels: Option<u64>, draft: Option<(u32, u32)>) -> Result<RgbImage, ImageError> {
  let decompressor = Decompressor::new()?;
  // SAFETY: the handle is valid and `data` is a readable buffer of the given length.
  if unsafe { tj::tj3DecompressHeader(decompressor.0, data.as_ptr(), data.len() as tj::size_t) } != 0 {
    return Err(ImageError::Unidentified);
  }
  let width = decompressor.get(tj::TJPARAM_TJPARAM_JPEGWIDTH) as u32;
  let height = decompressor.get(tj::TJPARAM_TJPARAM_JPEGHEIGHT) as u32;
  if width == 0 || height == 0 {
    return Err(ImageError::Unidentified);
  }
  check_pixel_limit(width, height, max_pixels)?;
  // Lossless JPEGs have no DCT to reduce.
  let reduction = match draft {
    Some((min_width, min_height)) if decompressor.get(tj::TJPARAM_TJPARAM_LOSSLESS) != 1 => {
      let scale = (width / min_width).min(height / min_height);
      [8, 4, 2].into_iter().find(|&reduction| scale >= reduction).unwrap_or(1)
    }
    _ => 1,
  };
  let factor = tj::tjscalingfactor { num: 1, denom: reduction as libc::c_int };
  // SAFETY: the handle is valid.
  if unsafe { tj::tj3SetScalingFactor(decompressor.0, factor) } != 0 {
    return Err(ImageError::decode(decompressor.error()));
  }
  let (width, height) = (width.div_ceil(reduction), height.div_ceil(reduction));
  let pixel_count = width as usize * height as usize;
  match decompressor.get(tj::TJPARAM_TJPARAM_COLORSPACE) as tj::TJCS {
    tj::TJCS_TJCS_CMYK | tj::TJCS_TJCS_YCCK => {
      let mut cmyk = vec![0u8; pixel_count * 4];
      decompressor.decompress(data, &mut cmyk, tj::TJPF_TJPF_CMYK)?;
      // Photoshop writes inverted CMYK, which libjpeg passes through as-is. Pillow reads it with
      // the "CMYK;I" raw mode whenever the Adobe marker is present, so invert it the same way.
      Ok(RgbImage { width, height, pixels: cmyk_to_rgb(&cmyk, has_adobe_marker(data)) })
    }
    tj::TJCS_TJCS_GRAY => {
      let mut luma = vec![0u8; pixel_count];
      decompressor.decompress(data, &mut luma, tj::TJPF_TJPF_GRAY)?;
      Ok(RgbImage::from_luma(width, height, &luma))
    }
    _ => {
      let mut pixels = vec![0u8; pixel_count * 3];
      decompressor.decompress(data, &mut pixels, tj::TJPF_TJPF_RGB)?;
      Ok(RgbImage { width, height, pixels })
    }
  }
}

fn has_adobe_marker(data: &[u8]) -> bool {
  // Walk the marker segments that precede the image data looking for APP14 "Adobe".
  let mut offset = 2;
  while offset + 4 <= data.len() && data[offset] == 0xFF {
    let marker = data[offset + 1];
    if marker == 0xDA {
      break;
    }
    let length = usize::from(u16::from_be_bytes([data[offset + 2], data[offset + 3]]));
    if marker == 0xEE && data[offset + 4..].starts_with(b"Adobe") {
      return true;
    }
    offset += 2 + length;
  }
  false
}

/// Pillow's `cmyk2rgb`.
fn cmyk_to_rgb(cmyk: &[u8], inverted: bool) -> Vec<u8> {
  let muldiv255 = |a: i32, b: i32| {
    let tmp = a * b + 128;
    ((tmp >> 8) + tmp) >> 8
  };
  cmyk
    .as_chunks::<4>()
    .0
    .iter()
    .flat_map(|pixel| {
      let pixel =
        [pixel[0], pixel[1], pixel[2], pixel[3]].map(|value| i32::from(if inverted { 255 - value } else { value }));
      let nk = 255 - pixel[3];
      [pixel[0], pixel[1], pixel[2]].map(|value| (nk - muldiv255(value, nk)).clamp(0, 255) as u8)
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn cmyk_conversion_matches_pillow() {
    assert_eq!(cmyk_to_rgb(&[0, 0, 0, 0], false), [255, 255, 255]);
    assert_eq!(cmyk_to_rgb(&[0, 0, 0, 255], false), [0, 0, 0]);
    assert_eq!(cmyk_to_rgb(&[255, 0, 128, 0], false), [0, 255, 127]);
    assert_eq!(cmyk_to_rgb(&[255, 255, 255, 255], true), [255, 255, 255]);
  }

  #[test]
  fn rejects_non_jpeg_data() {
    assert!(matches!(decode(b"not a jpeg", None, None), Err(ImageError::Unidentified)));
  }

  #[test]
  fn decodes_truncated_files_like_pillow() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/e2e/images/cat.jpg");
    let data = std::fs::read(path).unwrap();
    let image = decode(&data[..data.len() / 2], None, None).unwrap();
    assert_eq!((image.width, image.height), (384, 256));
  }

  #[test]
  fn drafts_the_largest_reduction_that_covers_the_requested_size() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/e2e/images/cat.jpg");
    let data = std::fs::read(path).unwrap();
    let size = |draft| {
      let image = decode(&data, None, Some(draft)).unwrap();
      assert_eq!(image.pixels.len(), (image.width * image.height * 3) as usize);
      (image.width, image.height)
    };
    assert_eq!(size((256, 256)), (384, 256));
    assert_eq!(size((128, 128)), (192, 128));
    assert_eq!(size((90, 60)), (96, 64));
    assert_eq!(size((1, 1)), (48, 32));
  }
}
