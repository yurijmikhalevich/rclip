//! RAW decoding through LibRaw, mirroring what the Python version does with rawpy.

use super::preprocess::IMAGE_SIZE;
use super::{ImageError, RgbImage};

#[cfg(not(target_env = "msvc"))]
pub fn decode(data: &[u8], max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
  let raw = LibRaw::open(data)?;
  // The embedded JPEG preview is usually large enough for the model input and far cheaper to
  // decode than demosaicing the whole sensor, so prefer it and fall back to processing otherwise.
  if let Some(jpeg) = raw.jpeg_thumbnail() {
    let preview = super::decode(&jpeg, "jpg", max_pixels)?;
    if preview.width.min(preview.height) >= IMAGE_SIZE {
      return Ok(preview);
    }
  }
  raw.process_half_size()
}

#[cfg(target_env = "msvc")]
pub fn decode(_data: &[u8], _max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
  Err(ImageError::decode("RAW images are not supported by this build of rclip"))
}

#[cfg(not(target_env = "msvc"))]
struct LibRaw(*mut rsraw_sys::libraw_data_t);

#[cfg(not(target_env = "msvc"))]
impl LibRaw {
  fn open(data: &[u8]) -> Result<Self, ImageError> {
    // SAFETY: libraw_init allocates a fresh handle; it is freed in Drop.
    let handle = unsafe { rsraw_sys::libraw_init(0) };
    if handle.is_null() {
      return Err(ImageError::decode("LibRaw could not be initialized"));
    }
    let raw = LibRaw(handle);
    // SAFETY: `data` outlives `raw`, which only reads from it while this function's caller holds it.
    let code = unsafe { rsraw_sys::libraw_open_buffer(raw.0, data.as_ptr().cast(), data.len()) };
    if code != 0 {
      return Err(ImageError::Unidentified);
    }
    Ok(raw)
  }

  fn jpeg_thumbnail(&self) -> Option<Vec<u8>> {
    // SAFETY: the handle is valid and opened.
    if unsafe { rsraw_sys::libraw_unpack_thumb(self.0) } != 0 {
      return None;
    }
    let image = self.make_image(rsraw_sys::libraw_dcraw_make_mem_thumb).ok()?;
    (image.kind == rsraw_sys::LibRaw_image_formats_LIBRAW_IMAGE_JPEG).then_some(image.data)
  }

  /// Equivalent to rawpy's `postprocess(half_size=True)` with its default parameters.
  fn process_half_size(&self) -> Result<RgbImage, ImageError> {
    // SAFETY: the handle is valid; the parameters are plain values inside it.
    unsafe {
      let params = &mut (*self.0).params;
      params.half_size = 1;
      // rawpy passes gamma as (1 / 2.222, 4.5) and enables exposure correction with a 1.0 shift.
      params.gamm[0] = 1.0 / 2.222;
      params.gamm[1] = 4.5;
      params.exp_correc = -1;
      params.exp_shift = 1.0;
      check(rsraw_sys::libraw_unpack(self.0))?;
      check(rsraw_sys::libraw_dcraw_process(self.0))?;
    }
    let image = self.make_image(rsraw_sys::libraw_dcraw_make_mem_image)?;
    if image.colors != 3 || image.bits != 8 {
      return Err(ImageError::decode("LibRaw produced an unexpected pixel format"));
    }
    Ok(RgbImage { width: image.width, height: image.height, pixels: image.data })
  }

  fn make_image(
    &self,
    make: unsafe extern "C" fn(
      *mut rsraw_sys::libraw_data_t,
      *mut libc::c_int,
    ) -> *mut rsraw_sys::libraw_processed_image_t,
  ) -> Result<ProcessedImage, ImageError> {
    let mut code = 0;
    // SAFETY: the handle is valid; the returned image is copied and then released.
    unsafe {
      let image = make(self.0, &mut code);
      if image.is_null() {
        check(code)?;
        return Err(ImageError::decode("LibRaw could not produce an image"));
      }
      let processed = ProcessedImage {
        kind: (*image).type_,
        width: u32::from((*image).width),
        height: u32::from((*image).height),
        colors: (*image).colors,
        bits: (*image).bits,
        data: std::slice::from_raw_parts((*image).data.as_ptr(), (*image).data_size as usize).to_vec(),
      };
      rsraw_sys::libraw_dcraw_clear_mem(image);
      Ok(processed)
    }
  }
}

#[cfg(not(target_env = "msvc"))]
impl Drop for LibRaw {
  fn drop(&mut self) {
    // SAFETY: the handle was created by libraw_init and is closed exactly once.
    unsafe { rsraw_sys::libraw_close(self.0) }
  }
}

#[cfg(not(target_env = "msvc"))]
struct ProcessedImage {
  kind: rsraw_sys::LibRaw_image_formats,
  width: u32,
  height: u32,
  colors: u16,
  bits: u16,
  data: Vec<u8>,
}

#[cfg(not(target_env = "msvc"))]
fn check(code: libc::c_int) -> Result<(), ImageError> {
  if code == 0 {
    return Ok(());
  }
  // SAFETY: libraw_strerror returns a pointer to a static string.
  let message = unsafe { std::ffi::CStr::from_ptr(rsraw_sys::libraw_strerror(code)) };
  Err(ImageError::decode(message.to_string_lossy()))
}
