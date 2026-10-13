//! RAW decoding through LibRaw, mirroring what the Python version does with rawpy.

use std::ffi::{CStr, c_char, c_int, c_uint, c_ushort, c_void};

use super::preprocess::IMAGE_SIZE;
use super::{ImageError, RgbImage};

// LibRaw decodes deflate-compressed DNGs with zlib; referencing the crate makes sure it is linked.
use libz_sys as _;

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

/// LibRaw's `libraw_data_t`, used only through pointers.
#[repr(C)]
struct LibRawData {
  _private: [u8; 0],
}

/// LibRaw's `libraw_processed_image_t`.
#[repr(C)]
struct ProcessedImage {
  kind: c_int,
  height: c_ushort,
  width: c_ushort,
  colors: c_ushort,
  bits: c_ushort,
  data_size: c_uint,
  data: [u8; 1],
}

const LIBRAW_IMAGE_JPEG: c_int = 1;

unsafe extern "C" {
  fn libraw_init(flags: c_uint) -> *mut LibRawData;
  fn libraw_open_buffer(data: *mut LibRawData, buffer: *const c_void, size: usize) -> c_int;
  fn libraw_unpack_thumb(data: *mut LibRawData) -> c_int;
  fn libraw_unpack(data: *mut LibRawData) -> c_int;
  fn libraw_dcraw_process(data: *mut LibRawData) -> c_int;
  fn libraw_dcraw_make_mem_thumb(data: *mut LibRawData, error: *mut c_int) -> *mut ProcessedImage;
  fn libraw_dcraw_make_mem_image(data: *mut LibRawData, error: *mut c_int) -> *mut ProcessedImage;
  fn libraw_dcraw_clear_mem(image: *mut ProcessedImage);
  fn libraw_close(data: *mut LibRawData);
  fn libraw_strerror(error: c_int) -> *const c_char;
  /// Defined in `raw_params.cpp`.
  fn rclip_libraw_use_half_size(data: *mut LibRawData);
}

struct LibRaw(*mut LibRawData);

impl LibRaw {
  fn open(data: &[u8]) -> Result<Self, ImageError> {
    // SAFETY: libraw_init allocates a fresh handle; it is freed in Drop.
    let handle = unsafe { libraw_init(0) };
    if handle.is_null() {
      return Err(ImageError::decode("LibRaw could not be initialized"));
    }
    let raw = LibRaw(handle);
    // SAFETY: LibRaw reads from `data` only while the handle is used, which `decode` keeps within
    // the lifetime of `data`.
    if unsafe { libraw_open_buffer(raw.0, data.as_ptr().cast(), data.len()) } != 0 {
      return Err(ImageError::Unidentified);
    }
    Ok(raw)
  }

  fn jpeg_thumbnail(&self) -> Option<Vec<u8>> {
    // SAFETY: the handle is valid and opened.
    if unsafe { libraw_unpack_thumb(self.0) } != 0 {
      return None;
    }
    let (kind, image) = self.take_image(libraw_dcraw_make_mem_thumb).ok()?;
    (kind == LIBRAW_IMAGE_JPEG).then_some(image.pixels)
  }

  /// Equivalent to rawpy's `postprocess(half_size=True)` with its default parameters.
  fn process_half_size(&self) -> Result<RgbImage, ImageError> {
    // SAFETY: the handle is valid and opened.
    unsafe {
      rclip_libraw_use_half_size(self.0);
      check(libraw_unpack(self.0))?;
      check(libraw_dcraw_process(self.0))?;
    }
    let (_, image) = self.take_image(libraw_dcraw_make_mem_image)?;
    if image.pixels.len() != image.width as usize * image.height as usize * 3 {
      return Err(ImageError::decode("LibRaw produced an unexpected pixel format"));
    }
    Ok(image)
  }

  /// Copies an image produced by LibRaw and frees LibRaw's copy. JPEG thumbnails are returned as
  /// their encoded bytes in `pixels`.
  fn take_image(
    &self,
    make: unsafe extern "C" fn(*mut LibRawData, *mut c_int) -> *mut ProcessedImage,
  ) -> Result<(c_int, RgbImage), ImageError> {
    let mut error = 0;
    // SAFETY: the handle is valid; the returned image is copied before being released.
    unsafe {
      let image = make(self.0, &mut error);
      if image.is_null() {
        check(error)?;
        return Err(ImageError::decode("LibRaw could not produce an image"));
      }
      let bytes = std::slice::from_raw_parts((*image).data.as_ptr(), (*image).data_size as usize).to_vec();
      let result = if (*image).kind != LIBRAW_IMAGE_JPEG && ((*image).colors != 3 || (*image).bits != 8) {
        Err(ImageError::decode("LibRaw produced an unexpected pixel format"))
      } else {
        let (width, height) = (u32::from((*image).width), u32::from((*image).height));
        Ok(((*image).kind, RgbImage { width, height, pixels: bytes }))
      };
      libraw_dcraw_clear_mem(image);
      result
    }
  }
}

impl Drop for LibRaw {
  fn drop(&mut self) {
    // SAFETY: the handle was created by libraw_init and is closed exactly once.
    unsafe { libraw_close(self.0) }
  }
}

fn check(code: c_int) -> Result<(), ImageError> {
  if code == 0 {
    return Ok(());
  }
  // SAFETY: libraw_strerror returns a pointer to a static string.
  let message = unsafe { CStr::from_ptr(libraw_strerror(code)) };
  Err(ImageError::decode(message.to_string_lossy()))
}
