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
pub fn decode(_data: &[u8], _max_pixels: Option<u64>) -> Result<super::RgbImage, super::ImageError> {
  Err(super::ImageError::decode("native HEIF decoding is available only on macOS and Windows"))
}
