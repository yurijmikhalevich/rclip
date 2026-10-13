//! CLIP image preprocessing for ViT-B-32-256/datacomp_s34b_b86k, matching open_clip's pipeline:
//! Resize(256, bicubic) → CenterCrop(256) → RGB float32 [0, 1] → Normalize(CLIP mean/std) → CHW.

use super::RgbImage;
use super::resample::{Filter, resize};

pub const IMAGE_SIZE: u32 = 256;
/// Length of one preprocessed image tensor (3x256x256).
pub const TENSOR_LEN: usize = 3 * (IMAGE_SIZE * IMAGE_SIZE) as usize;

const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// Preprocesses an image into a normalized `[3, 256, 256]` tensor, appending it to `output`.
pub fn preprocess_into(image: &RgbImage, output: &mut Vec<f32>) {
  let (width, height) = (image.width as f64, image.height as f64);
  let size = f64::from(IMAGE_SIZE);
  // The shorter side becomes 256; the longer one is truncated like Python's int().
  let (resized_width, resized_height) = if width < height {
    (IMAGE_SIZE, (height * size / width) as u32)
  } else {
    ((width * size / height) as u32, IMAGE_SIZE)
  };
  let resized = resize(image, resized_width, resized_height, Filter::Bicubic);
  // Python's round() rounds halves to even.
  let left = ((f64::from(resized_width) - size) / 2.0).round_ties_even() as u32;
  let top = ((f64::from(resized_height) - size) / 2.0).round_ties_even() as u32;

  let start = output.len();
  output.resize(start + TENSOR_LEN, 0.0);
  let planes = &mut output[start..];
  let plane_len = (IMAGE_SIZE * IMAGE_SIZE) as usize;
  let stride = resized.width as usize * 3;
  for y in 0..IMAGE_SIZE as usize {
    let row_start = (top as usize + y) * stride + left as usize * 3;
    let row = &resized.pixels[row_start..row_start + IMAGE_SIZE as usize * 3];
    for (x, pixel) in row.chunks_exact(3).enumerate() {
      for channel in 0..3 {
        let value = f32::from(pixel[channel]) / 255.0;
        planes[channel * plane_len + y * IMAGE_SIZE as usize + x] = (value - MEAN[channel]) / STD[channel];
      }
    }
  }
}

pub fn preprocess(image: &RgbImage) -> Vec<f32> {
  let mut output = Vec::with_capacity(TENSOR_LEN);
  preprocess_into(image, &mut output);
  output
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn produces_a_normalized_chw_tensor() {
    let image = RgbImage { width: 300, height: 200, pixels: [255, 0, 128].repeat(300 * 200) };
    let tensor = preprocess(&image);
    assert_eq!(tensor.len(), TENSOR_LEN);
    let plane = (IMAGE_SIZE * IMAGE_SIZE) as usize;
    assert!((tensor[0] - (1.0 - MEAN[0]) / STD[0]).abs() < 1e-6);
    assert!((tensor[plane] - (0.0 - MEAN[1]) / STD[1]).abs() < 1e-6);
    assert!((tensor[2 * plane + 5] - (128.0 / 255.0 - MEAN[2]) / STD[2]).abs() < 1e-6);
  }

  #[test]
  fn upscales_small_images() {
    let image = RgbImage { width: 10, height: 20, pixels: vec![7; 10 * 20 * 3] };
    assert_eq!(preprocess(&image).len(), TENSOR_LEN);
  }
}
