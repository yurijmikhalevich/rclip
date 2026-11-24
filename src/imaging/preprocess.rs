//! CLIP image preprocessing for ViT-B-32-256/datacomp_s34b_b86k, matching open_clip's pipeline:
//! Resize(256, bicubic) → CenterCrop(256) → RGB float32 [0, 1] → Normalize(CLIP mean/std) → CHW.

use super::RgbImage;
use super::resample::{Filter, resize_cropped};

pub const IMAGE_SIZE: u32 = 256;
/// Length of one preprocessed image tensor (3x256x256).
pub const TENSOR_LEN: usize = 3 * (IMAGE_SIZE * IMAGE_SIZE) as usize;

const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

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
  // Python's round() rounds halves to even.
  let left = ((f64::from(resized_width) - size) / 2.0).round_ties_even() as u32;
  let top = ((f64::from(resized_height) - size) / 2.0).round_ties_even() as u32;
  // Only the center crop is resized, so that images with extreme aspect ratios do not need memory
  // for their whole resized length.
  let crop = (left, top, IMAGE_SIZE, IMAGE_SIZE);
  let cropped = resize_cropped(image, resized_width, resized_height, crop, Filter::Bicubic);

  let start = output.len();
  output.resize(start + TENSOR_LEN, 0.0);
  let planes = &mut output[start..];
  let plane_len = (IMAGE_SIZE * IMAGE_SIZE) as usize;
  for (index, pixel) in cropped.pixels.as_chunks::<3>().0.iter().enumerate() {
    for channel in 0..3 {
      let value = f32::from(pixel[channel]) / 255.0;
      planes[channel * plane_len + index] = (value - MEAN[channel]) / STD[channel];
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
  fn returns_the_model_input_for_any_size_and_aspect_ratio() {
    for (width, height) in [(512, 384), (384, 512), (1024, 768), (256, 256), (200, 100), (10, 20), (1000, 333)] {
      let image = RgbImage { width, height, pixels: vec![7; (width * height * 3) as usize] };
      assert_eq!(preprocess(&image).len(), TENSOR_LEN, "{width}x{height}");
    }
  }

  #[test]
  fn crops_the_center_rounding_half_pixel_offsets_to_even_like_python() {
    // Red holds the column and green holds the row, so the first value of a plane shows where the
    // crop starts. A shorter side of 256 means that the image is cropped without resizing.
    let coordinates = |width: u32, height: u32| RgbImage {
      width,
      height,
      pixels: (0..height).flat_map(|y| (0..width).flat_map(move |x| [x as u8, y as u8, 0])).collect(),
    };
    let normalized = |value: u8, channel: usize| (f32::from(value) / 255.0 - MEAN[channel]) / STD[channel];
    for (width, left) in [(384, 64), (257, 0), (259, 2), (261, 2)] {
      let tensor = preprocess(&coordinates(width, 256));
      assert!((tensor[0] - normalized(left, 0)).abs() < 1e-6, "{width}");
    }
    let tensor = preprocess(&coordinates(256, 384));
    assert!((tensor[(IMAGE_SIZE * IMAGE_SIZE) as usize] - normalized(64, 1)).abs() < 1e-6);
  }

  #[test]
  fn handles_extreme_aspect_ratios_without_resizing_the_whole_image() {
    // Resized whole, these would be 256x25,600,000 and 25,600,000x256 pixels, about 20 GB each.
    for (width, height) in [(1, 100_000), (100_000, 1)] {
      let image = RgbImage { width, height, pixels: vec![7; (width * height * 3) as usize] };
      let tensor = preprocess(&image);
      assert!((tensor[0] - (7.0 / 255.0 - MEAN[0]) / STD[0]).abs() < 1e-6, "{width}x{height}");
    }
  }
}
