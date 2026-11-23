//! A bit-exact port of Pillow's convolution resampler for 8-bit RGB images.
//!
//! The embeddings stored in the index depend on the exact pixels fed to the model, so the
//! resampler reproduces Pillow's `Image.resize` arithmetic (filter support, coefficient
//! normalization, 22-bit fixed-point accumulation, and horizontal-then-vertical pass order)
//! instead of using a faster but numerically different SIMD resizer. The loops are laid out so
//! that the compiler can vectorize them without changing the integer results.

use std::ops::Range;

use super::RgbImage;

const PRECISION_BITS: u32 = 32 - 8 - 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
  /// Pillow's `BICUBIC` (Keys cubic with a = -0.5).
  Bicubic,
  /// Pillow's `LANCZOS` (Lanczos-3).
  Lanczos,
}

impl Filter {
  fn support(self) -> f64 {
    match self {
      Filter::Bicubic => 2.0,
      Filter::Lanczos => 3.0,
    }
  }

  fn weight(self, x: f64) -> f64 {
    match self {
      Filter::Bicubic => {
        const A: f64 = -0.5;
        let x = x.abs();
        if x < 1.0 {
          ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
          (((x - 5.0) * x + 8.0) * x - 4.0) * A
        } else {
          0.0
        }
      }
      Filter::Lanczos => {
        if (-3.0..3.0).contains(&x) {
          sinc(x) * sinc(x / 3.0)
        } else {
          0.0
        }
      }
    }
  }
}

fn sinc(x: f64) -> f64 {
  if x == 0.0 {
    return 1.0;
  }
  let x = x * std::f64::consts::PI;
  x.sin() / x
}

/// Fixed-point convolution coefficients for one axis.
struct Coefficients {
  /// Maximum number of taps per output sample; `weights` holds `ksize` values per output sample.
  ksize: usize,
  /// `(first input index, number of taps)` for each output sample.
  bounds: Vec<(usize, usize)>,
  weights: Vec<i32>,
}

impl Coefficients {
  /// The coefficients of the output samples in `window` when resizing from `in_size` to `out_size`.
  fn new(in_size: usize, out_size: usize, window: Range<usize>, filter: Filter) -> Self {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(window.len());
    let mut weights = vec![0i32; window.len() * ksize];
    let mut taps = vec![0f64; ksize];
    for (xx, row) in window.zip(weights.chunks_exact_mut(ksize)) {
      let center = (xx as f64 + 0.5) * scale;
      let ss = 1.0 / filterscale;
      // Pillow truncates toward zero after adding 0.5, which is what `as` does here.
      let xmin = ((center - support + 0.5) as i64).max(0) as usize;
      let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
      let mut total = 0.0;
      for (x, tap) in taps[..xmax].iter_mut().enumerate() {
        *tap = filter.weight((x as f64 + xmin as f64 - center + 0.5) * ss);
        total += *tap;
      }
      for (weight, &tap) in row.iter_mut().zip(&taps[..xmax]) {
        let tap = if total != 0.0 { tap / total } else { tap };
        let fixed = tap * f64::from(1u32 << PRECISION_BITS);
        *weight = if tap < 0.0 { (fixed - 0.5) as i32 } else { (fixed + 0.5) as i32 };
      }
      bounds.push((xmin, xmax));
    }
    Self { ksize, bounds, weights }
  }

  fn weights(&self, index: usize) -> &[i32] {
    let (_, taps) = self.bounds[index];
    &self.weights[index * self.ksize..index * self.ksize + taps]
  }
}

#[inline(always)]
fn clip8(value: i32) -> u8 {
  (value >> PRECISION_BITS).clamp(0, 255) as u8
}

const ROUNDING: i32 = 1 << (PRECISION_BITS - 1);

/// Resizes `image` to `width`x`height` exactly as Pillow's `Image.resize` does for RGB images.
pub fn resize(image: &RgbImage, width: u32, height: u32, filter: Filter) -> RgbImage {
  resize_cropped(image, width, height, (0, 0, width, height), filter)
}

/// Like [`resize`] followed by cropping the result to `crop` (left, top, width, height), but
/// computes only the cropped pixels, so memory use does not grow with what is cropped away.
pub fn resize_cropped(
  image: &RgbImage,
  width: u32,
  height: u32,
  crop: (u32, u32, u32, u32),
  filter: Filter,
) -> RgbImage {
  let (in_width, in_height) = (image.width as usize, image.height as usize);
  let (left, top, out_width, out_height) = (crop.0 as usize, crop.1 as usize, crop.2 as usize, crop.3 as usize);
  // Pillow skips the pass along an axis whose size does not change. Such a pass would leave the
  // pixels as they are (its weights are exactly 1 and 0), so it also runs to crop that axis.
  let horizontal_pass = width as usize != in_width || out_width != in_width;
  let vertical_pass = height as usize != in_height || out_height != in_height;
  if !horizontal_pass && !vertical_pass {
    return image.clone();
  }
  let horizontal = Coefficients::new(in_width, width as usize, left..left + out_width, filter);
  let mut vertical = Coefficients::new(in_height, height as usize, top..top + out_height, filter);

  let mut source: std::borrow::Cow<'_, [u8]> = std::borrow::Cow::Borrowed(&image.pixels);
  let mut source_width = in_width;
  if horizontal_pass {
    // Only the rows the vertical pass reads are resampled horizontally.
    let first_row = vertical.bounds[0].0;
    let (last_start, last_taps) = vertical.bounds[out_height - 1];
    let last_row = last_start + last_taps;
    for bound in &mut vertical.bounds {
      bound.0 -= first_row;
    }
    source = std::borrow::Cow::Owned(resample_horizontal(
      &image.pixels[first_row * in_width * 3..last_row * in_width * 3],
      in_width,
      &horizontal,
    ));
    source_width = out_width;
  }
  let pixels = if vertical_pass { resample_vertical(&source, source_width, &vertical) } else { source.into_owned() };
  RgbImage { width: crop.2, height: crop.3, pixels }
}

fn resample_horizontal(rows: &[u8], in_width: usize, coefficients: &Coefficients) -> Vec<u8> {
  let out_width = coefficients.bounds.len();
  let row_count = rows.len() / (in_width * 3);
  let mut output = vec![0u8; out_width * 3 * row_count];
  // Deinterleaving each row into planes lets every tap loop run over contiguous memory.
  let mut planes = [vec![0i32; in_width], vec![0i32; in_width], vec![0i32; in_width]];
  for (row, out_row) in rows.chunks_exact(in_width * 3).zip(output.chunks_exact_mut(out_width * 3)) {
    for (x, pixel) in row.as_chunks::<3>().0.iter().enumerate() {
      planes[0][x] = i32::from(pixel[0]);
      planes[1][x] = i32::from(pixel[1]);
      planes[2][x] = i32::from(pixel[2]);
    }
    for (xx, out_pixel) in out_row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
      let (xmin, _) = coefficients.bounds[xx];
      let weights = coefficients.weights(xx);
      for (channel, plane) in planes.iter().enumerate() {
        let taps = &plane[xmin..xmin + weights.len()];
        let sum = taps.iter().zip(weights).fold(ROUNDING, |sum, (&value, &weight)| sum.wrapping_add(value * weight));
        out_pixel[channel] = clip8(sum);
      }
    }
  }
  output
}

fn resample_vertical(source: &[u8], width: usize, coefficients: &Coefficients) -> Vec<u8> {
  let stride = width * 3;
  let out_height = coefficients.bounds.len();
  let mut output = vec![0u8; stride * out_height];
  let mut sums = vec![0i32; stride];
  for (yy, out_row) in output.chunks_exact_mut(stride).enumerate() {
    let (ymin, _) = coefficients.bounds[yy];
    sums.fill(ROUNDING);
    for (y, &weight) in coefficients.weights(yy).iter().enumerate() {
      let row = &source[(ymin + y) * stride..(ymin + y + 1) * stride];
      for (sum, &value) in sums.iter_mut().zip(row) {
        *sum = sum.wrapping_add(i32::from(value) * weight);
      }
    }
    for (out, &sum) in out_row.iter_mut().zip(&sums) {
      *out = clip8(sum);
    }
  }
  output
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn identity_resize_returns_the_same_pixels() {
    let image = RgbImage { width: 2, height: 1, pixels: vec![1, 2, 3, 4, 5, 6] };
    assert_eq!(resize(&image, 2, 1, Filter::Bicubic).pixels, image.pixels);
  }

  #[test]
  fn cropped_resize_matches_cropping_the_whole_resize() {
    let image =
      RgbImage { width: 37, height: 23, pixels: (0..37 * 23 * 3).map(|value| (value * 7 % 251) as u8).collect() };
    let cases = [(80, 50, (10, 5, 30, 20)), (37, 50, (3, 0, 30, 50)), (80, 23, (0, 4, 80, 10)), (37, 23, (5, 6, 7, 8))];
    for (width, height, (left, top, crop_width, crop_height)) in cases {
      for filter in [Filter::Bicubic, Filter::Lanczos] {
        let whole = resize(&image, width, height, filter);
        let expected: Vec<u8> = (top..top + crop_height)
          .flat_map(|y| {
            let start = ((y * width + left) * 3) as usize;
            whole.pixels[start..start + crop_width as usize * 3].to_vec()
          })
          .collect();
        let cropped = resize_cropped(&image, width, height, (left, top, crop_width, crop_height), filter);
        assert_eq!(cropped.pixels, expected, "{width}x{height} {filter:?}");
      }
    }
  }

  #[test]
  fn constant_image_stays_constant() {
    let image = RgbImage { width: 37, height: 23, pixels: vec![200; 37 * 23 * 3] };
    for (width, height) in [(10, 7), (80, 50), (37, 5), (5, 23)] {
      let resized = resize(&image, width, height, Filter::Bicubic);
      assert!(resized.pixels.iter().all(|&value| value == 200));
      let resized = resize(&image, width, height, Filter::Lanczos);
      assert!(resized.pixels.iter().all(|&value| value == 200));
    }
  }
}
