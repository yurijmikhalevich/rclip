//! Image resizing with the convolution filters Pillow uses, through `fast_image_resize`.

use fast_image_resize as fr;

use super::RgbImage;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
  /// Pillow's `BICUBIC` (Catmull-Rom), which the model's preprocessing was trained with.
  Bicubic,
  /// Pillow's `LANCZOS` (Lanczos-3).
  Lanczos,
}

/// Resizes `image` to `width`x`height`.
pub fn resize(image: &RgbImage, width: u32, height: u32, filter: Filter) -> RgbImage {
  resize_cropped(image, width, height, (0, 0, width, height), filter)
}

/// Like [`resize`] followed by cropping the result to `crop` (left, top, width, height), but computes only the
/// cropped pixels, so memory use does not grow with what is cropped away.
pub fn resize_cropped(
  image: &RgbImage,
  width: u32,
  height: u32,
  crop: (u32, u32, u32, u32),
  filter: Filter,
) -> RgbImage {
  let source = fr::images::ImageRef::new(image.width, image.height, &image.pixels, fr::PixelType::U8x3)
    .expect("the pixels fill the image");
  let mut target = fr::images::Image::new(crop.2, crop.3, fr::PixelType::U8x3);
  // The crop of the resized image, in the coordinates of the source image. Multiplying before dividing maps a crop
  // that reaches the edge exactly onto it, and the clamping keeps rounding from pushing it past the edge.
  let to_source = |value: u32, resized: u32, source: u32| f64::from(value) * f64::from(source) / f64::from(resized);
  let (left, top) = (to_source(crop.0, width, image.width), to_source(crop.1, height, image.height));
  let crop_width = (to_source(crop.0 + crop.2, width, image.width) - left).min(f64::from(image.width) - left);
  let crop_height = (to_source(crop.1 + crop.3, height, image.height) - top).min(f64::from(image.height) - top);
  let filter = match filter {
    Filter::Bicubic => fr::FilterType::CatmullRom,
    Filter::Lanczos => fr::FilterType::Lanczos3,
  };
  let options =
    fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(filter)).crop(left, top, crop_width, crop_height);
  fr::Resizer::new().resize(&source, &mut target, &options).expect("both images are RGB and the crop is inside");
  RgbImage { width: crop.2, height: crop.3, pixels: target.into_vec() }
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
        // The crop's offset changes how a few filter weights round, which moves a few samples by one.
        let off_by_more_than_one = cropped.pixels.iter().zip(&expected).any(|(a, b)| a.abs_diff(*b) > 1);
        assert!(cropped.pixels.len() == expected.len() && !off_by_more_than_one, "{width}x{height} {filter:?}");
      }
    }
  }

  #[test]
  fn resizes_between_any_sizes() {
    for (source_width, source_height) in [(1, 1), (7, 3), (37, 23), (120, 80), (171, 256)] {
      let image = RgbImage {
        width: source_width,
        height: source_height,
        pixels: vec![9; (source_width * source_height * 3) as usize],
      };
      for (width, height) in [(1, 1), (3, 7), (13, 5), (41, 29), (221, 165), (640, 480)] {
        for filter in [Filter::Bicubic, Filter::Lanczos] {
          assert_eq!(resize(&image, width, height, filter).pixels.len(), (width * height * 3) as usize);
          let crop = (width / 3, height / 3, width - width / 3, height - height / 3);
          assert_eq!(resize_cropped(&image, width, height, crop, filter).pixels.len(), (crop.2 * crop.3 * 3) as usize);
        }
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
