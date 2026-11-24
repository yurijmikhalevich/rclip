//! JPEG 2000 decoding through OpenJPEG, which is what Pillow uses as well.

use super::{ImageError, RgbImage, check_pixel_limit};

const JP2_SIGNATURE: &[u8] = b"\x00\x00\x00\x0cjP  \r\n\x87\n";
const J2K_SIGNATURE: &[u8] = b"\xff\x4f\xff\x51";

pub fn is_jp2(data: &[u8]) -> bool {
  data.starts_with(JP2_SIGNATURE) || data.starts_with(J2K_SIGNATURE)
}

pub fn decode(data: &[u8], max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
  let codestream = codestream(data).ok_or(ImageError::Unidentified)?;
  let (width, height) = dimensions(codestream).ok_or(ImageError::Unidentified)?;
  check_pixel_limit(width, height, max_pixels)?;
  let image = jpeg2k::Image::from_bytes(data).map_err(ImageError::decode)?;
  let components = image.components();
  let plane = |index: usize| {
    subsampling(codestream, index)
      .and_then(|factors| plane(&components[index], factors, width, height))
      .ok_or_else(|| ImageError::decode("a JPEG 2000 component does not cover the image"))
  };
  match components.len() {
    0 => Err(ImageError::Unidentified),
    1 | 2 => Ok(RgbImage::from_luma(width, height, &plane(0)?)),
    _ => {
      // Like Pillow, treat components as YCbCr when the color space says so, or when it is
      // unspecified and the first subsampled component is a chroma one.
      let first_subsampled = (0..components.len()).find(|&index| subsampling(codestream, index) != Some((1, 1)));
      let ycc = match image.color_space() {
        jpeg2k::ColorSpace::SYCC => true,
        jpeg2k::ColorSpace::Unknown | jpeg2k::ColorSpace::Unspecified => matches!(first_subsampled, Some(1 | 2)),
        _ => false,
      };
      let (first, second, third) = (plane(0)?, plane(1)?, plane(2)?);
      let pixels = first
        .iter()
        .zip(&second)
        .zip(&third)
        .flat_map(
          |((&first, &second), &third)| if ycc { ycc_to_rgb(first, second, third) } else { [first, second, third] },
        )
        .collect();
      Ok(RgbImage { width, height, pixels })
    }
  }
}

/// The horizontal and vertical subsampling factors of component `index`, from the `SIZ` marker.
fn subsampling(codestream: &[u8], index: usize) -> Option<(u32, u32)> {
  // The component fields (Ssiz, XRsiz, YRsiz) follow the tile sizes and offsets and Csiz.
  let fields = codestream.get(42 + 3 * index..45 + 3 * index)?;
  Some((u32::from(fields[1]), u32::from(fields[2]))).filter(|&(dx, dy)| dx > 0 && dy > 0)
}

/// The component's samples at the full image size, as 8-bit values. Like Pillow, subsampled
/// components are upsampled by repeating each sample.
fn plane(component: &jpeg2k::ImageComponent, (dx, dy): (u32, u32), width: u32, height: u32) -> Option<Vec<u8>> {
  // Pillow's j2k unpackers shift every sample to 8 bits after re-centering signed components.
  let precision = component.precision() as i32;
  let offset = if component.is_signed() { 1 << (precision - 1) } else { 0 };
  let shift = 8 - precision;
  let to_u8 = |value: i32| {
    let value = (value + offset) as u32;
    (if shift < 0 { value >> -shift } else { value << shift }) as u8
  };
  let (samples, stride) = (component.data(), component.width() as usize);
  (0..height)
    .flat_map(|y| {
      (0..width).map(move |x| samples.get((y / dy) as usize * stride + (x / dx) as usize).copied().map(to_u8))
    })
    .collect()
}

/// Converts YCbCr to RGB with the fixed-point JPEG/JFIF tables of Pillow's `ConvertYCbCr.c`.
fn ycc_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
  // Pillow's table entries are the coefficient times 64, truncated toward zero after adding 0.5.
  let table = |coefficient: f64, value: u8| (coefficient * 64.0 * (f64::from(value) - 128.0) + 0.5) as i32;
  let y = i32::from(y);
  [y + (table(1.402, cr) >> 6), y + ((table(-0.34414, cb) + table(-0.71414, cr)) >> 6), y + (table(1.772, cb) >> 6)]
    .map(|value| value.clamp(0, 255) as u8)
}

fn be32(data: &[u8], offset: usize) -> Option<u32> {
  Some(u32::from_be_bytes(data.get(offset..offset + 4)?.try_into().ok()?))
}

/// The codestream that OpenJPEG decodes: the data itself, or the contents of the first `jp2c` box
/// of a JP2 file.
fn codestream(mut data: &[u8]) -> Option<&[u8]> {
  if data.starts_with(J2K_SIGNATURE) {
    return Some(data);
  }
  loop {
    let (header, length) = match be32(data, 0)? {
      0 => (8, data.len()),
      1 => (16, usize::try_from(u64::from_be_bytes(data.get(8..16)?.try_into().ok()?)).ok()?),
      length => (8, usize::try_from(length).ok()?),
    };
    if length < header {
      return None;
    }
    if data.get(4..8)? == b"jp2c" {
      return data.get(header..);
    }
    data = data.get(length..)?;
  }
}

/// Reads the image size from the codestream's `SIZ` marker, which is the size OpenJPEG decodes.
fn dimensions(codestream: &[u8]) -> Option<(u32, u32)> {
  if !codestream.starts_with(J2K_SIGNATURE) {
    return None;
  }
  // SOC, SIZ marker, Lsiz, Rsiz, then Xsiz, Ysiz, XOsiz, YOsiz.
  let (x, y, x0, y0) = (be32(codestream, 8)?, be32(codestream, 12)?, be32(codestream, 16)?, be32(codestream, 20)?);
  Some((x.checked_sub(x0)?, y.checked_sub(y0)?))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The start of a codestream: SOC, SIZ, Lsiz, Rsiz, Xsiz, Ysiz, XOsiz, and YOsiz.
  fn codestream_header(x: u32, y: u32, x0: u32, y0: u32) -> Vec<u8> {
    let mut data = J2K_SIGNATURE.to_vec();
    data.extend_from_slice(&[0, 41, 0, 0]);
    for value in [x, y, x0, y0] {
      data.extend_from_slice(&value.to_be_bytes());
    }
    data
  }

  #[test]
  fn reads_codestream_dimensions() {
    let data = codestream_header(640, 480, 10, 20);
    assert!(is_jp2(&data));
    assert_eq!(codestream(&data).and_then(dimensions), Some((630, 460)));
  }

  #[test]
  fn reads_jp2_dimensions_from_the_codestream_that_is_decoded() {
    let jp2_box = |kind: &[u8], contents: &[u8]| [&(8 + contents.len() as u32).to_be_bytes(), kind, contents].concat();
    let mut data = JP2_SIGNATURE.to_vec();
    // A box that happens to contain "ihdr" followed by a 1x1 size, before the real header.
    data.extend(jp2_box(b"xml ", b"ihdr\0\0\0\x01\0\0\0\x01"));
    data.extend(jp2_box(b"jp2h", &jp2_box(b"ihdr", b"\0\0\0\x04\0\0\0\x06\0\x03\x07\x07\0\0")));
    data.extend(jp2_box(b"jp2c", &codestream_header(6, 4, 0, 0)));
    assert!(is_jp2(&data));
    assert_eq!(codestream(&data).and_then(dimensions), Some((6, 4)));
  }

  #[test]
  fn converts_subsampled_ycbcr_to_rgb_like_pillow() {
    // Pillow 13's pixels for a 6x4 image with 4:2:0 chroma, stored as a JP2 file with the sYCC
    // color space and as a bare codestream, whose color space Pillow infers from the subsampling.
    let expected = [
      255, 60, 0, 244, 5, 0, 189, 189, 189, 242, 242, 242, 0, 56, 160, 0, 29, 133, 255, 179, 84, 255, 71, 0, 119, 119,
      119, 98, 98, 98, 144, 255, 255, 147, 255, 255, 255, 178, 135, 149, 52, 9, 0, 163, 255, 0, 122, 255, 199, 239, 0,
      7, 47, 0, 104, 7, 0, 153, 56, 13, 0, 66, 237, 0, 199, 255, 15, 55, 0, 137, 177, 0,
    ];
    for data in [include_bytes!("testdata/ycc420.jp2").as_slice(), include_bytes!("testdata/ycc420.j2k")] {
      assert_eq!(decode(data, None).unwrap(), RgbImage { width: 6, height: 4, pixels: expected.to_vec() });
    }
  }
}
