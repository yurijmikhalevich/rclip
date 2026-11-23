//! JPEG 2000 decoding through OpenJPEG, which is what Pillow uses as well.

use super::{ImageError, RgbImage, check_pixel_limit};

const JP2_SIGNATURE: &[u8] = b"\x00\x00\x00\x0cjP  \r\n\x87\n";
const J2K_SIGNATURE: &[u8] = b"\xff\x4f\xff\x51";

pub fn is_jp2(data: &[u8]) -> bool {
  data.starts_with(JP2_SIGNATURE) || data.starts_with(J2K_SIGNATURE)
}

pub fn decode(data: &[u8], max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
  let (width, height) = codestream(data).and_then(dimensions).ok_or(ImageError::Unidentified)?;
  check_pixel_limit(width, height, max_pixels)?;
  let image = jpeg2k::Image::from_bytes(data).map_err(ImageError::decode)?;
  let components = image.components();
  let first = components.first().ok_or(ImageError::Unidentified)?;
  let (width, height) = (first.width(), first.height());
  let length = width as usize * height as usize;
  if components.iter().any(|component| component.data().len() < length) {
    return Err(ImageError::decode("JPEG 2000 components with different sizes are not supported"));
  }
  // Pillow's j2k unpackers shift every sample to 8 bits after re-centering signed components.
  let to_u8 = |component: &jpeg2k::ImageComponent| -> Vec<u8> {
    let precision = component.precision() as i32;
    let offset = if component.is_signed() { 1 << (precision - 1) } else { 0 };
    let shift = 8 - precision;
    component.data()[..length]
      .iter()
      .map(|&value| {
        let value = (value + offset) as u32;
        (if shift < 0 { value >> -shift } else { value << shift }) as u8
      })
      .collect()
  };
  match components {
    [luma] | [luma, _] => Ok(RgbImage::from_luma(width, height, &to_u8(luma))),
    [r, g, b, ..] => {
      let (r, g, b) = (to_u8(r), to_u8(g), to_u8(b));
      let pixels = r.iter().zip(&g).zip(&b).flat_map(|((&r, &g), &b)| [r, g, b]).collect();
      Ok(RgbImage { width, height, pixels })
    }
    [] => Err(ImageError::Unidentified),
  }
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
}
