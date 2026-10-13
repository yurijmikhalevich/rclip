//! JPEG 2000 decoding through OpenJPEG, which is what Pillow uses as well.

use super::{ImageError, RgbImage, check_pixel_limit};

const JP2_SIGNATURE: &[u8] = b"\x00\x00\x00\x0cjP  \r\n\x87\n";
const J2K_SIGNATURE: &[u8] = b"\xff\x4f\xff\x51";

pub fn is_jp2(data: &[u8]) -> bool {
  data.starts_with(JP2_SIGNATURE) || data.starts_with(J2K_SIGNATURE)
}

pub fn decode(data: &[u8], max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
  let (width, height) = dimensions(data).ok_or(ImageError::Unidentified)?;
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

/// Reads the image size from the JP2 `ihdr` box or the codestream `SIZ` marker.
fn dimensions(data: &[u8]) -> Option<(u32, u32)> {
  let be32 = |offset: usize| Some(u32::from_be_bytes(data.get(offset..offset + 4)?.try_into().ok()?));
  if data.starts_with(J2K_SIGNATURE) {
    // SOC, SIZ marker, Lsiz, Rsiz, then Xsiz, Ysiz, XOsiz, YOsiz.
    let (x, y, x0, y0) = (be32(8)?, be32(12)?, be32(16)?, be32(20)?);
    return Some((x.checked_sub(x0)?, y.checked_sub(y0)?));
  }
  let position = data.windows(4).position(|window| window == b"ihdr")?;
  let (height, width) = (be32(position + 4)?, be32(position + 8)?);
  Some((width, height))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn reads_codestream_dimensions() {
    let mut data = J2K_SIGNATURE.to_vec();
    data.extend_from_slice(&[0, 41, 0, 0]);
    for value in [640u32, 480, 10, 20] {
      data.extend_from_slice(&value.to_be_bytes());
    }
    assert!(is_jp2(&data));
    assert_eq!(dimensions(&data), Some((630, 460)));
  }
}
