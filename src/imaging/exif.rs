//! Reads the EXIF orientation tag, used to show images upright in the interactive UI.

const ORIENTATION_TAG: u16 = 0x0112;

/// Returns the EXIF orientation (1-8) stored in a JPEG, TIFF, PNG, or WebP file, if any.
pub fn orientation(data: &[u8]) -> Option<u16> {
  let tiff = if data.starts_with(&[0xFF, 0xD8]) {
    jpeg_exif(data)?
  } else if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") {
    data
  } else if data.starts_with(b"\x89PNG\r\n\x1a\n") {
    png_exif(data)?
  } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
    webp_exif(data)?
  } else {
    return None;
  };
  tiff_orientation(tiff.strip_prefix(b"Exif\0\0").unwrap_or(tiff)).filter(|value| (1..=8).contains(value))
}

fn jpeg_exif(data: &[u8]) -> Option<&[u8]> {
  let mut offset = 2;
  while offset + 4 <= data.len() && data[offset] == 0xFF {
    let marker = data[offset + 1];
    if marker == 0xDA || marker == 0xD9 {
      return None;
    }
    let length = usize::from(u16::from_be_bytes([data[offset + 2], data[offset + 3]]));
    let segment = data.get(offset + 4..offset + 2 + length)?;
    if marker == 0xE1 && segment.starts_with(b"Exif\0\0") {
      return Some(segment);
    }
    offset += 2 + length;
  }
  None
}

fn png_exif(data: &[u8]) -> Option<&[u8]> {
  let mut offset = 8;
  while offset + 8 <= data.len() {
    let length = u32::from_be_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
    let kind = &data[offset + 4..offset + 8];
    if kind == b"eXIf" {
      return data.get(offset + 8..offset + 8 + length);
    }
    if kind == b"IDAT" {
      return None;
    }
    offset += 12 + length;
  }
  None
}

fn webp_exif(data: &[u8]) -> Option<&[u8]> {
  let mut offset = 12;
  while offset + 8 <= data.len() {
    let length = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().ok()?) as usize;
    if &data[offset..offset + 4] == b"EXIF" {
      return data.get(offset + 8..offset + 8 + length);
    }
    offset += 8 + length + (length & 1);
  }
  None
}

fn tiff_orientation(tiff: &[u8]) -> Option<u16> {
  let little_endian = match tiff.get(..2)? {
    b"II" => true,
    b"MM" => false,
    _ => return None,
  };
  let u16_at = |offset: usize| -> Option<u16> {
    let bytes: [u8; 2] = tiff.get(offset..offset + 2)?.try_into().ok()?;
    Some(if little_endian { u16::from_le_bytes(bytes) } else { u16::from_be_bytes(bytes) })
  };
  let u32_at = |offset: usize| -> Option<u32> {
    let bytes: [u8; 4] = tiff.get(offset..offset + 4)?.try_into().ok()?;
    Some(if little_endian { u32::from_le_bytes(bytes) } else { u32::from_be_bytes(bytes) })
  };
  let ifd = u32_at(4)? as usize;
  let entries = usize::from(u16_at(ifd)?);
  (0..entries).map(|index| ifd + 2 + index * 12).find(|&entry| u16_at(entry) == Some(ORIENTATION_TAG)).and_then(
    |entry| {
      // The value is a SHORT stored inline at the start of the value field.
      u16_at(entry + 8)
    },
  )
}

#[cfg(test)]
mod tests {
  use super::*;

  fn tiff(orientation: u16, little_endian: bool) -> Vec<u8> {
    let u16b = |value: u16| if little_endian { value.to_le_bytes() } else { value.to_be_bytes() };
    let u32b = |value: u32| if little_endian { value.to_le_bytes() } else { value.to_be_bytes() };
    let mut data = if little_endian { b"II".to_vec() } else { b"MM".to_vec() };
    data.extend(u16b(42));
    data.extend(u32b(8));
    data.extend(u16b(1));
    data.extend(u16b(ORIENTATION_TAG));
    data.extend(u16b(3));
    data.extend(u32b(1));
    data.extend(u16b(orientation));
    data.extend([0, 0]);
    data.extend(u32b(0));
    data
  }

  #[test]
  fn reads_tiff_orientation_in_both_byte_orders() {
    assert_eq!(orientation(&tiff(6, true)), Some(6));
    assert_eq!(orientation(&tiff(8, false)), Some(8));
    assert_eq!(orientation(&tiff(0, true)), None);
  }

  #[test]
  fn reads_jpeg_app1_exif() {
    let exif = [b"Exif\0\0".to_vec(), tiff(3, false)].concat();
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0, 4, 0, 0, 0xFF, 0xE1];
    jpeg.extend(((exif.len() + 2) as u16).to_be_bytes());
    jpeg.extend(exif);
    jpeg.extend([0xFF, 0xDA]);
    assert_eq!(orientation(&jpeg), Some(3));
  }

  #[test]
  fn reads_png_and_webp_exif() {
    let exif = tiff(5, true);
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend((exif.len() as u32).to_be_bytes());
    png.extend(b"eXIf");
    png.extend(&exif);
    png.extend([0; 4]);
    assert_eq!(orientation(&png), Some(5));

    let mut webp = b"RIFF\0\0\0\0WEBP".to_vec();
    webp.extend(b"EXIF");
    webp.extend((exif.len() as u32).to_le_bytes());
    webp.extend(&exif);
    assert_eq!(orientation(&webp), Some(5));
  }

  #[test]
  fn ignores_files_without_exif() {
    assert_eq!(orientation(b"GIF89a"), None);
    assert_eq!(orientation(&[0xFF, 0xD8, 0xFF, 0xDA]), None);
  }
}
