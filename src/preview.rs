//! Inline image previews with the Kitty graphics protocol.

use std::io::Write;

use anyhow::Result;
use base64::Engine;

use crate::imaging::resample::{Filter, resize};
use crate::imaging::{self, RgbImage};

const CHUNK_SIZE: usize = 4096;

/// Prints `filepath` scaled down to at most `height` pixels, followed by a newline.
pub fn preview(output: &mut impl Write, filepath: &str, height: u32) -> Result<()> {
  // Previews are shown one at a time for images the user asked to see, so the indexing pixel cap
  // does not apply.
  let image = imaging::read_image(filepath, None, None)?;
  let image = if height < image.height {
    let width = (f64::from(height) * f64::from(image.width) / f64::from(image.height)) as u32;
    resize(&image, width.max(1), height, Filter::Lanczos)
  } else {
    image
  };
  let payload = base64::engine::general_purpose::STANDARD.encode(encode_png(&image)?);
  let tmux = std::env::var_os("TMUX").is_some() || std::env::var("TERM").is_ok_and(|term| term.starts_with("tmux"));
  write_kitty_image(output, &payload, tmux)?;
  writeln!(output)?;
  Ok(())
}

pub fn encode_png(image: &RgbImage) -> Result<Vec<u8>> {
  let mut png = Vec::new();
  image::ImageEncoder::write_image(
    image::codecs::png::PngEncoder::new(&mut png),
    &image.pixels,
    image.width,
    image.height,
    image::ExtendedColorType::Rgb8,
  )?;
  Ok(png)
}

/// Transmits and displays a base64 PNG in 4 KiB chunks, wrapped for tmux passthrough if needed.
fn write_kitty_image(output: &mut impl Write, payload: &str, tmux: bool) -> std::io::Result<()> {
  let chunks: Vec<&str> =
    payload.as_bytes().chunks(CHUNK_SIZE).map(|chunk| std::str::from_utf8(chunk).expect("base64 is ASCII")).collect();
  for (index, chunk) in chunks.iter().enumerate() {
    let more = u8::from(index + 1 < chunks.len());
    let command = if index == 0 { format!("a=T,f=100,q=2,m={more}") } else { format!("q=2,m={more}") };
    let sequence = format!("\x1b_G{command};{chunk}\x1b\\");
    if tmux {
      write!(output, "\x1bPtmux;{}\x1b\\", sequence.replace('\x1b', "\x1b\x1b"))?;
    } else {
      output.write_all(sequence.as_bytes())?;
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn chunks_payloads_like_the_python_version() {
    let payload = "A".repeat(CHUNK_SIZE + 10);
    let mut output = Vec::new();
    write_kitty_image(&mut output, &payload, false).unwrap();
    let output = String::from_utf8(output).unwrap();
    assert!(output.starts_with(&format!("\x1b_Ga=T,f=100,q=2,m=1;{}\x1b\\", "A".repeat(CHUNK_SIZE))));
    assert!(output.ends_with(&format!("\x1b_Gq=2,m=0;{}\x1b\\", "A".repeat(10))));
  }

  #[test]
  fn wraps_sequences_for_tmux() {
    let mut output = Vec::new();
    write_kitty_image(&mut output, "QQ==", true).unwrap();
    assert_eq!(String::from_utf8(output).unwrap(), "\x1bPtmux;\x1b\x1b_Ga=T,f=100,q=2,m=0;QQ==\x1b\x1b\\\x1b\\");
  }
}
