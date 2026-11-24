//! Inline image previews with the Kitty graphics protocol.

use std::io::Write;

use anyhow::Result;

use crate::imaging;
use crate::imaging::resample::{Filter, resize};

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
  let sequence = crate::kitty::transmit_png("a=T,f=100,q=2", &image.png()?, crate::kitty::in_tmux());
  writeln!(output, "{sequence}")?;
  Ok(())
}
