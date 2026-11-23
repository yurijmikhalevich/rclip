//! The Kitty graphics protocol: transmitting images, and placing them with Unicode placeholders.
//!
//! In the interactive UI, an image is transmitted once and given a virtual placement of `columns`x`rows` cells.
//! It is then displayed by drawing placeholder characters whose foreground color encodes the image id and whose
//! diacritics encode the row and column, so images move and clip with the rest of the terminal UI like ordinary
//! text.

use std::fmt::Write as _;

use base64::Engine;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

const PLACEHOLDER: char = '\u{10EEEE}';
const CHUNK_SIZE: usize = 4096;

/// The largest number of rows or columns a placement can span.
pub const MAX_CELLS: u16 = DIACRITICS.len() as u16;

/// Whether the output goes through tmux, which forwards the commands only when they are wrapped.
pub fn in_tmux() -> bool {
  std::env::var_os("TMUX").is_some_and(|tmux| !tmux.is_empty())
    || std::env::var("TERM").is_ok_and(|term| term.starts_with("tmux"))
}

/// Frames a PNG image for transmission in 4 KiB chunks, with `keys` in the first one.
pub fn transmit_png(keys: &str, png: &[u8], tmux: bool) -> String {
  let payload = base64::engine::general_purpose::STANDARD.encode(png);
  let mut sequence = String::with_capacity(payload.len() + payload.len() / CHUNK_SIZE * 32 + 64);
  let chunks: Vec<&[u8]> = payload.as_bytes().chunks(CHUNK_SIZE).collect();
  for (index, chunk) in chunks.iter().enumerate() {
    let more = u8::from(index + 1 < chunks.len());
    let chunk = std::str::from_utf8(chunk).expect("base64 is ASCII");
    // Only the first chunk carries the image's keys; the protocol allows just `m` and `q` after it.
    let keys = if index == 0 { format!("{keys},m={more}") } else { format!("q=2,m={more}") };
    push_command(&mut sequence, &format!("{keys};{chunk}"), tmux);
  }
  sequence
}

/// Transmits a PNG image and creates its virtual placement.
pub fn transmit(id: u32, png: &[u8], columns: u16, rows: u16, tmux: bool) -> Vec<u8> {
  let mut sequence = transmit_png(&format!("i={id},f=100,q=2"), png, tmux);
  push_command(&mut sequence, &format!("a=p,i={id},c={columns},r={rows},U=1,q=2"), tmux);
  sequence.into_bytes()
}

/// Frees an image and its placements in the terminal.
pub fn delete(id: u32, tmux: bool) -> Vec<u8> {
  let mut sequence = String::new();
  push_command(&mut sequence, &format!("a=d,d=I,i={id},q=2"), tmux);
  sequence.into_bytes()
}

fn push_command(sequence: &mut String, command: &str, tmux: bool) {
  if tmux {
    // tmux forwards escape sequences wrapped in a DCS passthrough with doubled escapes.
    let _ = write!(sequence, "\x1bPtmux;\x1b\x1b_G{command}\x1b\x1b\\\x1b\\");
  } else {
    let _ = write!(sequence, "\x1b_G{command}\x1b\\");
  }
}

/// Draws the placeholders of image `id` placed at `columns`x`rows` cells, starting at `origin` (which
/// may lie partly outside `clip`); only cells inside `clip` are drawn.
pub fn draw(buffer: &mut Buffer, id: u32, origin: (i32, i32), size: (u16, u16), clip: Rect) {
  let color = Color::Rgb((id >> 16) as u8, (id >> 8) as u8, id as u8);
  let (columns, rows) = (size.0.min(MAX_CELLS), size.1.min(MAX_CELLS));
  for row in 0..rows {
    let y = origin.1 + i32::from(row);
    if y < i32::from(clip.top()) || y >= i32::from(clip.bottom()) {
      continue;
    }
    for column in 0..columns {
      let x = origin.0 + i32::from(column);
      if x < i32::from(clip.left()) || x >= i32::from(clip.right()) {
        continue;
      }
      let mut symbol = String::with_capacity(12);
      symbol.push(PLACEHOLDER);
      symbol.push(DIACRITICS[usize::from(row)]);
      symbol.push(DIACRITICS[usize::from(column)]);
      if let Some(cell) = buffer.cell_mut((x as u16, y as u16)) {
        cell.set_symbol(&symbol).set_fg(color);
      }
    }
  }
}

/// Hands out image ids. Ids fit in 24 bits so that the foreground color alone identifies the image;
/// they start at a per-process offset to avoid clashing with other programs' images.
pub struct IdAllocator(u32);

impl IdAllocator {
  pub fn new() -> Self {
    IdAllocator((std::process::id().wrapping_mul(7919) % 0x7f_0000) + 1)
  }

  pub fn next(&mut self) -> u32 {
    let id = self.0;
    self.0 = if self.0 >= 0xff_ffff { 1 } else { self.0 + 1 };
    id
  }
}

/// Row and column diacritics from Kitty's `rowcolumn-diacritics.txt`.
const DIACRITICS: [char; 297] = [
  '\u{305}',
  '\u{30d}',
  '\u{30e}',
  '\u{310}',
  '\u{312}',
  '\u{33d}',
  '\u{33e}',
  '\u{33f}',
  '\u{346}',
  '\u{34a}',
  '\u{34b}',
  '\u{34c}',
  '\u{350}',
  '\u{351}',
  '\u{352}',
  '\u{357}',
  '\u{35b}',
  '\u{363}',
  '\u{364}',
  '\u{365}',
  '\u{366}',
  '\u{367}',
  '\u{368}',
  '\u{369}',
  '\u{36a}',
  '\u{36b}',
  '\u{36c}',
  '\u{36d}',
  '\u{36e}',
  '\u{36f}',
  '\u{483}',
  '\u{484}',
  '\u{485}',
  '\u{486}',
  '\u{487}',
  '\u{592}',
  '\u{593}',
  '\u{594}',
  '\u{595}',
  '\u{597}',
  '\u{598}',
  '\u{599}',
  '\u{59c}',
  '\u{59d}',
  '\u{59e}',
  '\u{59f}',
  '\u{5a0}',
  '\u{5a1}',
  '\u{5a8}',
  '\u{5a9}',
  '\u{5ab}',
  '\u{5ac}',
  '\u{5af}',
  '\u{5c4}',
  '\u{610}',
  '\u{611}',
  '\u{612}',
  '\u{613}',
  '\u{614}',
  '\u{615}',
  '\u{616}',
  '\u{617}',
  '\u{657}',
  '\u{658}',
  '\u{659}',
  '\u{65a}',
  '\u{65b}',
  '\u{65d}',
  '\u{65e}',
  '\u{6d6}',
  '\u{6d7}',
  '\u{6d8}',
  '\u{6d9}',
  '\u{6da}',
  '\u{6db}',
  '\u{6dc}',
  '\u{6df}',
  '\u{6e0}',
  '\u{6e1}',
  '\u{6e2}',
  '\u{6e4}',
  '\u{6e7}',
  '\u{6e8}',
  '\u{6eb}',
  '\u{6ec}',
  '\u{730}',
  '\u{732}',
  '\u{733}',
  '\u{735}',
  '\u{736}',
  '\u{73a}',
  '\u{73d}',
  '\u{73f}',
  '\u{740}',
  '\u{741}',
  '\u{743}',
  '\u{745}',
  '\u{747}',
  '\u{749}',
  '\u{74a}',
  '\u{7eb}',
  '\u{7ec}',
  '\u{7ed}',
  '\u{7ee}',
  '\u{7ef}',
  '\u{7f0}',
  '\u{7f1}',
  '\u{7f3}',
  '\u{816}',
  '\u{817}',
  '\u{818}',
  '\u{819}',
  '\u{81b}',
  '\u{81c}',
  '\u{81d}',
  '\u{81e}',
  '\u{81f}',
  '\u{820}',
  '\u{821}',
  '\u{822}',
  '\u{823}',
  '\u{825}',
  '\u{826}',
  '\u{827}',
  '\u{829}',
  '\u{82a}',
  '\u{82b}',
  '\u{82c}',
  '\u{82d}',
  '\u{951}',
  '\u{953}',
  '\u{954}',
  '\u{f82}',
  '\u{f83}',
  '\u{f86}',
  '\u{f87}',
  '\u{135d}',
  '\u{135e}',
  '\u{135f}',
  '\u{17dd}',
  '\u{193a}',
  '\u{1a17}',
  '\u{1a75}',
  '\u{1a76}',
  '\u{1a77}',
  '\u{1a78}',
  '\u{1a79}',
  '\u{1a7a}',
  '\u{1a7b}',
  '\u{1a7c}',
  '\u{1b6b}',
  '\u{1b6d}',
  '\u{1b6e}',
  '\u{1b6f}',
  '\u{1b70}',
  '\u{1b71}',
  '\u{1b72}',
  '\u{1b73}',
  '\u{1cd0}',
  '\u{1cd1}',
  '\u{1cd2}',
  '\u{1cda}',
  '\u{1cdb}',
  '\u{1ce0}',
  '\u{1dc0}',
  '\u{1dc1}',
  '\u{1dc3}',
  '\u{1dc4}',
  '\u{1dc5}',
  '\u{1dc6}',
  '\u{1dc7}',
  '\u{1dc8}',
  '\u{1dc9}',
  '\u{1dcb}',
  '\u{1dcc}',
  '\u{1dd1}',
  '\u{1dd2}',
  '\u{1dd3}',
  '\u{1dd4}',
  '\u{1dd5}',
  '\u{1dd6}',
  '\u{1dd7}',
  '\u{1dd8}',
  '\u{1dd9}',
  '\u{1dda}',
  '\u{1ddb}',
  '\u{1ddc}',
  '\u{1ddd}',
  '\u{1dde}',
  '\u{1ddf}',
  '\u{1de0}',
  '\u{1de1}',
  '\u{1de2}',
  '\u{1de3}',
  '\u{1de4}',
  '\u{1de5}',
  '\u{1de6}',
  '\u{1dfe}',
  '\u{20d0}',
  '\u{20d1}',
  '\u{20d4}',
  '\u{20d5}',
  '\u{20d6}',
  '\u{20d7}',
  '\u{20db}',
  '\u{20dc}',
  '\u{20e1}',
  '\u{20e7}',
  '\u{20e9}',
  '\u{20f0}',
  '\u{2cef}',
  '\u{2cf0}',
  '\u{2cf1}',
  '\u{2de0}',
  '\u{2de1}',
  '\u{2de2}',
  '\u{2de3}',
  '\u{2de4}',
  '\u{2de5}',
  '\u{2de6}',
  '\u{2de7}',
  '\u{2de8}',
  '\u{2de9}',
  '\u{2dea}',
  '\u{2deb}',
  '\u{2dec}',
  '\u{2ded}',
  '\u{2dee}',
  '\u{2def}',
  '\u{2df0}',
  '\u{2df1}',
  '\u{2df2}',
  '\u{2df3}',
  '\u{2df4}',
  '\u{2df5}',
  '\u{2df6}',
  '\u{2df7}',
  '\u{2df8}',
  '\u{2df9}',
  '\u{2dfa}',
  '\u{2dfb}',
  '\u{2dfc}',
  '\u{2dfd}',
  '\u{2dfe}',
  '\u{2dff}',
  '\u{a66f}',
  '\u{a67c}',
  '\u{a67d}',
  '\u{a6f0}',
  '\u{a6f1}',
  '\u{a8e0}',
  '\u{a8e1}',
  '\u{a8e2}',
  '\u{a8e3}',
  '\u{a8e4}',
  '\u{a8e5}',
  '\u{a8e6}',
  '\u{a8e7}',
  '\u{a8e8}',
  '\u{a8e9}',
  '\u{a8ea}',
  '\u{a8eb}',
  '\u{a8ec}',
  '\u{a8ed}',
  '\u{a8ee}',
  '\u{a8ef}',
  '\u{a8f0}',
  '\u{a8f1}',
  '\u{aab0}',
  '\u{aab2}',
  '\u{aab3}',
  '\u{aab7}',
  '\u{aab8}',
  '\u{aabe}',
  '\u{aabf}',
  '\u{aac1}',
  '\u{fe20}',
  '\u{fe21}',
  '\u{fe22}',
  '\u{fe23}',
  '\u{fe24}',
  '\u{fe25}',
  '\u{fe26}',
  '\u{10a0f}',
  '\u{10a38}',
  '\u{1d185}',
  '\u{1d186}',
  '\u{1d187}',
  '\u{1d188}',
  '\u{1d189}',
  '\u{1d1aa}',
  '\u{1d1ab}',
  '\u{1d1ac}',
  '\u{1d1ad}',
  '\u{1d242}',
  '\u{1d243}',
  '\u{1d244}',
];

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn frames_transmissions_in_chunks_followed_by_a_placement() {
    let png = vec![0u8; 4000];
    let sequence = String::from_utf8(transmit(7, &png, 3, 2, false)).unwrap();
    assert!(sequence.starts_with("\x1b_Gi=7,f=100,q=2,m=1;"));
    assert!(sequence.contains("\x1b\\\x1b_Gq=2,m=0;"));
    assert!(sequence.ends_with("\x1b_Ga=p,i=7,c=3,r=2,U=1,q=2\x1b\\"));
  }

  #[test]
  fn frames_previews_like_the_python_version() {
    let sequence = transmit_png("a=T,f=100,q=2", &[0; 3073], false);
    let chunk = "A".repeat(CHUNK_SIZE);
    assert!(sequence.starts_with(&format!("\x1b_Ga=T,f=100,q=2,m=1;{chunk}\x1b\\\x1b_Gq=2,m=0;")));
    assert_eq!(transmit_png("a=T,f=100,q=2", b"A", true), "\x1bPtmux;\x1b\x1b_Ga=T,f=100,q=2,m=0;QQ==\x1b\x1b\\\x1b\\");
  }

  #[test]
  fn wraps_commands_for_tmux() {
    let sequence = String::from_utf8(delete(9, true)).unwrap();
    assert_eq!(sequence, "\x1bPtmux;\x1b\x1b_Ga=d,d=I,i=9,q=2\x1b\x1b\\\x1b\\");
  }

  #[test]
  fn draws_clipped_placeholders_with_the_id_color() {
    let mut buffer = Buffer::empty(Rect::new(0, 0, 4, 3));
    draw(&mut buffer, 0x01_02_03, (-1, 1), (3, 3), Rect::new(0, 0, 4, 2));
    let cell = &buffer[(0, 1)];
    assert_eq!(cell.symbol(), "\u{10EEEE}\u{305}\u{30d}");
    assert_eq!(cell.fg, Color::Rgb(1, 2, 3));
    assert_eq!(buffer[(0, 0)].symbol(), " ");
    assert_eq!(buffer[(2, 1)].symbol(), " ");
  }

  #[test]
  fn ids_stay_within_24_bits() {
    let mut ids = IdAllocator(0xff_ffff);
    assert_eq!(ids.next(), 0xff_ffff);
    assert_eq!(ids.next(), 1);
  }
}
