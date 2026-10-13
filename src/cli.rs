//! Command-line arguments, mirroring the Python version's argparse interface.

use std::ffi::OsString;

use clap::error::ErrorKind;
use clap::{ArgAction, CommandFactory, Parser};

const HINTS: &str = "\
hints:
  - relative file path should be prefixed with ./, e.g. \"./cat.jpg\", not \"cat.jpg\"
  - any query can be prefixed with a multiplier, e.g. \"2:cat\", \"0.5:./cat-sleeps-on-a-chair.jpg\"; adding a \
multiplier is especially useful when combining image and text queries because image queries are usually weighted \
more than text ones

get help:
  https://github.com/yurijmikhalevich/rclip/discussions/new/choose";

/// The image pixel cap selected with `--max-image-megapixels`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaxImagePixels {
  /// Chosen from the available memory.
  Auto,
  /// No cap.
  Disabled,
  Limit(u64),
}

#[derive(Debug, Parser)]
#[command(
  name = "rclip",
  version,
  about = "rclip is a semantic photo search tool for the command line",
  after_help = HINTS,
  disable_version_flag = true,
  max_term_width = 100
)]
pub struct Args {
  /// a text query or a path/URL to an image file; cannot be used with --interactive
  pub query: Option<String>,

  #[arg(long, short = 'v', action = ArgAction::Version, help = concat!("prints \"rclip ", env!("CARGO_PKG_VERSION"), "\""))]
  version: (),

  /// a text query or a path/URL to an image file to add to the "original" query, can be used multiple times;
  /// "+ QUERY" is a shorthand
  #[arg(long, short = 'a', value_name = "QUERY", allow_hyphen_values = true)]
  pub add: Vec<String>,

  /// a text query or a path/URL to an image file to subtract from the "original" query, can be used multiple
  /// times; "- QUERY" is a shorthand
  #[arg(long, short = 's', visible_alias = "sub", value_name = "QUERY", allow_hyphen_values = true)]
  pub subtract: Vec<String>,

  /// number of top results to display, or load per search batch with --interactive; default: 10, or 100 with
  /// --interactive; interactive maximum: 100
  #[arg(long, short = 't', value_parser = positive_int)]
  pub top: Option<usize>,

  /// opens the interactive terminal UI; enter text queries in the UI
  #[arg(long, short = 'i', conflicts_with_all = ["preview", "filepath_only"])]
  pub interactive: bool,

  /// preview results in the terminal using Kitty graphics (e.g. Kitty, Ghostty, latest iTerm2, Konsole)
  #[arg(long, short = 'p', conflicts_with = "filepath_only")]
  pub preview: bool,

  /// outputs only filepaths
  #[arg(long, short = 'f')]
  pub filepath_only: bool,

  /// forces interactive mode even if the terminal is not recognized as supporting Kitty graphics
  #[arg(long)]
  pub force_interactive: bool,

  /// forces image previews even if the terminal is not recognized as supporting Kitty graphics
  #[arg(long)]
  pub force_preview: bool,

  /// preview height in pixels; default: 400
  #[arg(long, short = 'H', value_name = "PREVIEW_HEIGHT_PX", default_value_t = 400)]
  pub preview_height: u32,

  /// allows to skip updating the index if no images were added, changed, or removed
  #[arg(long, short = 'n', visible_aliases = ["skip-index", "skip-indexing"])]
  pub no_indexing: bool,

  /// the size of the image batch used when updating the search index; larger values may improve the indexing
  /// speed a bit on some hardware but will increase RAM usage; default: 8
  #[arg(long, short = 'b', value_parser = positive_int, default_value_t = 8)]
  pub indexing_batch_size: usize,

  /// dir to exclude from search, can be used multiple times; adding this argument overrides the default of
  /// ("@eaDir", "node_modules", ".git", "System Volume Information")
  #[arg(long)]
  pub exclude_dir: Option<Vec<String>>,

  /// index and search dot-prefixed hidden files and directories (e.g. .DS_Store, ._IMG_1234.JPG,
  /// .Spotlight-V100); by default they are skipped since they are usually OS metadata rather than user files
  #[arg(long)]
  pub include_hidden: bool,

  /// enables support for RAW images (ARW, CR2, and DNG are supported)
  #[arg(long)]
  pub experimental_raw_support: bool,

  /// maximum size, in megapixels, an image may have to be indexed; larger images are skipped to avoid running out
  /// of memory on huge or maliciously crafted images; pass "none" to disable the limit; default: chosen
  /// automatically based on available memory
  #[arg(long, value_name = "MP", value_parser = max_image_megapixels, default_value = "auto")]
  pub max_image_megapixels: MaxImagePixels,
}

fn positive_int(value: &str) -> Result<usize, String> {
  match value.parse::<i64>() {
    Ok(number) if number >= 1 => Ok(number as usize),
    Ok(_) => Err("should be >0".into()),
    Err(_) => Err(format!("invalid int value: '{value}'")),
  }
}

fn max_image_megapixels(value: &str) -> Result<MaxImagePixels, String> {
  let lower = value.to_ascii_lowercase();
  if lower == "auto" {
    return Ok(MaxImagePixels::Auto);
  }
  if ["none", "off", "disable", "disabled", "0"].contains(&lower.as_str()) {
    return Ok(MaxImagePixels::Disabled);
  }
  let megapixels: f64 =
    value.trim().parse().map_err(|_| "should be a number, \"auto\", or \"none\" to disable the limit".to_string())?;
  if megapixels.is_nan() || megapixels <= 0.0 {
    return Err("should be >0, \"auto\", or \"none\" to disable the limit".into());
  }
  Ok(MaxImagePixels::Limit((megapixels * 1_000_000.0).round_ties_even() as u64))
}

/// The arguments after validation, with the derived display mode and result count.
#[derive(Debug)]
pub struct Options {
  pub args: Args,
  pub top_k: usize,
}

/// argparse accepted `+` and `-` as option names (`prefix_chars="-+"`), so `rclip cat + dog - fence`
/// works; clap cannot declare such options, so translate them before parsing.
fn translate_sign_options(args: impl IntoIterator<Item = OsString>) -> Vec<OsString> {
  let mut translated = Vec::new();
  let mut options_ended = false;
  for (index, arg) in args.into_iter().enumerate() {
    if index > 0 && !options_ended {
      if arg == "--" {
        options_ended = true;
      } else if arg == "+" {
        translated.push("--add".into());
        continue;
      } else if arg == "-" {
        translated.push("--subtract".into());
        continue;
      }
    }
    translated.push(arg);
  }
  translated
}

pub fn parse() -> Options {
  parse_from(std::env::args_os())
}

pub fn parse_from(args: impl IntoIterator<Item = OsString>) -> Options {
  let mut args = Args::parse_from(translate_sign_options(args));
  let error = |message: &str| -> ! { Args::command().error(ErrorKind::ArgumentConflict, message).exit() };

  if args.force_interactive && args.preview && !args.force_preview {
    error("--force-interactive cannot be used with --preview");
  }
  if args.force_preview && args.interactive && !args.force_interactive {
    error("--force-preview cannot be used with --interactive");
  }
  if args.force_interactive && args.force_preview {
    error("--force-interactive and --force-preview cannot be used together");
  }
  if args.force_interactive && args.filepath_only {
    error("--force-interactive cannot be used with --filepath-only");
  }
  if args.force_preview && args.filepath_only {
    error("--force-preview cannot be used with --filepath-only");
  }
  args.interactive |= args.force_interactive;
  args.preview |= args.force_preview;

  if args.query.is_none() && !args.interactive {
    error("query is required unless --interactive is used");
  }
  if args.interactive && args.query.is_some() {
    error("--interactive does not accept a query");
  }
  if args.interactive && (!args.add.is_empty() || !args.subtract.is_empty()) {
    error("--interactive does not support --add or --subtract");
  }
  if args.interactive && args.top.is_some_and(|top| top > 100) {
    error("--top cannot exceed 100 with --interactive");
  }
  let top_k = args.top.unwrap_or(if args.interactive { 100 } else { 10 });
  Options { args, top_k }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn args(line: &[&str]) -> Args {
    Args::try_parse_from(translate_sign_options(
      std::iter::once("rclip").chain(line.iter().copied()).map(OsString::from),
    ))
    .unwrap()
  }

  #[test]
  fn verifies_the_command() {
    Args::command().debug_assert();
  }

  #[test]
  fn parses_sign_options_like_argparse() {
    let parsed = args(&["kitty", "+", "roof", "-", "fence", "--add", "puppy", "-a", "x", "--sub", "y", "-s", "z"]);
    assert_eq!(parsed.query.as_deref(), Some("kitty"));
    assert_eq!(parsed.add, ["roof", "puppy", "x"]);
    assert_eq!(parsed.subtract, ["fence", "y", "z"]);
  }

  #[test]
  fn parses_defaults() {
    let parsed = args(&["cat"]);
    assert_eq!(parsed.indexing_batch_size, 8);
    assert_eq!(parsed.preview_height, 400);
    assert_eq!(parsed.max_image_megapixels, MaxImagePixels::Auto);
    assert_eq!(parsed.exclude_dir, None);
    assert!(!parsed.no_indexing);
  }

  #[test]
  fn parses_aliases() {
    let parsed = args(&["cat", "--skip-index", "-t", "5", "--exclude-dir", "a", "--exclude-dir", "b"]);
    assert!(parsed.no_indexing);
    assert_eq!(parsed.top, Some(5));
    assert_eq!(parsed.exclude_dir, Some(vec!["a".into(), "b".into()]));
    assert!(args(&["cat", "--skip-indexing"]).no_indexing);
    assert!(args(&["cat", "-n"]).no_indexing);
  }

  #[test]
  fn parses_max_image_megapixels() {
    assert_eq!(max_image_megapixels("AUTO"), Ok(MaxImagePixels::Auto));
    for disabled in ["none", "off", "disable", "disabled", "0"] {
      assert_eq!(max_image_megapixels(disabled), Ok(MaxImagePixels::Disabled));
    }
    assert_eq!(max_image_megapixels("12.5"), Ok(MaxImagePixels::Limit(12_500_000)));
    assert!(max_image_megapixels("-1").is_err());
    assert!(max_image_megapixels("lots").is_err());
  }

  #[test]
  fn rejects_non_positive_numbers() {
    assert!(positive_int("0").is_err());
    assert_eq!(positive_int("3"), Ok(3));
  }

  #[test]
  fn display_modes_are_mutually_exclusive() {
    for line in [["cat", "-i", "-p"], ["cat", "-p", "-f"], ["cat", "-i", "-f"]] {
      let full = std::iter::once("rclip").chain(line).map(OsString::from);
      assert!(Args::try_parse_from(full).is_err());
    }
  }

  #[test]
  fn derives_top_k() {
    let options = parse_from(["rclip", "cat"].map(OsString::from));
    assert_eq!(options.top_k, 10);
    let options = parse_from(["rclip", "--force-interactive"].map(OsString::from));
    assert!(options.args.interactive);
    assert_eq!(options.top_k, 100);
  }
}
