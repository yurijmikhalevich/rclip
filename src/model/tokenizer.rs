//! The CLIP byte-level BPE tokenizer.
//!
//! Ported from OpenAI CLIP (MIT, Copyright (c) 2021 OpenAI), via OpenCLIP. See
//! compliance/model_repository/OPENAI-CLIP-MIT.txt in the source repository.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

pub const CONTEXT_LENGTH: usize = 77;

pub struct Tokenizer {
  byte_encoder: [char; 256],
  encoder: HashMap<String, i64>,
  bpe_ranks: HashMap<(String, String), usize>,
  cache: Mutex<HashMap<String, Vec<i64>>>,
  pattern: Regex,
  sot_token: i64,
  eot_token: i64,
}

impl Tokenizer {
  /// Loads the merges from a gzipped `bpe_simple_vocab_16e6.txt.gz` file.
  pub fn from_file(path: &Path) -> Result<Self> {
    let file = std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut merges = String::new();
    flate2::read::GzDecoder::new(file).read_to_string(&mut merges)?;
    Ok(Self::from_merges(&merges))
  }

  pub fn from_merges(merges_text: &str) -> Self {
    let (byte_order, byte_encoder) = bytes_to_unicode();
    let merges: Vec<(String, String)> = merges_text
      .split('\n')
      .skip(1)
      .take(49152 - 256 - 2)
      .filter_map(|line| line.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())))
      .collect();
    // The base vocabulary follows the order of the byte table, printable bytes first.
    let base: Vec<char> = byte_order.iter().map(|&byte| byte_encoder[byte as usize]).collect();
    let mut vocab: Vec<String> = base.iter().map(char::to_string).collect();
    vocab.extend(base.iter().map(|c| format!("{c}</w>")));
    vocab.extend(merges.iter().map(|(a, b)| format!("{a}{b}")));
    vocab.extend(["<start_of_text>".to_string(), "<end_of_text>".to_string()]);
    let encoder: HashMap<String, i64> = vocab.into_iter().enumerate().map(|(i, token)| (token, i as i64)).collect();
    let bpe_ranks = merges.into_iter().enumerate().map(|(rank, pair)| (pair, rank)).collect();
    let pattern =
      Regex::new(r"(?i)<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")
        .expect("the token pattern is valid");
    Tokenizer {
      byte_encoder,
      sot_token: encoder["<start_of_text>"],
      eot_token: encoder["<end_of_text>"],
      encoder,
      bpe_ranks,
      cache: Mutex::new(HashMap::new()),
      pattern,
    }
  }

  fn bpe(&self, token: &str) -> Vec<i64> {
    if let Some(cached) = self.cache.lock().expect("the cache lock is never poisoned").get(token) {
      return cached.clone();
    }
    let chars: Vec<char> = token.chars().collect();
    let mut word: Vec<String> = chars.iter().map(char::to_string).collect();
    if let Some(last) = word.last_mut() {
      last.push_str("</w>");
    }
    loop {
      let best = word
        .windows(2)
        .filter_map(|pair| self.bpe_ranks.get(&(pair[0].clone(), pair[1].clone())).map(|&rank| (rank, pair)))
        .min_by_key(|(rank, _)| *rank);
      let Some((_, pair)) = best else { break };
      let (first, second) = (pair[0].clone(), pair[1].clone());
      let mut merged = Vec::with_capacity(word.len());
      let mut index = 0;
      while index < word.len() {
        if index + 1 < word.len() && word[index] == first && word[index + 1] == second {
          merged.push(format!("{first}{second}"));
          index += 2;
        } else {
          merged.push(std::mem::take(&mut word[index]));
          index += 1;
        }
      }
      word = merged;
      if word.len() == 1 {
        break;
      }
    }
    let ids: Vec<i64> = word.iter().map(|piece| self.encoder[piece]).collect();
    self.cache.lock().expect("the cache lock is never poisoned").insert(token.to_string(), ids.clone());
    ids
  }

  pub fn encode(&self, text: &str) -> Vec<i64> {
    let text = whitespace_clean(&basic_clean(text)).to_lowercase();
    let mut tokens = Vec::new();
    for found in self.pattern.find_iter(&text) {
      let token = found.as_str();
      if token == "<start_of_text>" || token == "<end_of_text>" {
        tokens.push(self.encoder[token]);
        continue;
      }
      let encoded: String = token.bytes().map(|byte| self.byte_encoder[byte as usize]).collect();
      tokens.extend(self.bpe(&encoded));
    }
    tokens
  }

  /// Tokenizes `texts` into a row-major `[texts.len(), 77]` matrix.
  pub fn tokenize(&self, texts: &[&str]) -> Vec<i64> {
    let mut result = vec![0i64; texts.len() * CONTEXT_LENGTH];
    for (row, text) in result.as_chunks_mut::<CONTEXT_LENGTH>().0.iter_mut().zip(texts) {
      let mut tokens = vec![self.sot_token];
      tokens.extend(self.encode(text));
      tokens.push(self.eot_token);
      if tokens.len() > CONTEXT_LENGTH {
        tokens.truncate(CONTEXT_LENGTH);
        tokens[CONTEXT_LENGTH - 1] = self.eot_token;
      }
      row[..tokens.len()].copy_from_slice(&tokens);
    }
    result
  }
}

/// Maps every byte to a printable character. Returns the bytes in table order (printable bytes
/// first, as in CLIP's `bytes_to_unicode`) and the mapping indexed by byte value.
fn bytes_to_unicode() -> (Vec<u8>, [char; 256]) {
  let mut printable: Vec<u32> = (u32::from(b'!')..=u32::from(b'~')).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
  let mut code_points = printable.clone();
  let mut next = 0;
  for byte in 0..256u32 {
    if !printable.contains(&byte) {
      printable.push(byte);
      code_points.push(256 + next);
      next += 1;
    }
  }
  let mut table = ['\0'; 256];
  for (&byte, code_point) in printable.iter().zip(code_points) {
    table[byte as usize] = char::from_u32(code_point).expect("code points below 512 are valid");
  }
  (printable.into_iter().map(|byte| byte as u8).collect(), table)
}

fn basic_clean(text: &str) -> String {
  let text = fix_text(text);
  let text = html_escape::decode_html_entities(&text);
  html_escape::decode_html_entities(&text).trim().to_string()
}

fn whitespace_clean(text: &str) -> String {
  text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The subset of `ftfy.fix_text` that applies to search queries: HTML entities, terminal escapes,
/// C1 controls, Latin ligatures, character width, curly quotes, line breaks, control characters,
/// and NFC normalization. Mojibake repair is not reproduced; it does not affect typed queries.
fn fix_text(text: &str) -> String {
  let mut text = text.to_string();
  if !text.contains('<') {
    text = html_escape::decode_html_entities(&text).into_owned();
  }
  let text = remove_terminal_escapes(&text);
  let mut fixed = String::with_capacity(text.len());
  let mut chars = text.chars().peekable();
  while let Some(c) = chars.next() {
    match c {
      '\r' => {
        chars.next_if_eq(&'\n');
        fixed.push('\n');
      }
      '\u{2028}' | '\u{2029}' | '\u{85}' => fixed.push('\n'),
      '\u{80}'..='\u{9f}' => fixed.push(windows_1252(c as u8)),
      '\u{2bc}' | '\u{2018}'..='\u{201b}' => fixed.push('\''),
      '\u{201c}'..='\u{201f}' => fixed.push('"'),
      '\u{3000}' => fixed.push(' '),
      '\u{ff01}'..='\u{ffef}' => fixed.extend(c.to_string().nfkc()),
      c if is_removed_control(c) => {}
      c => match ligature(c) {
        Some(replacement) => fixed.push_str(replacement),
        None => fixed.push(c),
      },
    }
  }
  fixed.nfc().collect()
}

fn remove_terminal_escapes(text: &str) -> std::borrow::Cow<'_, str> {
  static ESCAPES: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*[a-zA-Z]").expect("the escape pattern is valid"));
  ESCAPES.replace_all(text, "")
}

fn is_removed_control(c: char) -> bool {
  matches!(c,
    '\u{0}'..='\u{8}' | '\u{b}' | '\u{e}'..='\u{1f}' | '\u{7f}' | '\u{206a}'..='\u{206f}' | '\u{feff}'
    | '\u{fff9}'..='\u{fffc}' | '\u{1d173}'..='\u{1d17a}' | '\u{e0000}'..='\u{e007f}')
}

fn ligature(c: char) -> Option<&'static str> {
  Some(match c {
    'Ĳ' => "IJ",
    'ĳ' => "ij",
    'ŉ' => "ʼn",
    'Ǳ' => "DZ",
    'ǲ' => "Dz",
    'ǳ' => "dz",
    'Ǆ' => "DŽ",
    'ǅ' => "Dž",
    'ǆ' => "dž",
    'Ǉ' => "LJ",
    'ǈ' => "Lj",
    'ǉ' => "lj",
    'Ǌ' => "NJ",
    'ǋ' => "Nj",
    'ǌ' => "nj",
    'ﬀ' => "ff",
    'ﬁ' => "fi",
    'ﬂ' => "fl",
    'ﬃ' => "ffi",
    'ﬄ' => "ffl",
    'ﬅ' => "ſt",
    'ﬆ' => "st",
    _ => return None,
  })
}

/// Decodes a C1 control byte as Windows-1252, the way ftfy's `fix_c1_controls` does.
fn windows_1252(byte: u8) -> char {
  const TABLE: [char; 32] = [
    '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8d}', 'Ž', '\u{8f}', '\u{90}', '‘', '’',
    '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9d}', 'ž', 'Ÿ',
  ];
  TABLE[(byte - 0x80) as usize]
}

#[cfg(test)]
mod tests {
  use super::*;

  fn tokenizer() -> Tokenizer {
    // A tiny merge table: "c a" → "ca", "ca t</w>" → "cat</w>".
    Tokenizer::from_merges("#version\nc a\nca t</w>\n")
  }

  #[test]
  fn applies_merges_in_rank_order() {
    let tokenizer = tokenizer();
    let ids = tokenizer.encode("Cat");
    assert_eq!(ids, vec![tokenizer.encoder["cat</w>"]]);
  }

  #[test]
  fn wraps_and_pads_to_the_context_length() {
    let tokenizer = tokenizer();
    let tokens = tokenizer.tokenize(&["cat", ""]);
    assert_eq!(tokens.len(), 2 * CONTEXT_LENGTH);
    assert_eq!(tokens[..3], [tokenizer.sot_token, tokenizer.encoder["cat</w>"], tokenizer.eot_token]);
    assert!(tokens[3..CONTEXT_LENGTH].iter().all(|&token| token == 0));
    assert_eq!(tokens[CONTEXT_LENGTH..CONTEXT_LENGTH + 2], [tokenizer.sot_token, tokenizer.eot_token]);
  }

  #[test]
  fn truncates_long_texts_ending_with_the_eot_token() {
    let tokenizer = tokenizer();
    let tokens = tokenizer.tokenize(&[&"cat ".repeat(200)]);
    assert_eq!(tokens[CONTEXT_LENGTH - 1], tokenizer.eot_token);
  }

  #[test]
  fn cleans_text_like_ftfy_and_html_unescape() {
    assert_eq!(whitespace_clean(&basic_clean("  \u{201c}ﬁsh\u{201d} &amp;amp; chips\r\n")), "\"fish\" & chips");
    assert_eq!(fix_text("ＡＢＣ\u{3000}x"), "ABC x");
    assert_eq!(fix_text("e\u{301}"), "é");
    assert_eq!(fix_text("\x1b[31mred\x1b[0m"), "red");
  }

  #[test]
  fn base_vocabulary_follows_the_byte_table_order() {
    let tokenizer = tokenizer();
    assert_eq!(tokenizer.encoder["!"], 0);
    assert_eq!(tokenizer.encoder["a</w>"], 320);
  }

  #[test]
  fn byte_encoder_is_a_bijection() {
    let (order, table) = bytes_to_unicode();
    assert_eq!(order[0], b'!');
    let unique: std::collections::HashSet<char> = table.iter().copied().collect();
    assert_eq!(unique.len(), 256);
    assert_eq!(table[b'a' as usize], 'a');
    assert_eq!(table[0], 'Ā');
  }
}
