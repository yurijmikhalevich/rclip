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

/// Like CLIP's `basic_clean`, except that of `ftfy.fix_text` only the NFC normalization is kept: its other fixes
/// (mojibake, ligatures, curly quotes, ...) only matter for unusual queries, and queries are never stored.
fn basic_clean(text: &str) -> String {
  let text: String = text.nfc().collect();
  let text = html_escape::decode_html_entities(&text);
  html_escape::decode_html_entities(&text).trim().to_string()
}

fn whitespace_clean(text: &str) -> String {
  text.split_whitespace().collect::<Vec<_>>().join(" ")
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
  fn cleans_text_like_clip() {
    assert_eq!(
      whitespace_clean(&basic_clean("  e\u{301}clair &amp;amp; chips\r\n\tfries ")),
      "\u{e9}clair & chips fries"
    );
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
