//! The CLIP model: text and image encoders plus query parsing.

#[cfg(target_os = "macos")]
pub mod coreml;
pub mod download;
mod onnx;
pub mod tokenizer;

use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use regex::Regex;

use crate::imaging::{self, ImageError, RgbImage, preprocess};
use download::ModelFiles;
use onnx::OnnxModel;
use tokenizer::Tokenizer;

pub const VECTOR_SIZE: usize = 512;

/// A query embedding. Like the Python version (where NumPy promotes the float32 embeddings when
/// multiplying them by float64 multipliers), queries are combined and scored in double precision.
pub type FeatureVector = Vec<f64>;

pub struct Model {
  files: ModelFiles,
  textual: OnceLock<OnnxModel>,
  visual: OnceLock<Arc<OnnxModel>>,
  tokenizer: OnceLock<Tokenizer>,
}

/// The visual encoder used for indexing: Core ML on macOS, ONNX Runtime elsewhere.
pub enum IndexEncoder {
  #[cfg(target_os = "macos")]
  CoreMl(coreml::CoreMlModel),
  Onnx(Arc<OnnxModel>),
}

impl IndexEncoder {
  /// Computes normalized embeddings for preprocessed images stored back to back in `batch`.
  pub fn encode(&self, batch: &[f32]) -> Result<Vec<f32>> {
    let count = batch.len() / preprocess::TENSOR_LEN;
    let mut features = match self {
      #[cfg(target_os = "macos")]
      IndexEncoder::CoreMl(model) => model.predict(batch)?,
      IndexEncoder::Onnx(model) => model.run(&[count, 3, 256, 256], batch.to_vec())?,
    };
    normalize_rows(&mut features);
    Ok(features)
  }
}

/// Errors in user-supplied queries; reported without a backtrace.
#[derive(Debug, thiserror::Error)]
pub enum QueryError {
  #[error("File \"{0}\" not found. Check if you have typos in the filename.")]
  NotFound(String),
  #[error("File \"{0}\" is not an image. You can only use image files or text as queries.")]
  NotAnImage(String),
  #[error("A query image is too large to process.")]
  TooLarge,
  #[error("{0}")]
  Other(#[from] anyhow::Error),
}

impl Model {
  pub fn new(files: ModelFiles) -> Self {
    Model { files, textual: OnceLock::new(), visual: OnceLock::new(), tokenizer: OnceLock::new() }
  }

  fn textual(&self) -> Result<&OnnxModel> {
    if self.textual.get().is_none() {
      let _ = self.textual.set(OnnxModel::load(&self.files.textual())?);
    }
    Ok(self.textual.get().expect("the textual model was just loaded"))
  }

  fn visual(&self) -> Result<&Arc<OnnxModel>> {
    if self.visual.get().is_none() {
      let _ = self.visual.set(Arc::new(OnnxModel::load(&self.files.visual())?));
    }
    Ok(self.visual.get().expect("the visual model was just loaded"))
  }

  fn tokenizer(&self) -> Result<&Tokenizer> {
    if self.tokenizer.get().is_none() {
      let _ = self.tokenizer.set(Tokenizer::from_file(&self.files.tokenizer_vocab())?);
    }
    Ok(self.tokenizer.get().expect("the tokenizer was just loaded"))
  }

  /// Loads the encoder used for indexing. It is owned by the caller so that its resources are
  /// released as soon as indexing finishes.
  pub fn index_encoder(&self) -> Result<IndexEncoder> {
    #[cfg(target_os = "macos")]
    if download::use_coreml_for_visual_index() {
      return Ok(IndexEncoder::CoreMl(coreml::CoreMlModel::load(&self.files.visual_coreml())?));
    }
    Ok(IndexEncoder::Onnx(Arc::clone(self.visual()?)))
  }

  pub fn text_features(&self, texts: &[&str]) -> Result<Vec<f32>> {
    let tokens = self.tokenizer()?.tokenize(texts);
    let mut features = self.textual()?.run(&[texts.len(), tokenizer::CONTEXT_LENGTH], tokens)?;
    normalize_rows(&mut features);
    Ok(features)
  }

  pub fn image_features(&self, images: &[RgbImage]) -> Result<Vec<f32>> {
    let mut batch = Vec::with_capacity(images.len() * preprocess::TENSOR_LEN);
    for image in images {
      preprocess::preprocess_into(image, &mut batch);
    }
    let mut features = self.visual()?.run(&[images.len(), 3, 256, 256], batch)?;
    normalize_rows(&mut features);
    Ok(features)
  }

  /// Combines text, local image, and image URL queries into one feature vector, weighting each
  /// by its optional `<multiplier>:` prefix.
  pub fn query_features(&self, queries: &[String]) -> Result<FeatureVector, QueryError> {
    let mut phrases = Vec::new();
    let mut files = Vec::new();
    let mut urls = Vec::new();
    for query in queries {
      let (multiplier, query) = split_multiplier(query);
      if is_http_url(query) {
        urls.push((multiplier, query));
      } else if is_file_path(query) {
        files.push((multiplier, query));
      } else {
        phrases.push((multiplier, query));
      }
    }

    let mut image_features = None;
    if !files.is_empty() || !urls.is_empty() {
      let mut images = Vec::new();
      for (_, url) in &urls {
        images.push(download_image(url)?);
      }
      for (_, path) in &files {
        images.push(read_query_image(path)?);
      }
      let multipliers: Vec<f64> = urls.iter().chain(&files).map(|(multiplier, _)| *multiplier).collect();
      image_features = Some(weighted_sum(&self.image_features(&images)?, &multipliers));
    }

    let mut text_features = None;
    if !phrases.is_empty() {
      let texts: Vec<&str> = phrases.iter().map(|(_, text)| *text).collect();
      let multipliers: Vec<f64> = phrases.iter().map(|(multiplier, _)| *multiplier).collect();
      text_features = Some(weighted_sum(&self.text_features(&texts)?, &multipliers));
    }

    Ok(match (text_features, image_features) {
      (Some(text), Some(image)) => text.iter().zip(&image).map(|(a, b)| a + b).collect(),
      (Some(features), None) | (None, Some(features)) => features,
      (None, None) => vec![0.0; VECTOR_SIZE],
    })
  }
}

fn weighted_sum(features: &[f32], multipliers: &[f64]) -> FeatureVector {
  let mut sum = vec![0.0f64; VECTOR_SIZE];
  for (row, &multiplier) in features.as_chunks::<VECTOR_SIZE>().0.iter().zip(multipliers) {
    for (total, &value) in sum.iter_mut().zip(row) {
      *total += f64::from(value) * multiplier;
    }
  }
  sum
}

fn normalize_rows(features: &mut [f32]) {
  for row in features.as_chunks_mut::<VECTOR_SIZE>().0 {
    let norm = row.iter().map(|&value| f64::from(value) * f64::from(value)).sum::<f64>().sqrt() as f32;
    for value in row {
      *value /= norm;
    }
  }
}

fn read_query_image(query: &str) -> Result<RgbImage, QueryError> {
  let path = query.strip_prefix("file://").unwrap_or(query);
  // Query images are chosen explicitly and decoded one at a time, so the indexing pixel cap does
  // not apply to them.
  imaging::read_image(path, None, Some(preprocess::IMAGE_SIZE)).map_err(|error| match error {
    ImageError::Io(error) if error.kind() == std::io::ErrorKind::NotFound => QueryError::NotFound(path.to_string()),
    ImageError::Unidentified => QueryError::NotAnImage(path.to_string()),
    ImageError::TooLarge { .. } => QueryError::TooLarge,
    error => QueryError::Other(anyhow::anyhow!("cannot read {path}: {error}")),
  })
}

const MAX_DOWNLOAD_SIZE_BYTES: u64 = 50_000_000;

// See: https://meta.wikimedia.org/wiki/User-Agent_policy
const USER_AGENT: &str = "rclip - (https://github.com/yurijmikhalevich/rclip)";

fn download_image(url: &str) -> Result<RgbImage, QueryError> {
  let agent: ureq::Agent = ureq::Agent::config_builder()
    .user_agent(USER_AGENT)
    .timeout_global(Some(std::time::Duration::from_secs(60)))
    .build()
    .into();
  // Servers that reject HEAD requests can still serve the image, so only the size is checked here.
  let head = agent
    .head(url)
    .config()
    .http_status_as_error(false)
    .build()
    .call()
    .with_context(|| format!("cannot download {url}"))?;
  if let Some(length) = head.headers().get("content-length").and_then(|value| value.to_str().ok()?.parse::<u64>().ok())
    && length > MAX_DOWNLOAD_SIZE_BYTES
  {
    return Err(QueryError::Other(anyhow::anyhow!("Avoiding download of large ({length} byte) file.")));
  }
  let mut response = agent.get(url).call().with_context(|| format!("cannot download {url}"))?;
  let content_type = response
    .headers()
    .get("content-type")
    .and_then(|value| value.to_str().ok())
    .map(|value| value.split(';').next().unwrap_or("").trim().to_ascii_lowercase())
    .unwrap_or_default();
  let data = response
    .body_mut()
    .with_config()
    .limit(MAX_DOWNLOAD_SIZE_BYTES)
    .read_to_vec()
    .with_context(|| format!("cannot download {url}"))?;
  let url_path = url.split(['?', '#']).next().unwrap_or(url);
  let mut extension = imaging::file_extension(url_path);
  if content_type == "image/heic" || content_type == "image/heif" {
    extension = "heic".into();
  }
  imaging::decode(&data, &extension, None, None).map_err(|error| match error {
    ImageError::Unidentified => QueryError::NotAnImage(url.to_string()),
    ImageError::TooLarge { .. } => QueryError::TooLarge,
    error => QueryError::Other(anyhow::anyhow!("cannot decode {url}: {error}")),
  })
}

fn multiplier_pattern() -> &'static Regex {
  static PATTERN: OnceLock<Regex> = OnceLock::new();
  PATTERN.get_or_init(|| {
    Regex::new(r"^(?P<multiplier>([0-9]+(\.[0-9]+)?|\.[0-9]+|[0-9]+\.)):(?P<query>.+)$")
      .expect("the multiplier pattern is valid")
  })
}

/// Splits an optional `<multiplier>:` prefix off a query, e.g. `"2:cat"` → `(2.0, "cat")`.
pub fn split_multiplier(query: &str) -> (f64, &str) {
  match multiplier_pattern().captures(query) {
    Some(captures) => {
      let multiplier = captures["multiplier"].parse::<f64>().expect("the pattern only matches numbers");
      (multiplier, captures.name("query").expect("the pattern has a query group").as_str())
    }
    None => (1.0, query),
  }
}

pub fn is_text_query(query: &str) -> bool {
  let (_, query) = split_multiplier(query);
  !is_http_url(query) && !is_file_path(query)
}

pub fn is_http_url(query: &str) -> bool {
  query.starts_with("https://") || query.starts_with("http://")
}

pub fn is_file_path(query: &str) -> bool {
  let bytes = query.as_bytes();
  let windows_absolute = bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'\\';
  query.starts_with('/') || query.starts_with("file://") || query.starts_with("./") || windows_absolute
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn extracts_query_multipliers() {
    assert_eq!(split_multiplier("cat"), (1.0, "cat"));
    assert_eq!(split_multiplier("2:cat"), (2.0, "cat"));
    assert_eq!(split_multiplier("0.5:./cat.jpg"), (0.5, "./cat.jpg"));
    assert_eq!(split_multiplier(".5:cat"), (0.5, "cat"));
    assert_eq!(split_multiplier("3.:cat"), (3.0, "cat"));
    assert_eq!(split_multiplier("2:"), (1.0, "2:"));
    assert_eq!(split_multiplier("a:cat"), (1.0, "a:cat"));
    assert_eq!(split_multiplier("1:2:cat"), (1.0, "2:cat"));
  }

  #[test]
  fn classifies_queries() {
    for query in ["cat", "a photo of a dog", "2:cat", "cat.jpg", "~/cat.jpg"] {
      assert!(is_text_query(query), "{query}");
    }
    for query in
      ["./cat.jpg", "/tmp/cat.jpg", "file:///tmp/cat.jpg", "C:\\cat.jpg", "https://x/y.jpg", "http://x", "2:./cat.jpg"]
    {
      assert!(!is_text_query(query), "{query}");
    }
  }

  #[test]
  fn downloads_images_from_servers_that_reject_head_requests() {
    use std::io::{BufRead, Write};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/cat.png", listener.local_addr().unwrap());
    let mut png = Vec::new();
    image::RgbImage::new(3, 2).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    std::thread::spawn(move || {
      for stream in listener.incoming() {
        let stream = stream.unwrap();
        let request: Vec<String> =
          std::io::BufReader::new(&stream).lines().map(Result::unwrap).take_while(|line| !line.is_empty()).collect();
        let response = if request[0].starts_with("HEAD ") {
          b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
        } else {
          let headers = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", png.len());
          [headers.as_bytes(), &png].concat()
        };
        (&stream).write_all(&response).unwrap();
      }
    });
    let image = download_image(&url).unwrap();
    assert_eq!((image.width, image.height), (3, 2));
  }

  #[test]
  fn weighted_sum_scales_and_adds_rows() {
    let mut features = vec![1.0; VECTOR_SIZE];
    features.extend(vec![2.0; VECTOR_SIZE]);
    let sum = weighted_sum(&features, &[0.5, -1.0]);
    assert!(sum.iter().all(|&value| value == -1.5));
  }

  #[test]
  fn normalizes_rows() {
    let mut features = vec![3.0; VECTOR_SIZE];
    normalize_rows(&mut features);
    let norm: f32 = features.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
  }
}
