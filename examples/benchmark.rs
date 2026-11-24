//! Accuracy benchmarks for the model and the search pipeline.
//!
//! Usage: `cargo run --release --example benchmark -- <cifar100|imagenet|similarity-search>...`
//!
//! `benchmarks/README.md` explains how to get the datasets and lists the expected results.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rclip::db::{Db, VectorCacheReset};
use rclip::imaging::{self, RgbImage};
use rclip::index::{self, IndexOptions};
use rclip::model::download::ModelFiles;
use rclip::model::{Model, VECTOR_SIZE};
use rclip::search::{Query, Search};
use rclip::walk::PathFilter;

fn main() -> Result<()> {
  let datasets =
    std::env::var_os("BENCHMARK_DATASET_DIR").map_or_else(|| PathBuf::from("benchmarks/datasets"), PathBuf::from);
  let batch_size = match std::env::var("BENCHMARK_BATCH_SIZE") {
    Ok(value) => value.parse().context("BENCHMARK_BATCH_SIZE must be a number")?,
    Err(_) => 256,
  };
  let files = ModelFiles::new(&rclip::paths::datadir()?);
  files.ensure_downloaded()?;
  let model = Model::new(files);
  let benchmarks: Vec<String> = std::env::args().skip(1).collect();
  if benchmarks.is_empty() {
    bail!("usage: benchmark <cifar100|imagenet|similarity-search>...");
  }
  // indexing ImageNet takes hours, so the benchmarks that need it share one index
  let mut imagenet_index = None;
  for benchmark in &benchmarks {
    match benchmark.as_str() {
      "cifar100" => cifar100(&model, &datasets, batch_size)?,
      "imagenet" | "similarity-search" => {
        if imagenet_index.is_none() {
          imagenet_index = Some(ImageNet::index(&model, &datasets, batch_size)?);
        }
        let index = imagenet_index.as_ref().expect("the images were just indexed");
        if benchmark == "imagenet" { imagenet(&model, index, batch_size)? } else { similarity_search(&model, index)? }
      }
      _ => bail!("unknown benchmark: {benchmark}"),
    }
  }
  Ok(())
}

/// Zero-shot classification of the CIFAR-100 test images by their class names.
fn cifar100(model: &Model, datasets: &Path, batch_size: usize) -> Result<()> {
  // a record is a coarse label, a fine label, and the red, green, and blue planes of a 32x32 image
  const PLANE: usize = 32 * 32;
  const RECORD: usize = 2 + 3 * PLANE;
  let dataset = datasets.join("cifar-100-binary");
  let names = std::fs::read_to_string(dataset.join("fine_label_names.txt")).context("cannot read the class names")?;
  let classes = text_features(model, &names.lines().collect::<Vec<_>>(), batch_size)?;
  let records = std::fs::read(dataset.join("test.bin")).context("cannot read the test images")?;
  let mut accuracy = Accuracy::default();
  for batch in records.chunks(RECORD * batch_size) {
    let images: Vec<RgbImage> = batch
      .as_chunks::<RECORD>()
      .0
      .iter()
      .map(|record| {
        let planes = &record[2..];
        let pixels = (0..PLANE).flat_map(|pixel| [planes[pixel], planes[PLANE + pixel], planes[2 * PLANE + pixel]]);
        RgbImage { width: 32, height: 32, pixels: pixels.collect() }
      })
      .collect();
    let features = model.image_features(&images)?;
    for (record, image) in batch.as_chunks::<RECORD>().0.iter().zip(features.as_chunks::<VECTOR_SIZE>().0) {
      accuracy.add(&best_classes(image, &classes), usize::from(record[1]));
    }
  }
  accuracy.print();
  Ok(())
}

/// Zero-shot classification of the indexed ImageNet-1k training images by their class descriptions.
fn imagenet(model: &Model, imagenet: &ImageNet, batch_size: usize) -> Result<()> {
  let classes = &imagenet.classes;
  let descriptions: Vec<String> = classes.iter().map(|(_, description)| format!("photo of {description}")).collect();
  let features = text_features(model, &descriptions.iter().map(String::as_str).collect::<Vec<_>>(), batch_size)?;
  let accuracies = imagenet.db.scan_vectors(&imagenet.images, Accuracy::default, |accuracy, filepath, vector| {
    let image: Vec<f32> = vector.as_chunks::<4>().0.iter().map(|bytes| f32::from_le_bytes(*bytes)).collect();
    let target = classes.iter().position(|(id, _)| id == image_class(filepath));
    accuracy.add(&best_classes(&image, &features), target.context("the image belongs to an unknown class")?);
    Ok(())
  })?;
  let mut total = Accuracy::default();
  for accuracy in accuracies {
    total.processed += accuracy.processed;
    total.top1 += accuracy.top1;
    total.top5 += accuracy.top5;
  }
  total.print();
  Ok(())
}

/// Searches by ten images of every ImageNet-1k class and measures how many of the 100 best matches
/// belong to the class of the query image.
fn similarity_search(model: &Model, imagenet: &ImageNet) -> Result<()> {
  const QUERIES_PER_CLASS: usize = 10;
  let filter = PathFilter::new(None, false, false);
  let search = Search { db: &imagenet.db, model, filter: &filter, directory: &imagenet.images };
  // sorted, so every run queries the same images
  let mut filepaths = imagenet.db.filepaths(&imagenet.images)?;
  filepaths.sort();
  let (mut queries, mut accuracy) = (0, 0.0);
  for (class, _) in &imagenet.classes {
    for filepath in filepaths.iter().filter(|filepath| image_class(filepath) == class).take(QUERIES_PER_CLASS) {
      let results = search.search(&Query { query: filepath, add: &[], subtract: &[] }, 100, None)?;
      let matches = results.iter().filter(|result| image_class(&result.filepath) == class).count();
      accuracy += matches as f64 / results.len() as f64;
      queries += 1;
    }
  }
  println!("Queries: {queries}");
  println!("Accuracy: {}", accuracy / f64::from(queries));
  Ok(())
}

struct ImageNet {
  /// Holds the index; deleted when the benchmark ends.
  _datadir: tempfile::TempDir,
  db: Db,
  images: String,
  /// The class IDs with their descriptions.
  classes: Vec<(String, String)>,
}

impl ImageNet {
  /// Indexes the training images into a temporary index.
  fn index(model: &Model, datasets: &Path, batch_size: usize) -> Result<Self> {
    let dataset = std::path::absolute(datasets.join("imagenet_1k"))?;
    let source = std::fs::read_to_string(dataset.join("classes.py")).context("cannot read the classes")?;
    let entry = regex::Regex::new(r#"(?m)^\s*"(n\d+)": "(.*)",?$"#).expect("the pattern is valid");
    let classes: Vec<(String, String)> =
      entry.captures_iter(&source).map(|class| (class[1].to_string(), class[2].to_string())).collect();
    let images = dataset.join("data").into_os_string().into_string().ok().context("the dataset path is not UTF-8")?;
    let datadir = tempfile::tempdir()?;
    let db = Db::open(&datadir.path().join("db.sqlite3"), VectorCacheReset::Allow)?;
    let workers = index::loading_workers();
    let options = IndexOptions {
      batch_size,
      raw_support: false,
      max_pixels: Some(imaging::default_max_image_pixels(workers)),
      workers,
    };
    let encoder = || -> Result<index::Encoder> {
      let encoder = model.index_encoder()?;
      Ok(Box::new(move |batch| encoder.encode(batch)))
    };
    index::ensure_index(&db, encoder, &PathFilter::new(None, false, false), &images, &options)?;
    Ok(ImageNet { _datadir: datadir, db, images, classes })
  }
}

/// ImageNet files are named `<class ID>_<image ID>.JPEG`.
fn image_class(filepath: &str) -> &str {
  let name = filepath.rsplit(std::path::MAIN_SEPARATOR).next().unwrap_or(filepath);
  name.split('_').next().unwrap_or(name)
}

fn text_features(model: &Model, texts: &[&str], batch_size: usize) -> Result<Vec<f32>> {
  let mut features = Vec::with_capacity(texts.len() * VECTOR_SIZE);
  for batch in texts.chunks(batch_size) {
    features.extend(model.text_features(batch)?);
  }
  Ok(features)
}

/// The indexes of the five classes most similar to `image`, best first.
fn best_classes(image: &[f32], classes: &[f32]) -> Vec<usize> {
  let mut scores: Vec<(usize, f32)> = classes
    .as_chunks::<VECTOR_SIZE>()
    .0
    .iter()
    .map(|class| class.iter().zip(image).map(|(class, image)| class * image).sum())
    .enumerate()
    .collect();
  scores.sort_by(|a, b| b.1.total_cmp(&a.1));
  scores.iter().take(5).map(|(class, _)| *class).collect()
}

#[derive(Default)]
struct Accuracy {
  processed: u64,
  top1: u64,
  top5: u64,
}

impl Accuracy {
  fn add(&mut self, best_classes: &[usize], target: usize) {
    self.processed += 1;
    self.top1 += u64::from(best_classes.first() == Some(&target));
    self.top5 += u64::from(best_classes.contains(&target));
  }

  fn print(&self) {
    println!("Processed: {}", self.processed);
    println!("Top-1 accuracy: {}", self.top1 as f64 / self.processed as f64);
    println!("Top-5 accuracy: {}", self.top5 as f64 / self.processed as f64);
  }
}
