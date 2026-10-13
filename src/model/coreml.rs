//! The visual encoder compiled for Core ML, used for indexing on macOS where it runs on the GPU and
//! Neural Engine.

use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use anyhow::{Context, Result, anyhow, bail};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
  MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel, MLModelConfiguration,
  MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

use super::VECTOR_SIZE;
use crate::imaging::preprocess::TENSOR_LEN;

/// The Core ML model takes fixed-size batches.
pub const BATCH_SIZE: usize = 8;

pub struct CoreMlModel {
  model: Retained<MLModel>,
}

// SAFETY: MLModel predictions are thread-safe, and the model is never mutated after loading.
unsafe impl Send for CoreMlModel {}
// SAFETY: see above.
unsafe impl Sync for CoreMlModel {}

impl CoreMlModel {
  /// Loads the compiled model next to `package`, compiling the package first if needed. A compiled
  /// model that fails to load (for example, one compiled by an older macOS) is recompiled once.
  pub fn load(package: &Path) -> Result<Self> {
    let compiled = ensure_compiled(package)?;
    match Self::load_compiled(&compiled) {
      Ok(model) => Ok(model),
      Err(_) => {
        std::fs::remove_dir_all(&compiled).ok();
        Self::load_compiled(&ensure_compiled(package)?)
      }
    }
  }

  fn load_compiled(path: &Path) -> Result<Self> {
    // SAFETY: plain Core ML calls with valid arguments.
    unsafe {
      let configuration = MLModelConfiguration::new();
      configuration.setComputeUnits(MLComputeUnits::All);
      let model = MLModel::modelWithContentsOfURL_configuration_error(&file_url(path), &configuration)
        .map_err(|error| ns_error("cannot load the Core ML model", &error))?;
      Ok(CoreMlModel { model })
    }
  }

  /// Computes embeddings for `count` preprocessed images stored back to back in `batch`.
  pub fn predict(&self, batch: &[f32]) -> Result<Vec<f32>> {
    let count = batch.len() / TENSOR_LEN;
    let mut output = Vec::with_capacity(count * VECTOR_SIZE);
    let mut padded = Vec::new();
    for chunk in batch.chunks(BATCH_SIZE * TENSOR_LEN) {
      let actual = chunk.len() / TENSOR_LEN;
      let input = if actual == BATCH_SIZE {
        chunk
      } else {
        // Pad partial batches by repeating the last image, then drop the padding's outputs.
        padded.clear();
        padded.extend_from_slice(chunk);
        let last = &chunk[(actual - 1) * TENSOR_LEN..];
        for _ in actual..BATCH_SIZE {
          padded.extend_from_slice(last);
        }
        &padded
      };
      let features = self.predict_batch(input)?;
      output.extend_from_slice(&features[..actual * VECTOR_SIZE]);
    }
    Ok(output)
  }

  fn predict_batch(&self, input: &[f32]) -> Result<Vec<f32>> {
    debug_assert_eq!(input.len(), BATCH_SIZE * TENSOR_LEN);
    // SAFETY: the multi-array borrows `input`, which outlives the synchronous prediction and the
    // array itself; Core ML only reads from it.
    unsafe {
      let shape = numbers(&[BATCH_SIZE, 3, 256, 256]);
      let strides = numbers(&[TENSOR_LEN, 256 * 256, 256, 1]);
      let array = MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
        MLMultiArray::alloc(),
        NonNull::new(input.as_ptr().cast_mut().cast()).expect("slices are never null"),
        &shape,
        MLMultiArrayDataType::Float32,
        &strides,
        None,
      )
      .map_err(|error| ns_error("cannot create the Core ML input", &error))?;
      let value = MLFeatureValue::featureValueWithMultiArray(&array);
      let key = NSString::from_str("input");
      let dictionary = NSDictionary::<NSString, AnyObject>::from_retained_objects(
        &[&*key],
        &[Retained::into_super(Retained::into_super(value))],
      );
      let provider =
        MLDictionaryFeatureProvider::initWithDictionary_error(MLDictionaryFeatureProvider::alloc(), &dictionary)
          .map_err(|error| ns_error("cannot create the Core ML input", &error))?;
      let result = self
        .model
        .predictionFromFeatures_error(ProtocolObject::<dyn MLFeatureProvider>::from_ref(&*provider))
        .map_err(|error| ns_error("Core ML prediction failed", &error))?;
      let output = result
        .featureValueForName(&NSString::from_str("output"))
        .and_then(|value| value.multiArrayValue())
        .ok_or_else(|| anyhow!("the Core ML model has no \"output\" multi-array"))?;
      read_matrix(&output)
    }
  }
}

/// Copies a `[rows, 512]` multi-array of float32 or float16 values into a contiguous vector.
unsafe fn read_matrix(array: &MLMultiArray) -> Result<Vec<f32>> {
  // SAFETY: the caller passes a valid multi-array; its data pointer stays valid while it lives.
  unsafe {
    let shape: Vec<usize> = array.shape().iter().map(|n| n.as_usize()).collect();
    let strides: Vec<usize> = array.strides().iter().map(|n| n.as_usize()).collect();
    let [rows, columns] = shape[..] else { bail!("unexpected Core ML output shape {shape:?}") };
    if columns != VECTOR_SIZE {
      bail!("unexpected Core ML output shape {shape:?}");
    }
    #[allow(deprecated)]
    let data = array.dataPointer();
    let mut output = Vec::with_capacity(rows * columns);
    let data_type = array.dataType();
    for row in 0..rows {
      for column in 0..columns {
        let offset = row * strides[0] + column * strides[1];
        output.push(if data_type == MLMultiArrayDataType::Float32 {
          *data.cast::<f32>().as_ptr().add(offset)
        } else if data_type == MLMultiArrayDataType::Float16 {
          f16_to_f32(*data.cast::<u16>().as_ptr().add(offset))
        } else {
          bail!("unexpected Core ML output data type");
        });
      }
    }
    Ok(output)
  }
}

fn f16_to_f32(bits: u16) -> f32 {
  let sign = u32::from(bits >> 15) << 31;
  let exponent = u32::from((bits >> 10) & 0x1f);
  let mantissa = u32::from(bits & 0x3ff);
  let value = match exponent {
    0 => {
      let magnitude = mantissa as f32 * 2f32.powi(-24);
      return if sign == 0 { magnitude } else { -magnitude };
    }
    0x1f => sign | 0x7f80_0000 | (mantissa << 13),
    _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
  };
  f32::from_bits(value)
}

fn numbers(values: &[usize]) -> Retained<NSArray<NSNumber>> {
  let numbers: Vec<Retained<NSNumber>> = values.iter().map(|&value| NSNumber::new_usize(value)).collect();
  NSArray::from_retained_slice(&numbers)
}

fn file_url(path: &Path) -> Retained<NSURL> {
  NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))
}

fn ns_error(context: &str, error: &NSError) -> anyhow::Error {
  anyhow!("{context}: {}", error.localizedDescription())
}

/// The compiled model sits next to the package, with a `.mlmodelc` extension.
pub fn compiled_path(package: &Path) -> PathBuf {
  package.with_extension("mlmodelc")
}

/// Compiles `package` into a `.mlmodelc` directory next to it unless that exists already.
pub fn ensure_compiled(package: &Path) -> Result<PathBuf> {
  let compiled = compiled_path(package);
  if compiled.is_dir() {
    return Ok(compiled);
  }
  // SAFETY: plain Core ML call with a valid URL. The synchronous variant is deprecated in favor of
  // a callback-based one, but rclip needs the compiled model before it can continue anyway.
  #[allow(deprecated)]
  let temporary = unsafe { MLModel::compileModelAtURL_error(&file_url(package)) }
    .map_err(|error| ns_error("cannot compile the Core ML model", &error))?;
  let temporary = temporary.path().map(|path| PathBuf::from(path.to_string())).context("compiled model has no path")?;
  // Core ML compiles into a temporary directory that may be on another volume, so fall back to a
  // copy when a rename is not possible.
  if std::fs::rename(&temporary, &compiled).is_err() {
    copy_dir(&temporary, &compiled)?;
    std::fs::remove_dir_all(&temporary).ok();
  }
  Ok(compiled)
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
  std::fs::create_dir_all(to)?;
  for entry in std::fs::read_dir(from)? {
    let entry = entry?;
    let target = to.join(entry.file_name());
    if entry.file_type()?.is_dir() {
      copy_dir(&entry.path(), &target)?;
    } else {
      std::fs::copy(entry.path(), target)?;
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn converts_half_floats() {
    assert_eq!(f16_to_f32(0x3c00), 1.0);
    assert_eq!(f16_to_f32(0xc000), -2.0);
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
    assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
  }
}
