//! ONNX Runtime sessions for the text encoder and the visual encoder.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use ort::environment::Environment;
use ort::logging::LogLevel;
use ort::session::Session;
use ort::value::Tensor;

fn environment() -> Result<&'static Environment> {
  static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
  if ENVIRONMENT.get().is_none() {
    // Microsoft's builds of ONNX Runtime send usage telemetry by default; rclip opts out.
    let environment =
      ort::init().with_name("rclip").with_telemetry(false).build().context("cannot initialize ONNX Runtime")?;
    let _ = ENVIRONMENT.set(environment);
  }
  Ok(ENVIRONMENT.get().expect("the environment was just initialized"))
}

pub struct OnnxModel {
  // `Session::run` needs exclusive access.
  session: Mutex<Session>,
}

impl OnnxModel {
  pub fn load(path: &Path) -> Result<Self> {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let load = || -> ort::Result<Session> {
      Session::builder(environment().map_err(|error| ort::Error::new(error.to_string()))?)?
        .with_log_level(LogLevel::Error)?
        .with_intra_threads(threads)?
        .commit_from_file(path)
    };
    let session = load().with_context(|| format!("cannot load the model at {}", path.display()))?;
    Ok(OnnxModel { session: Mutex::new(session) })
  }

  /// Runs the model on a row-major input of the given shape and returns the flat output.
  pub fn run<T>(&self, shape: &[usize], input: Vec<T>) -> Result<Vec<f32>>
  where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + Clone + 'static,
  {
    let shape: Vec<i64> = shape.iter().map(|&dim| dim as i64).collect();
    let tensor = Tensor::from_array((shape, input)).context("cannot create the model input")?;
    let mut session = self.session.lock().expect("the session lock is never poisoned");
    let outputs = session.run(ort::inputs!["input" => tensor]).context("model inference failed")?;
    let (_, features) = outputs["output"].try_extract_tensor::<f32>().context("unexpected model output")?;
    Ok(features.to_vec())
  }
}
