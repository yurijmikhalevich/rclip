//! Downloads the model files from Hugging Face into the rclip data directory.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

const REPO_ID: &str = "yurijmikhalevich/rclip-models";
pub const MODEL_SUBDIR: &str = "ViT-B-32-256-datacomp_s34b_b86k";
const VISUAL_ONNX: &str = "visual.onnx";
const TEXTUAL_ONNX: &str = "textual.onnx";
const VISUAL_COREML: &str = "visual.mlpackage";
const TOKENIZER_VOCAB: &str = "tokenizer/bpe_simple_vocab_16e6.txt.gz";
const LEGAL_FILES: &[&str] = &["LICENSE", "OPENAI-CLIP-MIT.txt", "README.md", "THIRD_PARTY_NOTICES.md"];
pub const USE_ONNX_ON_MACOS_ENV_VAR: &str = "RCLIP_USE_ONNX_ON_MACOS";

/// Locations of the model files inside the data directory.
#[derive(Clone, Debug)]
pub struct ModelFiles {
  datadir: PathBuf,
}

impl ModelFiles {
  pub fn new(datadir: &Path) -> Self {
    ModelFiles { datadir: datadir.to_path_buf() }
  }

  pub fn textual(&self) -> PathBuf {
    self.datadir.join(MODEL_SUBDIR).join(TEXTUAL_ONNX)
  }

  pub fn visual(&self) -> PathBuf {
    self.datadir.join(MODEL_SUBDIR).join(VISUAL_ONNX)
  }

  pub fn visual_coreml(&self) -> PathBuf {
    self.datadir.join(MODEL_SUBDIR).join(VISUAL_COREML)
  }

  pub fn tokenizer_vocab(&self) -> PathBuf {
    self.datadir.join(TOKENIZER_VOCAB)
  }

  /// Downloads whatever model files are missing, showing a single progress bar for all of them.
  pub fn ensure_downloaded(&self) -> Result<()> {
    let mut missing: Vec<String> = Vec::new();
    let mut missing_package = false;
    for (path, relative) in [
      (self.visual(), format!("{MODEL_SUBDIR}/{VISUAL_ONNX}")),
      (self.textual(), format!("{MODEL_SUBDIR}/{TEXTUAL_ONNX}")),
      (self.tokenizer_vocab(), TOKENIZER_VOCAB.to_string()),
    ] {
      if !path.is_file() {
        missing.push(relative);
      }
    }
    if use_coreml_for_visual_index() && !self.visual_coreml().is_dir() {
      missing_package = true;
    }
    missing.extend(LEGAL_FILES.iter().filter(|name| !self.datadir.join(name).is_file()).map(|name| name.to_string()));

    if !missing.is_empty() || missing_package {
      let hub = Hub::new()?;
      let files = hub.list_files()?;
      let package_prefix = format!("{MODEL_SUBDIR}/{VISUAL_COREML}/");
      let package_files: Vec<&RepoFile> = if missing_package {
        files.iter().filter(|file| file.path.starts_with(&package_prefix)).collect()
      } else {
        vec![]
      };
      let wanted: Vec<&RepoFile> = files.iter().filter(|file| missing.contains(&file.path)).collect();
      let total = wanted.iter().chain(&package_files).map(|file| file.size).sum();
      let progress = progress_bar(total);
      for file in wanted {
        hub.download(file, &self.datadir.join(&file.path), &progress)?;
      }
      if missing_package {
        // Download the package into a staging directory so that an interrupted download is never
        // mistaken for a complete package.
        let staging = self.visual_coreml().with_extension("mlpackage.incomplete");
        fs::remove_dir_all(&staging).ok();
        for file in package_files {
          hub.download(file, &staging.join(&file.path[package_prefix.len()..]), &progress)?;
        }
        fs::rename(&staging, self.visual_coreml())?;
      }
      progress.finish_and_clear();
    }

    #[cfg(target_os = "macos")]
    if use_coreml_for_visual_index() {
      super::coreml::ensure_compiled(&self.visual_coreml())?;
    }
    Ok(())
  }
}

pub fn use_coreml_for_visual_index() -> bool {
  cfg!(target_os = "macos") && std::env::var_os(USE_ONNX_ON_MACOS_ENV_VAR).is_none_or(|value| value.is_empty())
}

fn progress_bar(total: u64) -> ProgressBar {
  let progress = ProgressBar::with_draw_target(Some(total), indicatif::ProgressDrawTarget::stderr());
  progress.set_style(
    ProgressStyle::with_template(
      "{msg}: {percent:>3}%|{wide_bar}| {bytes}/{total_bytes} [{elapsed}<{eta}, {bytes_per_sec}]",
    )
    .expect("the progress template is valid"),
  );
  progress.set_message("Downloading the model");
  progress
}

#[derive(Debug, Deserialize)]
struct RepoInfo {
  siblings: Vec<Sibling>,
}

#[derive(Debug, Deserialize)]
struct Sibling {
  rfilename: String,
  size: Option<u64>,
}

#[derive(Debug)]
struct RepoFile {
  path: String,
  size: u64,
}

struct Hub {
  agent: ureq::Agent,
  endpoint: String,
  /// `HF_TOKEN`, which raises Hugging Face's rate limits when requests are authenticated with it.
  token: Option<String>,
}

impl Hub {
  fn new() -> Result<Self> {
    let agent = ureq::Agent::config_builder()
      .user_agent(concat!("rclip/", env!("CARGO_PKG_VERSION"), " (https://github.com/yurijmikhalevich/rclip)"))
      .build()
      .into();
    let endpoint =
      std::env::var("HF_ENDPOINT").ok().filter(|value| !value.is_empty()).unwrap_or("https://huggingface.co".into());
    let token = std::env::var("HF_TOKEN").ok().filter(|value| !value.is_empty());
    let endpoint = endpoint.trim_end_matches('/').to_string();
    Ok(Hub { agent, endpoint, token })
  }

  fn get(&self, url: &str) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
    let request = self.agent.get(url);
    match &self.token {
      Some(token) => request.header("Authorization", format!("Bearer {token}")),
      None => request,
    }
  }

  fn list_files(&self) -> Result<Vec<RepoFile>> {
    let url = format!("{}/api/models/{REPO_ID}?blobs=true", self.endpoint);
    let info: RepoInfo = self
      .get(&url)
      .call()
      .and_then(|mut response| response.body_mut().read_json())
      .with_context(|| format!("cannot list the model files at {url}"))?;
    Ok(
      info
        .siblings
        .into_iter()
        .map(|sibling| RepoFile { size: sibling.size.unwrap_or(0), path: sibling.rfilename })
        .collect(),
    )
  }

  fn download(&self, file: &RepoFile, target: &Path, progress: &ProgressBar) -> Result<()> {
    let url = format!("{}/{REPO_ID}/resolve/main/{}", self.endpoint, file.path);
    let mut response = self.get(&url).call().with_context(|| format!("cannot download {url}"))?;
    let mut reader = response.body_mut().with_config().limit(u64::MAX).reader();
    let partial = partial_path(target);
    fs::create_dir_all(target.parent().expect("model files live in a directory"))?;
    let mut output = fs::File::create(&partial).with_context(|| format!("cannot write {}", partial.display()))?;
    let mut buffer = vec![0u8; 1 << 20];
    let mut received = 0u64;
    loop {
      let read = reader.read(&mut buffer).with_context(|| format!("cannot download {url}"))?;
      if read == 0 {
        break;
      }
      output.write_all(&buffer[..read])?;
      received += read as u64;
      progress.inc(read as u64);
    }
    output.sync_all()?;
    if file.size != 0 && received != file.size {
      bail!("the download of {url} is incomplete: received {received} of {} bytes", file.size);
    }
    fs::rename(&partial, target)?;
    Ok(())
  }
}

fn partial_path(target: &Path) -> PathBuf {
  let mut name = target.file_name().expect("model files have names").to_os_string();
  name.push(".incomplete");
  target.with_file_name(name)
}
