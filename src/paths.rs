//! Where rclip keeps its index and model files.

use std::path::PathBuf;

use anyhow::{Context, Result};

/// The data directory: `RCLIP_DATADIR` if set, otherwise the platform's application data
/// directory (`~/.local/share/rclip`, `~/Library/Application Support/rclip`, or
/// `%USERPROFILE%/AppData/Roaming/rclip`). Created on first use.
pub fn datadir() -> Result<PathBuf> {
  let datadir = match std::env::var_os("RCLIP_DATADIR").filter(|value| !value.is_empty()) {
    Some(datadir) => PathBuf::from(datadir),
    None => system_datadir()?.join("rclip"),
  };
  std::fs::create_dir_all(&datadir).with_context(|| format!("cannot create {}", datadir.display()))?;
  Ok(datadir)
}

fn system_datadir() -> Result<PathBuf> {
  let home = home_dir().context("cannot determine the home directory")?;
  Ok(if cfg!(windows) {
    home.join("AppData").join("Roaming")
  } else if cfg!(target_os = "macos") {
    home.join("Library").join("Application Support")
  } else {
    home.join(".local").join("share")
  })
}

pub fn home_dir() -> Option<PathBuf> {
  let variable = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
  std::env::var_os(variable).filter(|value| !value.is_empty()).map(PathBuf::from)
}

/// An optional shared download cache for model files, set with `RCLIP_MODEL_CACHE_DIR`.
pub fn model_cache_dir() -> Result<Option<PathBuf>> {
  let Some(cache_dir) = std::env::var_os("RCLIP_MODEL_CACHE_DIR").filter(|value| !value.is_empty()) else {
    return Ok(None);
  };
  let cache_dir = PathBuf::from(cache_dir);
  std::fs::create_dir_all(&cache_dir).with_context(|| format!("cannot create {}", cache_dir.display()))?;
  Ok(Some(cache_dir))
}
