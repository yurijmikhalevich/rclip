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
