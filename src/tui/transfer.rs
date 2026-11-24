//! Copies images to the clipboard and downloads them through Kitty's `kitten` helper.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use base64::Engine;

use crate::imaging;

/// Formats the clipboard accepts as they are; others are converted to PNG first.
const CLIPBOARD_NATIVE_EXTENSIONS: &[&str] = &["bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp"];

pub fn kitten_executable() -> Result<PathBuf, String> {
  if let Some(path) = find_in_path("kitten") {
    return Ok(path);
  }
  if let Some(directory) = std::env::var_os("KITTY_INSTALLATION_DIR") {
    let path = Path::new(&directory).join("kitten");
    if path.is_file() {
      return Ok(path);
    }
  }
  Err("could not find Kitty's `kitten` executable".into())
}

fn find_in_path(name: &str) -> Option<PathBuf> {
  std::env::split_paths(&std::env::var_os("PATH")?).map(|directory| directory.join(name)).find(|path| path.is_file())
}

fn run_kitten(executable: &Path, args: &[&str], timeout: Option<Duration>) -> Result<String, String> {
  let mut child = Command::new(executable)
    .args(args)
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .map_err(|error| error.to_string())?;
  // Drain both pipes while waiting so that a chatty kitten never blocks on a full pipe.
  let drain = |pipe: Option<Box<dyn Read + Send>>| {
    std::thread::spawn(move || {
      let mut text = String::new();
      if let Some(mut pipe) = pipe {
        pipe.read_to_string(&mut text).ok();
      }
      text
    })
  };
  let stdout = drain(child.stdout.take().map(|pipe| Box::new(pipe) as Box<dyn Read + Send>));
  let stderr = drain(child.stderr.take().map(|pipe| Box::new(pipe) as Box<dyn Read + Send>));
  let started = Instant::now();
  let status = loop {
    if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
      break status;
    }
    if timeout.is_some_and(|timeout| started.elapsed() > timeout) {
      child.kill().ok();
      child.wait().ok();
      return Err("Kitty did not respond in time".into());
    }
    std::thread::sleep(Duration::from_millis(10));
  };
  let output = stdout.join().unwrap_or_default();
  let error = stderr.join().unwrap_or_default();
  if !status.success() {
    let error = error.trim();
    return Err(if error.is_empty() { format!("kitten exited with {status}") } else { error.to_string() });
  }
  Ok(output)
}

/// Copying and downloading go through Kitty's remote control, so the terminal must be Kitty itself.
/// `kitten query_terminal` (Kitty 0.38+) confirms that; older kittens cannot tell, so they pass.
/// Only a passed check is remembered, so one that failed, e.g. on a slow reply, runs again next time.
fn require_kitty() -> Result<PathBuf, String> {
  static CONFIRMED: AtomicBool = AtomicBool::new(false);
  let executable = kitten_executable()?;
  if !CONFIRMED.load(Ordering::Relaxed) {
    if run_kitten(&executable, &["query_terminal", "--help"], None).is_ok() {
      let response =
        run_kitten(&executable, &["query_terminal", "--wait-for", "1", "name"], Some(Duration::from_secs(2)))?;
      if response.trim() != "name: kitty" {
        return Err("Image copy and download require Kitty".into());
      }
    }
    CONFIRMED.store(true, Ordering::Relaxed);
  }
  Ok(executable)
}

/// Copies an image to the clipboard, converting uncommon formats to PNG.
pub fn copy_image_to_clipboard(filepath: &str) -> Result<(), String> {
  let executable = require_kitty()?;
  if CLIPBOARD_NATIVE_EXTENSIONS.contains(&imaging::file_extension(filepath).as_str()) {
    return run_kitten(&executable, &["clipboard", filepath], Some(Duration::from_secs(30))).map(drop);
  }
  let image = imaging::read_display_image(filepath, None).map_err(|error| error.to_string())?;
  // A private directory with a random name, so that other users can't plant a file at the converted image's path.
  // Dropping it removes it.
  let directory = tempfile::Builder::new().prefix("rclip-clipboard-").tempdir().map_err(|error| error.to_string())?;
  let converted = directory.path().join("image.png");
  image
    .png()
    .map_err(|error| error.to_string())
    .and_then(|png| std::fs::write(&converted, png).map_err(|error| error.to_string()))
    .and_then(|()| {
      run_kitten(&executable, &["clipboard", &converted.to_string_lossy()], Some(Duration::from_secs(30))).map(drop)
    })
}

pub fn is_remote_session() -> bool {
  std::env::var_os("SSH_CONNECTION").is_some() || std::env::var_os("SSH_TTY").is_some()
}

/// Downloads the original image to `~/Downloads` on the machine running Kitty.
pub fn download_image(filepath: &str) -> Result<(), String> {
  let executable = require_kitty()?;
  run_kitten(&executable, &["transfer", filepath, "Downloads/"], None).map(drop)
}

/// The OSC 52 sequence that puts `text` on the system clipboard.
pub fn osc52_copy(text: &str) -> String {
  format!("\x1b]52;c;{}\x07", base64::engine::general_purpose::STANDARD.encode(text))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn encodes_osc52() {
    assert_eq!(osc52_copy("/a b.jpg"), "\x1b]52;c;L2EgYi5qcGc=\x07");
  }

  /// Runs a shell command in place of `kitten`.
  #[cfg(unix)]
  fn run_shell(command: &str, timeout: Option<Duration>) -> Result<String, String> {
    run_kitten(Path::new("/bin/sh"), &["-c", command], timeout)
  }

  #[cfg(unix)]
  #[test]
  fn returns_what_kitten_prints() {
    assert_eq!(run_shell("echo 'name: kitty'; echo noise >&2", None).unwrap(), "name: kitty\n");
  }

  #[cfg(unix)]
  #[test]
  fn reports_why_kitten_failed() {
    assert_eq!(run_shell("echo ' permission denied ' >&2; exit 1", None).unwrap_err(), "permission denied");
    let silent = run_shell("exit 3", None).unwrap_err();
    assert!(silent.starts_with("kitten exited with") && silent.contains('3'), "{silent}");
    assert!(run_kitten(Path::new("/nonexistent/kitten"), &[], None).is_err());
  }

  #[cfg(unix)]
  #[test]
  fn stops_a_kitten_that_does_not_respond_in_time() {
    let started = Instant::now();
    let error = run_shell("exec sleep 60", Some(Duration::from_millis(50))).unwrap_err();
    assert_eq!(error, "Kitty did not respond in time");
    assert!(started.elapsed() < Duration::from_secs(30));
  }
}
