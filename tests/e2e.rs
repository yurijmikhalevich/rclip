//! End-to-end tests that run the rclip binary against the fixtures in `tests/e2e` and compare its
//! output with the snapshots. The Python version produced them, except for a few whose scores moved by a few
//! thousandths when rclip switched from Pillow's resizing to `fast_image_resize`.
//!
//! The first run downloads the model into a data directory under Cargo's target directory (or
//! `RCLIP_TEST_DATADIR`). The tests share that directory and run one at a time, each starting from
//! an empty index, like the Python test suite.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Mutex, MutexGuard};

/// Scores may differ by this much from the snapshots: Core ML and ONNX Runtime builds produce
/// slightly different floating-point results on different machines.
const SCORE_TOLERANCE: f64 = 0.0015;

fn lock() -> MutexGuard<'static, ()> {
  static LOCK: Mutex<()> = Mutex::new(());
  LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn datadir() -> PathBuf {
  let datadir = std::env::var_os("RCLIP_TEST_DATADIR")
    .map(PathBuf::from)
    .unwrap_or_else(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("e2e-datadir"));
  std::fs::create_dir_all(&datadir).unwrap();
  datadir
}

fn fixtures() -> PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("e2e")
}

struct Run {
  _guard: MutexGuard<'static, ()>,
  datadir: PathBuf,
}

impl Run {
  fn new() -> Self {
    let guard = lock();
    let datadir = datadir();
    std::fs::remove_file(datadir.join("db.sqlite3")).ok();
    Run { _guard: guard, datadir }
  }

  fn rclip(&self, cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rclip"))
      .args(args)
      .current_dir(cwd)
      .env("RCLIP_DATADIR", &self.datadir)
      .output()
      .expect("rclip runs")
  }

  /// Runs rclip and returns its stdout with the image directory replaced by a placeholder.
  fn output(&self, cwd: &Path, args: &[&str]) -> (bool, String) {
    let output = self.rclip(cwd, args);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let directory = format!("{}{}", cwd.display(), std::path::MAIN_SEPARATOR);
    let normalized = stdout
      .replace(&directory, "<test_images_dir>")
      .replace("./", "<test_images_dir>")
      .replace(std::path::MAIN_SEPARATOR, "/")
      .replace("\r\n", "\n");
    (output.status.success(), normalized)
  }
}

#[derive(Debug)]
struct Line {
  score: Option<f64>,
  text: String,
}

fn parse(output: &str) -> Vec<Line> {
  output
    .lines()
    .map(|line| match line.split_once('\t').and_then(|(score, path)| Some((score.parse::<f64>().ok()?, path))) {
      Some((score, path)) => Line { score: Some(score), text: path.to_string() },
      None => Line { score: None, text: line.to_string() },
    })
    .collect()
}

/// Whether output matches the expected output, allowing small score differences and reordering of
/// results whose scores are within the tolerance of each other.
fn matches(actual: &str, expected: &str) -> bool {
  let (actual, expected) = (parse(actual), parse(expected));
  let same_lines = actual.len() == expected.len()
    && actual.iter().zip(&expected).all(|(a, e)| match (a.score, e.score) {
      (Some(_), Some(_)) => true,
      (None, None) => a.text == e.text,
      _ => false,
    });
  // Near-tied results may swap places, so each result is compared with the same image's expected
  // result, and the results must remain sorted by score.
  let sorted =
    actual.windows(2).all(|pair| pair[0].score.zip(pair[1].score).is_none_or(|(above, below)| above >= below));
  let (actual, expected) = (by_image(&actual), by_image(&expected));
  same_lines && sorted && actual.iter().zip(&expected).all(|(a, e)| a.0 == e.0 && (a.1 - e.1).abs() <= SCORE_TOLERANCE)
}

/// The results keyed by the number of the output they belong to and their image, sorted by that key.
fn by_image(lines: &[Line]) -> Vec<((usize, &str), f64)> {
  let mut output = 0;
  let mut results: Vec<_> = lines
    .iter()
    .filter_map(|line| {
      output += usize::from(line.score.is_none());
      Some(((output, line.text.as_str()), line.score?))
    })
    .collect();
  results.sort_by_key(|result| result.0);
  results
}

fn assert_matches_snapshot(actual: &str, snapshot: &str) {
  let expected = std::fs::read_to_string(fixtures().join("output_snapshots").join(format!("{snapshot}.txt")))
    .unwrap()
    .replace("\r\n", "\n");
  assert!(
    matches(actual, &expected),
    "output differs from the {snapshot} snapshot:\n--- actual\n{actual}--- expected\n{expected}"
  );
}

#[test]
fn snapshots_let_near_ties_swap_but_keep_each_image_s_score() {
  let expected = "score\tfilepath\n0.500\tA\n0.499\tB\n0.498\tC\n0.300\tD\n";
  assert!(matches(expected, expected));
  assert!(matches("score\tfilepath\n0.499\tB\n0.499\tC\n0.499\tA\n0.300\tD\n", expected));
  // Every image's score moved by more than the tolerance, though the scores by position barely did.
  assert!(!matches("score\tfilepath\n0.500\tC\n0.499\tB\n0.498\tA\n0.300\tD\n", expected));
  assert!(!matches("score\tfilepath\nA\nB\nC\nD\n", expected));
  assert!(!matches("score\tfilepath\n0.300\tD\n0.500\tA\n0.499\tB\n0.498\tC\n", expected));
  assert!(!matches("score\tfilepath\n0.500\tA\n0.499\tB\n0.498\tC\n", expected));
}

fn images() -> PathBuf {
  fixtures().join("images")
}

fn nested() -> PathBuf {
  fixtures().join("images nested directories")
}

fn raw() -> PathBuf {
  fixtures().join("images raw")
}

fn results(output: &str) -> Vec<(String, f64)> {
  parse(output).into_iter().filter_map(|line| Some((line.text.trim_matches('"').to_string(), line.score?))).collect()
}

fn snapshot_test(cwd: &Path, args: &[&str], snapshot: &str) -> String {
  let run = Run::new();
  let (success, output) = run.output(cwd, args);
  assert!(success, "rclip failed:\n{output}");
  assert_matches_snapshot(&output, snapshot);
  output
}

#[test]
fn search() {
  snapshot_test(&images(), &["kitty"], "test_search");
}

#[test]
fn search_formats() {
  for (query, extension) in [
    ("tree", "webp"),
    ("boats on a lake", "png"),
    ("chess knight", "tiff"),
    ("chess pawns", "bmp"),
    ("chess queen", "gif"),
    ("duck", "jp2"),
    ("kookaburras", "pnm"),
    ("magpie", "pbm"),
    ("parrot and flowers", "pgm"),
    ("parrot looks into the camera", "ppm"),
  ] {
    let output = snapshot_test(&images(), &[query], &format!("test_search_format[{extension}]"));
    assert!(results(&output).iter().any(|(path, _)| path.contains(extension)), "{output}");
  }
}

#[cfg(any(target_os = "macos", windows))]
#[test]
fn search_heic() {
  let run = Run::new();
  let directory = fixtures().join("images heic");
  let indexing = String::from_utf8(run.rclip(&directory, &["bee"]).stderr).unwrap();
  // rclip decodes HEIC with the operating system's codec, which Windows doesn't always have
  if cfg!(windows) && indexing.contains("Windows has no working HEIF decoder") {
    eprintln!("skipping: Windows has no HEIC codec installed");
    return;
  }
  let (success, output) = run.output(&directory, &["--no-indexing", "bee"]);
  assert!(success, "{output}");
  let results = results(&output);
  let names: Vec<_> = results.iter().map(|(path, _)| path.trim_start_matches("<test_images_dir>")).collect();
  assert_eq!(names, ["bee.heic", "bee.jpg"], "{indexing}");
  assert!((results[0].1 - 0.286).abs() <= 0.002 && (results[1].1 - 0.274).abs() <= 0.002, "{output}");
}

#[test]
fn search_animated_gif() {
  let output = snapshot_test(&images(), &["bee animated", "--top", "15"], "test_search_animated_gif");
  assert!(output.contains("bee_animated.gif"));
}

#[test]
fn search_by_image() {
  let query = images().join("cat.jpg");
  snapshot_test(&images(), &[query.to_str().unwrap()], "test_search_by_image");
}

#[test]
fn search_by_image_from_url() {
  snapshot_test(
    &images(),
    &[
      "https://raw.githubusercontent.com/yurijmikhalevich/rclip/5630d6279ee94f0cad823777433d7fbeb921d19e/tests/e2e/images/cat.jpg",
    ],
    "test_search_by_image_from_url",
  );
}

#[test]
fn search_by_non_existing_file() {
  let run = Run::new();
  let output = run.rclip(&images(), &["--json", "./non-existing-file.jpg"]);
  assert!(!output.status.success());
  assert!(output.stdout.is_empty());
  assert!(String::from_utf8_lossy(&output.stderr).contains("non-existing-file.jpg\" not found."));
}

#[test]
fn search_by_not_an_image() {
  let run = Run::new();
  let query = images().join("not-an-image.txt");
  let output = run.rclip(&images(), &["--json", query.to_str().unwrap()]);
  assert!(!output.status.success());
  assert!(output.stdout.is_empty());
  assert!(String::from_utf8_lossy(&output.stderr).contains("not-an-image.txt\" is not an image."));
}

#[test]
fn add_queries() {
  snapshot_test(&images(), &["kitty", "--add", "puppy", "-a", "roof", "+", "fence"], "test_add_queries");
}

#[test]
fn subtract_queries() {
  snapshot_test(&images(), &["kitty", "--subtract", "puppy", "-s", "roof", "-", "fence"], "test_subtract_queries");
}

#[test]
fn add_and_subtract_queries() {
  snapshot_test(&images(), &["kitty", "+", "roof", "-", "fence"], "test_add_and_subtract_queries");
}

#[test]
fn query_multipliers() {
  snapshot_test(&images(), &["kitty", "+", "2:night", "-", "0.5:fence"], "test_query_multipliers");
}

#[test]
fn combine_text_query_with_image_query() {
  let query = images().join("cat.jpg");
  snapshot_test(
    &images(),
    &[query.to_str().unwrap(), "-", "3:cat", "+", "2:bee"],
    "test_combine_text_query_with_image_query",
  );
}

#[test]
fn combine_image_query_with_text_query() {
  let query = images().join("cat.jpg");
  snapshot_test(
    &images(),
    &["kitty", "-", query.to_str().unwrap(), "+", "1.5:bee"],
    "test_combine_image_query_with_text_query",
  );
}

#[test]
fn search_empty_dir() {
  let empty = fixtures().join("empty_directory");
  std::fs::create_dir_all(&empty).unwrap();
  snapshot_test(&empty, &["kitty"], "test_search_empty_dir");
}

#[test]
fn search_dir_with_multiple_nested_directories() {
  snapshot_test(&nested(), &["kitty"], "test_search_dir_with_multiple_nested_directories");
}

#[test]
fn search_dir_with_deeply_nested_directories() {
  snapshot_test(&nested(), &["bee"], "test_search_dir_with_deeply_nested_directories");
}

#[test]
fn handles_addition_and_deletion_of_images() {
  let run = Run::new();
  let bee = nested().join("misc").join("bees").join("bee.jpg");
  let copy = bee.with_file_name("bee_copy.jpg");
  struct Cleanup(PathBuf);
  impl Drop for Cleanup {
    fn drop(&mut self) {
      std::fs::remove_file(&self.0).ok();
    }
  }
  let _cleanup = Cleanup(copy.clone());

  let mut combined = run.output(&nested(), &["bee"]).1;
  std::fs::copy(&bee, &copy).unwrap();
  combined += &run.output(&nested(), &["bee"]).1;
  std::fs::remove_file(&copy).unwrap();
  combined += &run.output(&nested(), &["bee"]).1;
  assert_matches_snapshot(&combined, "test_handles_addition_and_deletion_of_images");
}

#[test]
fn ignores_raw_files_if_raw_support_is_disabled() {
  snapshot_test(&raw(), &["boat on a lake"], "test_ignores_raw_files_if_raw_support_is_disabled");
}

#[test]
fn ignores_raw_if_there_is_a_png_named_the_same_way_in_the_same_dir() {
  snapshot_test(
    &raw(),
    &["--experimental-raw-support", "boat on a lake"],
    "test_ignores_raw_if_there_is_a_png_named_the_same_way_in_the_same_dir",
  );
}

#[test]
fn can_read_raw_images() {
  for (query, file, snapshot) in [
    ("green ears of rye", "DSC08882.ARW", "test_can_read_arw_images"),
    ("dragon in a cave", "RAW_CANON_400D_ARGB.CR2", "test_can_read_cr2_images"),
    ("two parrots on a tree branch", "DSC03671.dng", "test_can_read_dng_images"),
    ("a bowl of tomatoes, potatoes, and a lemon", "RAW_CANON_EOS_R5.CR3", "test_can_read_cr3_images"),
    ("a lake in a mountain valley", "RAW_NIKON_D1X.NEF", "test_can_read_nef_images"),
    ("green leaves", "RAW_OLYMPUS_E-10.ORF", "test_can_read_orf_images"),
    ("a red wooden house", "RAW_FUJIFILM_FINEPIX_S5000.RAF", "test_can_read_raf_images"),
    ("yellow autumn bushes", "RAW_PANASONIC_DMC-LX7.RW2", "test_can_read_rw2_images"),
  ] {
    let output = snapshot_test(&raw(), &["--experimental-raw-support", query], snapshot);
    assert!(results(&output)[0].0.ends_with(file), "{output}");
  }
}

#[test]
fn unicode_filepaths() {
  snapshot_test(&fixtures().join("images unicode"), &["鳥"], "test_unicode_filepaths");
}

#[test]
fn filepath_only_prints_paths() {
  let run = Run::new();
  let (success, output) = run.output(&images(), &["kitty", "-f", "-t", "3"]);
  assert!(success);
  assert_eq!(
    output,
    "<test_images_dir>black kitty.jpeg\n<test_images_dir>cat.jpg\n<test_images_dir>black cat at night.jpg\n"
  );
}

#[test]
fn null_terminates_paths() {
  let run = Run::new();
  let (success, output) = run.output(&images(), &["kitty", "-0", "-t", "3"]);
  assert!(success);
  assert_eq!(
    output,
    "<test_images_dir>black kitty.jpeg\0<test_images_dir>cat.jpg\0<test_images_dir>black cat at night.jpg\0"
  );
}

#[test]
fn json_output_matches_the_table() {
  let run = Run::new();
  let table = String::from_utf8(run.rclip(&images(), &["kitty"]).stdout).unwrap();
  let json = run.rclip(&images(), &["kitty", "--json", "--no-indexing"]);
  assert!(json.status.success());
  let expected: Vec<serde_json::Value> = results(&table)
    .into_iter()
    .map(|(filepath, score)| serde_json::json!({ "score": score, "filepath": filepath }))
    .collect();
  assert_eq!(serde_json::from_slice::<serde_json::Value>(&json.stdout).unwrap(), serde_json::Value::Array(expected));
}

#[test]
fn no_indexing_searches_the_existing_index_only() {
  let run = Run::new();
  let (success, output) = run.output(&images(), &["kitty", "--no-indexing"]);
  assert!(success);
  assert_eq!(output, "score\tfilepath\n");
}

#[test]
fn interactive_rejects_unsupported_terminal_in_non_tty() {
  let run = Run::new();
  let output = run.rclip(&fixtures().join("empty_directory"), &["--interactive"]);
  assert_eq!(output.status.code(), Some(1));
  assert!(String::from_utf8_lossy(&output.stderr).contains("rclip:"));
}

#[test]
fn rejects_conflicting_arguments() {
  let run = Run::new();
  for args in [&["--force-interactive", "--force-preview"][..], &["--interactive", "cat"], &[]] {
    let output = run.rclip(&images(), args);
    assert_eq!(output.status.code(), Some(2), "{args:?}");
  }
}

#[test]
fn prints_the_version() {
  let run = Run::new();
  let output = run.rclip(&images(), &["--version"]);
  assert_eq!(String::from_utf8(output.stdout).unwrap(), format!("rclip {}\n", env!("CARGO_PKG_VERSION")));
}

/// Runs the interactive UI in a pseudo-terminal, lets it start, and quits it with Ctrl+C.
#[cfg(unix)]
#[test]
fn interactive_starts_and_exits_cleanly() {
  use std::io::{Read, Write};
  use std::os::fd::{FromRawFd, OwnedFd};
  use std::os::unix::process::CommandExt;
  use std::time::{Duration, Instant};

  let run = Run::new();
  let (mut controller, device) = unsafe {
    let (mut controller, mut device) = (0, 0);
    let mut size = libc::winsize { ws_row: 40, ws_col: 120, ws_xpixel: 1200, ws_ypixel: 800 };
    assert_eq!(
      libc::openpty(&mut controller, &mut device, std::ptr::null_mut(), std::ptr::null_mut(), &raw mut size),
      0
    );
    (std::fs::File::from_raw_fd(controller), OwnedFd::from_raw_fd(device))
  };
  let mut command = Command::new(env!("CARGO_BIN_EXE_rclip"));
  command
    .arg("--force-interactive")
    .current_dir(fixtures().join("empty_directory"))
    .env("RCLIP_DATADIR", &run.datadir)
    .stdin(device.try_clone().unwrap())
    .stdout(device.try_clone().unwrap())
    .stderr(device);
  // SAFETY: setsid and ioctl are async-signal-safe; they make the pseudo-terminal the controlling
  // terminal of the child, like a real terminal session.
  unsafe {
    command.pre_exec(|| {
      libc::setsid();
      libc::ioctl(0, libc::TIOCSCTTY as _, 0);
      Ok(())
    });
  }
  let mut child = command.spawn().unwrap();

  // Read the output on another thread until the UI has drawn its hotkeys.
  let mut reader = controller.try_clone().unwrap();
  let output = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
  let collected = std::sync::Arc::clone(&output);
  std::thread::spawn(move || {
    let mut buffer = [0u8; 4096];
    while let Ok(read) = reader.read(&mut buffer) {
      if read == 0 {
        break;
      }
      collected.lock().unwrap().extend_from_slice(&buffer[..read]);
    }
  });
  let started = Instant::now();
  while !String::from_utf8_lossy(&output.lock().unwrap()).contains("Quit") {
    assert!(started.elapsed() < Duration::from_secs(60), "the interactive UI did not start");
    std::thread::sleep(Duration::from_millis(50));
  }
  controller.write_all(b"\x03").unwrap();
  let status = loop {
    if let Some(status) = child.try_wait().unwrap() {
      break status;
    }
    assert!(started.elapsed() < Duration::from_secs(60), "the interactive UI did not exit");
    std::thread::sleep(Duration::from_millis(50));
  };
  assert!(status.success(), "{status}");
}
