//! End-to-end tests of what rclip shows in a terminal: image previews and the interactive UI.
//!
//! The interactive UI runs in a pseudo-terminal that the tests type into, with a script standing in
//! for Kitty's `kitten`. The ignored tests need the real one instead; run them inside a Kitty
//! terminal with `cargo test --release --test terminal -- --ignored`.
//!
//! Like the tests in `e2e.rs`, the ones that need the model share it in one data directory and run
//! one at a time, each starting from an empty index.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard};

use base64::Engine;

struct Run {
  _guard: MutexGuard<'static, ()>,
  datadir: PathBuf,
}

impl Run {
  fn new() -> Self {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let datadir = std::env::var_os("RCLIP_TEST_DATADIR")
      .map(PathBuf::from)
      .unwrap_or_else(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("e2e-datadir"));
    std::fs::create_dir_all(&datadir).unwrap();
    std::fs::remove_file(datadir.join("db.sqlite3")).ok();
    Run { _guard: guard, datadir }
  }

  fn rclip(&self, directory: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_rclip"));
    command.current_dir(directory).env("RCLIP_DATADIR", &self.datadir);
    command
  }
}

#[test]
fn rejects_unusable_terminals_and_arguments_before_setting_up() {
  let directory = tempfile::tempdir().unwrap();
  let datadir = directory.path().join("data");
  let run = |args: &[&str]| {
    let output = Command::new(env!("CARGO_BIN_EXE_rclip"))
      .args(args)
      .current_dir(directory.path())
      .env("RCLIP_DATADIR", &datadir)
      .output()
      .unwrap();
    assert!(!datadir.exists(), "{args:?} got as far as creating the data directory");
    (output.status.code(), String::from_utf8(output.stderr).unwrap())
  };
  // The output is captured here, so it is not a terminal.
  let not_a_terminal = |mode: &str| (Some(1), format!("rclip: {mode} requires a terminal (stdout is not a TTY)\n"));
  assert_eq!(run(&["--interactive"]), not_a_terminal("interactive mode"));
  assert_eq!(run(&["cat", "--preview"]), not_a_terminal("image preview"));
  for args in [
    &[][..],
    &["--interactive", "cat"],
    &["--interactive", "--add", "cat"],
    &["--interactive", "--subtract", "cat"],
    &["--interactive", "--top", "101"],
    &["--interactive", "--preview"],
    &["--force-interactive", "--preview"],
    &["--force-preview", "--interactive"],
    &["--force-interactive", "--force-preview"],
    &["--force-interactive", "--filepath-only"],
    &["cat", "--force-preview", "--filepath-only"],
  ] {
    assert_eq!(run(args).0, Some(2), "{args:?}");
  }
}

#[test]
fn previews_results_as_resized_pngs_in_kitty_chunks() {
  let run = Run::new();
  let directory = tempfile::tempdir().unwrap();
  // Noise compresses badly, so the preview at its full size needs several chunks.
  let mut noise = 1u32;
  let pixels = std::iter::repeat_with(|| {
    noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    noise.to_be_bytes()[0]
  });
  image::RgbImage::from_raw(200, 100, pixels.take(200 * 100 * 3).collect())
    .unwrap()
    .save(directory.path().join("noise.png"))
    .unwrap();

  const TMUX: Option<&str> = Some("/tmp/tmux-test/default,123,0");
  for (term, tmux, wrapped, height) in [
    ("xterm-kitty", None, false, 1),
    ("xterm-256color", Some(""), false, 50),
    ("screen-256color", None, false, 400),
    ("screen-256color", TMUX, true, 1),
    ("xterm-256color", TMUX, true, 50),
    ("tmux-256color", None, true, 400),
  ] {
    let mut command = run.rclip(directory.path());
    command.args(["noise", "--force-preview", "--preview-height", &height.to_string()]).env("TERM", term);
    match tmux {
      Some(tmux) => command.env("TMUX", tmux),
      None => command.env_remove("TMUX"),
    };
    let output = command.output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let (results, preview) = stdout.split_at(stdout.find('\x1b').expect("a preview is printed"));
    assert!(results.ends_with("noise.png\"\n"), "{results:?}");

    let (prefix, suffix) = if wrapped { ("\x1bPtmux;\x1b\x1b_G", "\x1b\x1b\\\x1b\\") } else { ("\x1b_G", "\x1b\\") };
    let mut commands: Vec<&str> =
      preview.strip_suffix('\n').expect("a newline ends the preview").split(suffix).collect();
    assert_eq!(commands.pop(), Some(""));
    let mut payload = String::new();
    for (index, command) in commands.iter().enumerate() {
      let command = command.strip_prefix(prefix).unwrap_or_else(|| panic!("{term} {tmux:?}: {command:?}"));
      let (header, chunk) = command.split_once(';').unwrap();
      let more = u8::from(index + 1 < commands.len());
      assert_eq!(header, if index == 0 { format!("a=T,f=100,q=2,m={more}") } else { format!("q=2,m={more}") });
      assert!(!chunk.is_empty() && chunk.len() <= 4096);
      payload.push_str(chunk);
    }
    assert!(commands.len() > 1 || height < 400);
    let png = base64::engine::general_purpose::STANDARD.decode(payload).unwrap();
    assert_eq!(image::guess_format(&png).unwrap(), image::ImageFormat::Png);
    let shown = image::load_from_memory(&png).unwrap();
    // Previews keep the image's aspect ratio and never enlarge it.
    assert_eq!((shown.width(), shown.height()), (2 * height.min(100), height.min(100)));
  }
}

#[cfg(unix)]
mod interactive {
  use std::fs::File;
  use std::io::{Read, Write};
  use std::os::fd::{FromRawFd, OwnedFd};
  use std::os::unix::fs::PermissionsExt;
  use std::os::unix::process::CommandExt;
  use std::process::Child;
  use std::sync::Arc;
  use std::time::{Duration, Instant, SystemTime};

  use super::*;

  /// A directory with an 80x60 image per name, and the images' paths. Each image is older than the
  /// one before it, which is the order the UI lists them in.
  fn images(names: &[&str]) -> (tempfile::TempDir, Vec<String>) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let paths = names.iter().zip(0..).map(|(name, age)| {
      let path = root.join(name);
      image::RgbImage::from_pixel(80, 60, image::Rgb([200, 30, 30])).save(&path).unwrap();
      let modified = SystemTime::now() - Duration::from_secs(60 * age);
      File::options().write(true).open(&path).unwrap().set_modified(modified).unwrap();
      path.to_str().unwrap().to_string()
    });
    let paths = paths.collect();
    (directory, paths)
  }

  const DOWN: &str = "\x1b[B";
  const RIGHT: &str = "\x1b[C";
  const CTRL_C: &str = "\x03";
  const SCREEN_ENTERED: &str = "\x1b[?1049h";
  /// Starting may have to download the model.
  const STARTUP_TIMEOUT: Duration = Duration::from_secs(600);
  const TIMEOUT: Duration = Duration::from_secs(60);

  /// The text on the screen of a terminal that received `output`: just enough of a terminal to follow
  /// where the UI moves the cursor to and what it writes there.
  fn screen_text(output: &str) -> String {
    let mut rows: Vec<Vec<char>> = Vec::new();
    let (mut row, mut column) = (0, 0);
    let mut chars = output.chars().peekable();
    while let Some(c) = chars.next() {
      match c {
        '\x1b' => match chars.next() {
          Some('[') => {
            let parameters: String = std::iter::from_fn(|| chars.next_if(|c| !('\x40'..='\x7e').contains(c))).collect();
            match chars.next() {
              Some('H') => {
                let mut position =
                  parameters.split(';').map(|number| number.parse().unwrap_or(1usize).saturating_sub(1));
                (row, column) = (position.next().unwrap_or(0), position.next().unwrap_or(0));
              }
              Some('J') => rows.clear(),
              _ => {}
            }
          }
          // Images, clipboard writes, and terminal queries end with ST or BEL.
          Some('_' | ']' | 'P') => {
            while let Some(c) = chars.next() {
              if c == '\x07' || (c == '\x1b' && chars.next_if_eq(&'\\').is_some()) {
                break;
              }
            }
          }
          _ => {}
        },
        '\r' => column = 0,
        '\n' => row += 1,
        c if !c.is_control() => {
          // A cell of an image is a placeholder followed by two combining characters.
          let symbol = if c == '\u{10EEEE}' { chars.nth(1).map_or(c, |_| '▒') } else { c };
          rows.resize(rows.len().max(row + 1), Vec::new());
          let cells = &mut rows[row];
          cells.resize(cells.len().max(column + 1), ' ');
          cells[column] = symbol;
          column += 1;
        }
        _ => {}
      }
    }
    rows.iter().map(|row| row.iter().collect::<String>()).collect::<Vec<_>>().join("\n")
  }

  /// rclip's interactive UI running in a pseudo-terminal.
  struct Tui {
    child: Child,
    controller: File,
    output: Arc<Mutex<Vec<u8>>>,
  }

  impl Tui {
    /// Starts `command` as the interactive UI in a 120x40 pseudo-terminal. With `own_terminal` it is
    /// the UI's controlling terminal, like in a real terminal session; without, the programs the UI
    /// runs still talk to the terminal the tests run in.
    fn start(mut command: Command, own_terminal: bool) -> Self {
      // SAFETY: openpty fills in the two descriptors, which are then owned by the returned values.
      let (controller, device) = unsafe {
        let (mut controller, mut device) = (0, 0);
        let mut size = libc::winsize { ws_row: 40, ws_col: 120, ws_xpixel: 1200, ws_ypixel: 800 };
        assert_eq!(
          libc::openpty(&mut controller, &mut device, std::ptr::null_mut(), std::ptr::null_mut(), &raw mut size),
          0
        );
        (File::from_raw_fd(controller), OwnedFd::from_raw_fd(device))
      };
      command
        .arg("--force-interactive")
        .stdin(device.try_clone().unwrap())
        .stdout(device.try_clone().unwrap())
        .stderr(device);
      if own_terminal {
        // SAFETY: setsid and ioctl are async-signal-safe.
        unsafe {
          command.pre_exec(|| {
            libc::setsid();
            libc::ioctl(0, libc::TIOCSCTTY as _, 0);
            Ok(())
          });
        }
      }
      let child = command.spawn().unwrap();
      let output = Arc::new(Mutex::new(Vec::new()));
      let (mut reader, collected) = (controller.try_clone().unwrap(), Arc::clone(&output));
      std::thread::spawn(move || {
        let mut buffer = [0u8; 4096];
        while let Ok(read) = reader.read(&mut buffer) {
          if read == 0 {
            break;
          }
          collected.lock().unwrap().extend_from_slice(&buffer[..read]);
        }
      });
      let tui = Tui { child, controller, output };
      let shows_results = |tui: &Tui| tui.screen().contains("Quit") && !tui.screen().contains("Searching…");
      tui.wait_until("the UI to show its first results", STARTUP_TIMEOUT, shows_results);
      tui
    }

    /// Everything the UI wrote to the terminal so far.
    fn output(&self) -> String {
      String::from_utf8_lossy(&self.output.lock().unwrap()).into_owned()
    }

    fn screen(&self) -> String {
      screen_text(&self.output())
    }

    fn send(&mut self, keys: &str) {
      self.controller.write_all(keys.as_bytes()).unwrap();
    }

    /// Clicks the left mouse button `times` at a zero-based position.
    fn click(&mut self, (column, row): (usize, usize), times: usize) {
      let click = format!("\x1b[<0;{0};{1}M\x1b[<0;{0};{1}m", column + 1, row + 1);
      self.send(&click.repeat(times));
    }

    fn wait_until(&self, what: &str, timeout: Duration, done: impl Fn(&Self) -> bool) {
      let started = Instant::now();
      while !done(self) {
        assert!(started.elapsed() < timeout, "timed out waiting for {what}; the screen shows:\n{}", self.screen());
        std::thread::sleep(Duration::from_millis(20));
      }
    }

    fn wait_for(&self, text: &str) {
      self.wait_until(text, TIMEOUT, |tui| tui.screen().contains(text));
    }

    /// Waits until the UI has entered its screen `times`: once at the start, and once more after each
    /// time it left the screen to `kitten`.
    fn wait_for_screen(&self, times: usize) {
      self.wait_until("the UI to return", TIMEOUT, |tui| tui.output().matches(SCREEN_ENTERED).count() == times);
    }

    /// The zero-based position of `text` on the screen.
    fn position_of(&self, text: &str) -> (usize, usize) {
      let screen = self.screen();
      let found = screen.lines().enumerate().find_map(|(row, line)| Some((line.find(text)?, row, line)));
      let (index, row, line) = found.unwrap_or_else(|| panic!("{text:?} is not on the screen:\n{screen}"));
      (line[..index].chars().count(), row)
    }

    fn quit(mut self) {
      self.send(CTRL_C);
      let started = Instant::now();
      let status = loop {
        if let Some(status) = self.child.try_wait().unwrap() {
          break status;
        }
        assert!(started.elapsed() < TIMEOUT, "the UI did not exit");
        std::thread::sleep(Duration::from_millis(20));
      };
      assert!(status.success(), "{status}");
    }
  }

  impl Drop for Tui {
    fn drop(&mut self) {
      self.child.kill().ok();
      self.child.wait().ok();
    }
  }

  /// The UI's command with only the programs in `path` available and no hints of Kitty, tmux, or SSH.
  fn tui_command(run: &Run, directory: &Path, path: &Path) -> Command {
    let mut command = run.rclip(directory);
    command.env("PATH", path).env("TERM", "xterm-256color");
    for variable in ["TMUX", "KITTY_INSTALLATION_DIR", "SSH_CONNECTION", "SSH_TTY"] {
      command.env_remove(variable);
    }
    command
  }

  /// A script standing in for Kitty's `kitten`: it logs its arguments, keeps what it is told to copy,
  /// and behaves as the files next to it say.
  struct Kitten(tempfile::TempDir);

  impl Kitten {
    const SCRIPT: &str = r#"#!/bin/sh
here=${0%/*}
printf '%s\n' "$*" >> "$here/log"
case "$1 $2" in
  "query_terminal --help") test -e "$here/terminal" ;;
  "query_terminal --wait-for")
    if test -e "$here/hangs"; then exec /bin/sleep 60; fi
    /bin/cat "$here/terminal" ;;
  *)
    if test -e "$here/denied"; then
      echo "permission denied" >&2
      exit 1
    fi
    if test "$1" = clipboard; then /bin/cp "$2" "$here/copied"; fi ;;
esac
"#;

    /// `terminal` is the name of the terminal the kitten finds; an old kitten (`None`) cannot ask for it.
    fn new(terminal: Option<&str>) -> Self {
      let kitten = Kitten(tempfile::tempdir().unwrap());
      let script = kitten.path().join("kitten");
      std::fs::write(&script, Self::SCRIPT).unwrap();
      std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
      if let Some(terminal) = terminal {
        kitten.finds(terminal);
      }
      kitten
    }

    fn path(&self) -> &Path {
      self.0.path()
    }

    fn finds(&self, terminal: &str) {
      std::fs::write(self.path().join("terminal"), format!("name: {terminal}\n")).unwrap();
    }

    /// The arguments of each run so far.
    fn log(&self) -> Vec<String> {
      std::fs::read_to_string(self.path().join("log")).unwrap_or_default().lines().map(String::from).collect()
    }
  }

  const ASKS_FOR_THE_TERMINAL: [&str; 2] = ["query_terminal --help", "query_terminal --wait-for 1 name"];

  #[test]
  fn copies_and_downloads_the_selected_image_through_kitten() {
    let run = Run::new();
    let (directory, paths) = images(&["cat.jpg", "dog.ppm"]);
    let kitten = Kitten::new(Some("kitty"));
    let mut command = tui_command(&run, directory.path(), kitten.path());
    command.env("SSH_CONNECTION", "client 1 server 2");
    let mut tui = Tui::start(command, true);
    tui.wait_for("dog.ppm");
    assert_eq!(kitten.log(), [""; 0]);

    tui.send(DOWN);
    tui.wait_for(&paths[0]);
    tui.send("y");
    tui.wait_for("Image copied");
    assert_eq!(kitten.log(), [ASKS_FOR_THE_TERMINAL[0], ASKS_FOR_THE_TERMINAL[1], &format!("clipboard {}", paths[0])]);
    assert_eq!(std::fs::read(kitten.path().join("copied")).unwrap(), std::fs::read(&paths[0]).unwrap());

    // The UI leaves its screen to kitten, says why, and ends kitten's terminal query before it returns.
    let output = tui.output();
    let mut returned = 0;
    for step in [SCREEN_ENTERED, "\x1b[?1049l", "rclip: pausing the TUI to copy the image…", "\x1b\\\x1b[?1049h"] {
      let found = output[returned..].find(step).unwrap_or_else(|| panic!("{step:?} is missing or out of order"));
      returned += found + step.len();
    }
    // The terminal dropped the images along with the screen, so both are sent again.
    let sent_again = |tui: &Tui| tui.output()[returned..].matches(",f=100,q=2,m=0;").count() >= 2;
    tui.wait_until("the images to be sent again", TIMEOUT, sent_again);

    // Kitty was confirmed by the first copy. An uncommon format is copied as a PNG.
    tui.send(RIGHT);
    tui.wait_for(&paths[1]);
    tui.send("y");
    tui.wait_for_screen(3);
    let log = kitten.log();
    assert!(log.len() == 4 && log[3].starts_with("clipboard ") && log[3].ends_with("image.png"), "{log:?}");
    let copied = std::fs::read(kitten.path().join("copied")).unwrap();
    assert_eq!(image::guess_format(&copied).unwrap(), image::ImageFormat::Png);
    let copied = image::load_from_memory(&copied).unwrap();
    assert_eq!((copied.width(), copied.height()), (80, 60));

    tui.send("s");
    tui.wait_for("Saved to ~/Downloads");
    assert_eq!(kitten.log()[4..], [format!("transfer {} Downloads/", paths[1])]);
    assert!(tui.output().contains("rclip: pausing the TUI to download the image…"));

    tui.send("p");
    tui.wait_for("Path copied");
    let path = base64::engine::general_purpose::STANDARD.encode(&paths[1]);
    assert!(tui.output().contains(&format!("\x1b]52;c;{path}\x07")));
    tui.quit();
  }

  #[test]
  fn reports_failed_copies_and_downloads_and_tries_again_later() {
    let run = Run::new();
    let (directory, paths) = images(&["cat.jpg"]);
    let kitten = Kitten::new(Some("xterm-kitty"));
    let mut command = tui_command(&run, directory.path(), kitten.path());
    // Only what the terminal answers counts, not what the environment suggests.
    command.env("SSH_TTY", "/dev/pts/0").env("TERM", "xterm-kitty").env("KITTY_WINDOW_ID", "1");
    let mut tui = Tui::start(command, true);
    tui.send(DOWN);
    tui.send("y");
    tui.wait_for("Unable to copy image");
    assert!(tui.screen().contains("Image copy and download require Kitty"));
    tui.send("s");
    tui.wait_for("Unable to download image");
    // Nothing is sent to a terminal that is not Kitty.
    assert_eq!(kitten.log(), [ASKS_FOR_THE_TERMINAL, ASKS_FOR_THE_TERMINAL].concat());

    // The terminal is asked again after a failure, such as an answer that takes too long.
    kitten.finds("kitty");
    let hangs = kitten.path().join("hangs");
    std::fs::write(&hangs, "").unwrap();
    tui.send("y");
    tui.wait_for("Kitty did not respond in time");
    std::fs::remove_file(hangs).unwrap();
    // kitten's own failures are reported too.
    let denied = kitten.path().join("denied");
    std::fs::write(&denied, "").unwrap();
    tui.send("y");
    tui.wait_for("permission denied");
    std::fs::remove_file(denied).unwrap();
    tui.send("y");
    tui.wait_for("Image copied");
    let copy = format!("clipboard {}", paths[0]);
    assert_eq!(kitten.log()[4..], [ASKS_FOR_THE_TERMINAL, ASKS_FOR_THE_TERMINAL, [&copy, &copy]].concat());
    // Each time, the UI ended kitten's terminal query before returning to its screen.
    assert_eq!(tui.output().matches("\x1b\\\x1b[?1049h").count(), 5);
    tui.quit();
  }

  #[test]
  fn an_old_kitten_copies_without_asking_for_the_terminal() {
    let run = Run::new();
    let (directory, paths) = images(&["cat.jpg"]);
    let kitten = Kitten::new(None);
    // A kitten that is not on the PATH is found in Kitty's installation directory.
    let mut command = tui_command(&run, directory.path(), directory.path());
    command.env("KITTY_INSTALLATION_DIR", kitten.path());
    let mut tui = Tui::start(command, true);
    tui.send(DOWN);
    tui.send("y");
    tui.wait_for("Image copied");
    assert_eq!(kitten.log(), [ASKS_FOR_THE_TERMINAL[0], &format!("clipboard {}", paths[0])]);

    // Without SSH, the image is on this machine already.
    tui.send("s");
    tui.wait_for("Image is already local");
    assert_eq!(kitten.log().len(), 2);
    tui.quit();
  }

  #[test]
  fn a_missing_kitten_is_reported_without_leaving_the_screen() {
    let run = Run::new();
    let (directory, _) = images(&["cat.jpg"]);
    let mut command = tui_command(&run, directory.path(), directory.path());
    command.env("SSH_CONNECTION", "client 1 server 2");
    let mut tui = Tui::start(command, true);
    tui.send(DOWN);
    tui.send("y");
    tui.wait_for("Unable to copy image");
    assert!(tui.screen().contains("could not find Kitty's `kitten` executable"));
    tui.send("s");
    tui.wait_for("Unable to download image");
    assert_eq!(tui.output().matches(SCREEN_ENTERED).count(), 1);
    assert!(!tui.output().contains("pausing the TUI"));
    tui.quit();
  }

  #[test]
  fn the_mouse_selects_images_and_switches_views() {
    let run = Run::new();
    let (directory, paths) = images(&["cat.jpg", "dog.jpg", "owl.jpg"]);
    let mut tui = Tui::start(tui_command(&run, directory.path(), directory.path()), true);
    tui.wait_for("owl.jpg");

    // A double click selects the card and opens it in the detail view.
    tui.click(tui.position_of("dog.jpg"), 2);
    tui.wait_for("Grid view");
    tui.wait_for(&paths[1]);
    // The filmstrip's middle slot shows the selection; a click on the slot to its right moves there.
    tui.click((60 + 17, 33), 1);
    tui.wait_for(&paths[2]);
    // A double click on the image returns to the grid, where the search field takes the focus on a click.
    tui.click((60, 15), 2);
    tui.wait_until("the grid", TIMEOUT, |tui| !tui.screen().contains("Grid view"));
    tui.click((10, 2), 1);
    tui.wait_for(" Down ");
    tui.send(DOWN);
    tui.wait_for("hjkl");
    assert!(tui.screen().contains(&paths[2]));
    tui.quit();
  }

  fn clipboard_png() -> Vec<u8> {
    if cfg!(target_os = "macos") {
      let output = Command::new("osascript").args(["-e", "the clipboard as «class PNGf»"]).output().unwrap();
      let data = String::from_utf8(output.stdout).unwrap();
      let hex = data.trim().strip_prefix("«data PNGf").and_then(|hex| hex.strip_suffix('»')).expect("a PNG is copied");
      return (0..hex.len()).step_by(2).map(|index| u8::from_str_radix(&hex[index..index + 2], 16).unwrap()).collect();
    }
    let output = Command::new("xclip").args(["-selection", "clipboard", "-t", "image/png", "-o"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    output.stdout
  }

  /// Copies an image through the Kitty terminal the tests run in, by way of its real `kitten`, and
  /// reads it back from the system clipboard with `xclip` (or `osascript` on macOS).
  #[test]
  #[ignore = "needs to run inside a Kitty terminal"]
  fn kitty_copies_the_image_to_the_system_clipboard() {
    assert!(std::env::var_os("KITTY_WINDOW_ID").is_some(), "this test needs to run inside a Kitty terminal");
    let run = Run::new();
    let (directory, paths) = images(&["source.png"]);
    let mut tui = Tui::start(run.rclip(directory.path()), false);
    tui.send(DOWN);
    tui.send("y");
    tui.wait_for("Image copied");
    let copied = image::load_from_memory(&clipboard_png()).unwrap().to_rgb8();
    assert_eq!(copied, image::open(&paths[0]).unwrap().to_rgb8());
    tui.quit();
  }

  /// A plain pseudo-terminal answers neither Kitty's query protocol nor its clipboard requests.
  #[test]
  #[ignore = "needs Kitty's `kitten`"]
  fn kitty_copy_fails_fast_in_a_terminal_that_is_not_kitty() {
    let asks_for_the_terminal = Command::new("kitten").args(["query_terminal", "--help"]).output().unwrap();
    if !asks_for_the_terminal.status.success() {
      eprintln!("skipped: a `kitten` older than 0.38 cannot ask which terminal it runs in");
      return;
    }
    let run = Run::new();
    let (directory, _) = images(&["source.png"]);
    let mut tui = Tui::start(run.rclip(directory.path()), true);
    tui.send(DOWN);
    tui.send("y");
    tui.wait_for("Unable to copy image");
    assert!(!tui.screen().contains("Image copied"));
    tui.quit();
  }
}
