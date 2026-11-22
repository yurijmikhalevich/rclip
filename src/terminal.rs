//! Detects whether the terminal can display images with the Kitty graphics protocol.

use std::io::IsTerminal;

/// What the support checks need to know about the environment.
pub struct Environment<'a> {
  pub var: &'a dyn Fn(&str) -> Option<String>,
  pub is_windows: bool,
  pub is_atty: bool,
}

impl Environment<'_> {
  fn get(&self, name: &str) -> Option<String> {
    (self.var)(name).filter(|value| !value.is_empty())
  }

  fn has(&self, name: &str) -> bool {
    (self.var)(name).is_some()
  }

  fn equals(&self, name: &str, expected: &str) -> bool {
    (self.var)(name).as_deref() == Some(expected)
  }

  fn is_kitty(&self) -> bool {
    self.get("KITTY_WINDOW_ID").is_some() || self.get("KITTY_PID").is_some() || self.equals("TERM", "xterm-kitty")
  }

  fn is_ghostty(&self) -> bool {
    self.get("GHOSTTY_RESOURCES_DIR").is_some()
      || self.equals("TERM_PROGRAM", "ghostty")
      || self.equals("TERM", "xterm-ghostty")
  }

  fn is_wezterm(&self) -> bool {
    self.equals("TERM_PROGRAM", "WezTerm")
      || self.get("WEZTERM_EXECUTABLE").is_some()
      || self.get("WEZTERM_PANE").is_some()
  }

  fn is_alacritty(&self) -> bool {
    self.equals("TERM", "alacritty") || self.get("ALACRITTY_LOG").is_some() || self.get("ALACRITTY_WINDOW_ID").is_some()
  }

  /// `None` if this is not iTerm2, otherwise whether its version supports Kitty graphics.
  fn iterm2_supported(&self) -> Option<bool> {
    if !self.equals("TERM_PROGRAM", "iTerm.app") {
      return None;
    }
    Some(match self.get("TERM_PROGRAM_VERSION") {
      Some(version) => parse_version(&version) >= (3, 5, 0),
      None => true,
    })
  }
}

pub fn current_environment<'a>(var: &'a dyn Fn(&str) -> Option<String>) -> Environment<'a> {
  Environment { var, is_windows: cfg!(windows), is_atty: std::io::stdout().is_terminal() }
}

pub fn env_var(name: &str) -> Option<String> {
  std::env::var(name).ok()
}

fn parse_version(version: &str) -> (u32, u32, u32) {
  let mut parts = version
    .split(|c: char| !c.is_ascii_digit())
    .filter(|part| !part.is_empty())
    .map(|part| part.parse::<u32>().unwrap_or(u32::MAX));
  (parts.next().unwrap_or(0), parts.next().unwrap_or(0), parts.next().unwrap_or(0))
}

/// Interactive mode needs Kitty graphics with Unicode placeholders.
pub fn interactive_support(env: &Environment<'_>) -> Result<(), String> {
  if !env.is_atty {
    return Err("interactive mode requires a terminal (stdout is not a TTY)".into());
  }
  if env.is_kitty() || env.is_ghostty() {
    return Ok(());
  }
  match env.iterm2_supported() {
    Some(true) => return Ok(()),
    Some(false) => {
      return Err(
        "iTerm2 versions prior to 3.5.0 do not support the Kitty Graphics Protocol. \
        Please update iTerm2. To run anyway, use --force-interactive."
          .into(),
      );
    }
    None => {}
  }
  if env.is_wezterm() {
    return Err(
      "WezTerm supports image previews (--preview), but does not support the Kitty Unicode placeholders \
      required for interactive mode (--interactive). Consider using Ghostty or Kitty. \
      To run anyway, use --force-interactive."
        .into(),
    );
  }
  if env.has("KONSOLE_VERSION") {
    return Err(
      "Konsole supports image previews (--preview), but does not support the Kitty Unicode placeholders \
      required for interactive mode (--interactive). Consider using Ghostty or Kitty. \
      To run anyway, use --force-interactive."
        .into(),
    );
  }
  Err(unsupported(
    env,
    "interactive",
    "Consider using Ghostty or Kitty (via WSL)",
    "Consider using Ghostty, Kitty, or iTerm2",
    "Consider using Ghostty or Kitty",
    "Kitty, Ghostty",
    "could not detect a terminal emulator that supports the Kitty Graphics Protocol with Unicode placeholders \
    (e.g. Kitty, Ghostty, or iTerm2 >= 3.5.0). To run anyway, use --force-interactive.",
  ))
}

/// Previews need baseline Kitty graphics.
pub fn preview_support(env: &Environment<'_>) -> Result<(), String> {
  if !env.is_atty {
    return Err("image preview requires a terminal (stdout is not a TTY)".into());
  }
  if env.is_kitty() || env.is_ghostty() {
    return Ok(());
  }
  match env.iterm2_supported() {
    Some(true) => return Ok(()),
    Some(false) => {
      return Err(
        "iTerm2 versions prior to 3.5.0 do not support the Kitty Graphics Protocol. \
        Please update iTerm2. To preview anyway, use --force-preview."
          .into(),
      );
    }
    None => {}
  }
  if env.is_wezterm() || env.has("KONSOLE_VERSION") {
    return Ok(());
  }
  Err(unsupported(
    env,
    "preview",
    "Consider using Ghostty or WezTerm",
    "Consider using Ghostty, Kitty, WezTerm, or iTerm2",
    "Consider using Ghostty, Kitty, or WezTerm",
    "Kitty, Ghostty, WezTerm",
    "could not detect a terminal emulator that supports the Kitty Graphics Protocol \
    (e.g. Kitty, Ghostty, WezTerm, Konsole, or iTerm2 >= 3.5.0). To preview anyway, use --force-preview.",
  ))
}

/// Explains why a known-unsupported terminal cannot show images.
fn unsupported(
  env: &Environment<'_>,
  mode: &str,
  windows_alternatives: &str,
  apple_alternatives: &str,
  alacritty_alternatives: &str,
  tmux_terminals: &str,
  unknown: &str,
) -> String {
  let (flag, verb) =
    if mode == "interactive" { ("--force-interactive", "run") } else { ("--force-preview", "preview") };
  if env.has("WT_SESSION") {
    return format!(
      "Windows Terminal does not support the Kitty Graphics Protocol. {windows_alternatives}. \
      To {verb} anyway, use {flag}."
    );
  }
  if env.is_windows {
    return format!(
      "Windows Command Prompt and PowerShell do not support the Kitty Graphics Protocol. \
      {windows_alternatives}. To {verb} anyway, use {flag}."
    );
  }
  if env.equals("TERM_PROGRAM", "Apple_Terminal") {
    return format!(
      "Apple Terminal (Terminal.app) does not support the Kitty Graphics Protocol. {apple_alternatives}. \
      To {verb} anyway, use {flag}."
    );
  }
  if env.is_alacritty() {
    return format!(
      "Alacritty does not support the Kitty Graphics Protocol. {alacritty_alternatives}. To {verb} anyway, use {flag}."
    );
  }
  if let Some(term) = (env.var)("TERM").filter(|term| term == "dumb" || term == "linux") {
    return format!(
      "terminal type '{term}' does not support the Kitty Graphics Protocol. To {verb} anyway, use {flag}."
    );
  }
  if env.has("TMUX") {
    return format!(
      "running inside tmux without outer terminal identification. \
      If your terminal supports Kitty graphics (e.g. {tmux_terminals}), ensure 'set -g allow-passthrough on' \
      is set in your .tmux.conf and pass {flag}."
    );
  }
  unknown.to_string()
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::collections::HashMap;

  type Support = fn(&Environment<'_>) -> Result<(), String>;

  /// Both checks, each with the flag it tells the user to override it with.
  const MODES: [(Support, &str); 2] =
    [(interactive_support, "--force-interactive"), (preview_support, "--force-preview")];

  fn check(vars: &[(&str, &str)], is_windows: bool, support: Support) -> Result<(), String> {
    let vars: HashMap<String, String> = vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let lookup = move |name: &str| vars.get(name).cloned();
    support(&Environment { var: &lookup, is_windows, is_atty: true })
  }

  #[test]
  fn supports_kitty_ghostty_and_new_iterm2() {
    for vars in [
      vec![("KITTY_WINDOW_ID", "1")],
      vec![("KITTY_PID", "42")],
      vec![("TERM", "xterm-kitty")],
      vec![("GHOSTTY_RESOURCES_DIR", "/path/to/ghostty")],
      vec![("TERM_PROGRAM", "ghostty")],
      vec![("TERM", "xterm-ghostty")],
      vec![("TERM_PROGRAM", "iTerm.app"), ("TERM_PROGRAM_VERSION", "3.5.0")],
      vec![("TERM_PROGRAM", "iTerm.app"), ("TERM_PROGRAM_VERSION", "3.5.2")],
      vec![("TERM_PROGRAM", "iTerm.app"), ("TERM_PROGRAM_VERSION", "3.5")],
      vec![("TERM_PROGRAM", "iTerm.app")],
    ] {
      assert!(check(&vars, false, interactive_support).is_ok(), "{vars:?}");
      assert!(check(&vars, false, preview_support).is_ok(), "{vars:?}");
    }
  }

  #[test]
  fn rejects_old_iterm2() {
    let vars = [("TERM_PROGRAM", "iTerm.app"), ("TERM_PROGRAM_VERSION", "3.4.19")];
    for (support, flag) in MODES {
      let reason = check(&vars, false, support).unwrap_err();
      assert!(reason.contains("prior to 3.5.0") && reason.contains(flag), "{reason}");
    }
  }

  #[test]
  fn wezterm_and_konsole_support_previews_only() {
    for vars in [
      vec![("TERM_PROGRAM", "WezTerm")],
      vec![("WEZTERM_EXECUTABLE", "/path/to/wezterm")],
      vec![("WEZTERM_PANE", "1")],
      vec![("KONSOLE_VERSION", "")],
    ] {
      let reason = check(&vars, false, interactive_support).unwrap_err();
      assert!(reason.contains("Unicode placeholders") && reason.contains("--force-interactive"), "{reason}");
      assert!(check(&vars, false, preview_support).is_ok(), "{vars:?}");
    }
  }

  #[test]
  fn explains_unsupported_terminals() {
    for (vars, is_windows, explanation) in [
      (vec![("WT_SESSION", "abcd-1234")], true, "Windows Terminal does not support"),
      (vec![], true, "Windows Command Prompt and PowerShell do not support"),
      (vec![("TERM_PROGRAM", "Apple_Terminal")], false, "Apple Terminal (Terminal.app) does not support"),
      (vec![("TERM", "alacritty")], false, "Alacritty does not support"),
      (vec![("ALACRITTY_LOG", "/tmp/alacritty.log")], false, "Alacritty does not support"),
      (vec![("ALACRITTY_WINDOW_ID", "1")], false, "Alacritty does not support"),
      (vec![("TERM", "dumb")], false, "terminal type 'dumb' does not support"),
      (vec![("TERM", "linux")], false, "terminal type 'linux' does not support"),
      (vec![("TMUX", "/tmp/tmux-1000/default,1234,0")], false, "running inside tmux without outer terminal"),
      (vec![], false, "could not detect a terminal emulator"),
    ] {
      for (support, flag) in MODES {
        let reason = check(&vars, is_windows, support).unwrap_err();
        assert!(reason.starts_with(explanation) && reason.contains(flag), "{vars:?}: {reason}");
      }
    }
    assert_eq!(
      check(&[("TERM_PROGRAM", "Apple_Terminal")], false, preview_support).unwrap_err(),
      "Apple Terminal (Terminal.app) does not support the Kitty Graphics Protocol. \
      Consider using Ghostty, Kitty, WezTerm, or iTerm2. To preview anyway, use --force-preview."
    );
    assert!(check(&[("TMUX", "/tmp/x")], false, interactive_support).unwrap_err().contains("allow-passthrough"));
  }

  #[test]
  fn recognizes_a_supported_terminal_inside_tmux_and_on_windows() {
    for (support, _) in MODES {
      assert!(check(&[("TMUX", "/tmp/x"), ("KITTY_WINDOW_ID", "1")], false, support).is_ok());
      assert!(check(&[("TERM_PROGRAM", "ghostty")], true, support).is_ok());
    }
  }

  #[test]
  fn requires_a_tty_even_in_a_supported_terminal() {
    let lookup = |name: &str| (name == "KITTY_WINDOW_ID").then(|| "1".to_string());
    let env = Environment { var: &lookup, is_windows: false, is_atty: false };
    assert_eq!(interactive_support(&env).unwrap_err(), "interactive mode requires a terminal (stdout is not a TTY)");
    assert_eq!(preview_support(&env).unwrap_err(), "image preview requires a terminal (stdout is not a TTY)");
  }

  #[test]
  fn parses_versions() {
    assert_eq!(parse_version("3.5.0beta2"), (3, 5, 0));
    assert_eq!(parse_version("3.10"), (3, 10, 0));
    assert_eq!(parse_version("3.5.2"), (3, 5, 2));
    assert_eq!(parse_version("3.4.19"), (3, 4, 19));
    assert_eq!(parse_version("4"), (4, 0, 0));
    assert_eq!(parse_version(""), (0, 0, 0));
    assert_eq!(parse_version("invalid"), (0, 0, 0));
  }
}
