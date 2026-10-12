import os
import re
import sys
from typing import Mapping


def _parse_version(version_str: str) -> tuple[int, ...]:
  parts = [int(part) for part in re.findall(r"\d+", version_str)[:3]]
  return tuple(parts + [0] * (3 - len(parts)))


def is_interactive_terminal_supported(
  *,
  env: Mapping[str, str] = os.environ,
  platform: str = sys.platform,
  is_atty: bool | None = None,
) -> tuple[bool, str | None]:
  if is_atty is None:
    is_atty = sys.stdout.isatty()
  if not is_atty:
    return False, "interactive mode requires a terminal (stdout is not a TTY)"

  # Tier A: Fully supported terminals with Kitty graphics + Unicode placeholders (U=1)
  if env.get("KITTY_WINDOW_ID") or env.get("KITTY_PID") or env.get("TERM") == "xterm-kitty":
    return True, None
  if env.get("GHOSTTY_RESOURCES_DIR") or env.get("TERM_PROGRAM") == "ghostty" or env.get("TERM") == "xterm-ghostty":
    return True, None
  if env.get("TERM_PROGRAM") == "iTerm.app":
    version = env.get("TERM_PROGRAM_VERSION")
    if not version or _parse_version(version) >= (3, 5, 0):
      return True, None
    return False, (
      "iTerm2 versions prior to 3.5.0 do not support the Kitty Graphics Protocol. "
      "Please update iTerm2. To run anyway, use --force-interactive."
    )

  # Terminals that support baseline Kitty graphics but lack Unicode placeholders (U=1)
  if env.get("TERM_PROGRAM") == "WezTerm" or env.get("WEZTERM_EXECUTABLE") or env.get("WEZTERM_PANE"):
    return False, (
      "WezTerm supports image previews (--preview), but does not support the Kitty Unicode placeholders "
      "required for interactive mode (--interactive). Consider using Ghostty or Kitty. "
      "To run anyway, use --force-interactive."
    )
  if "KONSOLE_VERSION" in env:
    return False, (
      "Konsole supports image previews (--preview), but does not support the Kitty Unicode placeholders "
      "required for interactive mode (--interactive). Consider using Ghostty or Kitty. "
      "To run anyway, use --force-interactive."
    )

  # Known unsupported terminals
  if "WT_SESSION" in env:
    return False, (
      "Windows Terminal does not support the Kitty Graphics Protocol. "
      "Consider using Ghostty or Kitty (via WSL). To run anyway, use --force-interactive."
    )
  if platform == "win32":
    return False, (
      "Windows Command Prompt and PowerShell do not support the Kitty Graphics Protocol. "
      "Consider using Ghostty or Kitty (via WSL). To run anyway, use --force-interactive."
    )
  if env.get("TERM_PROGRAM") == "Apple_Terminal":
    return False, (
      "Apple Terminal (Terminal.app) does not support the Kitty Graphics Protocol. "
      "Consider using Ghostty, Kitty, or iTerm2. To run anyway, use --force-interactive."
    )
  if env.get("TERM") == "alacritty" or env.get("ALACRITTY_LOG") or env.get("ALACRITTY_WINDOW_ID"):
    return False, (
      "Alacritty does not support the Kitty Graphics Protocol. "
      "Consider using Ghostty or Kitty. To run anyway, use --force-interactive."
    )
  if env.get("TERM") in ("dumb", "linux"):
    return False, (
      f"terminal type '{env.get('TERM')}' does not support the Kitty Graphics Protocol. "
      "To run anyway, use --force-interactive."
    )
  if "TMUX" in env:
    return False, (
      "running inside tmux without outer terminal identification. "
      "If your terminal supports Kitty graphics (e.g. Kitty, Ghostty), ensure 'set -g allow-passthrough on' "
      "is set in your .tmux.conf and pass --force-interactive."
    )

  return False, (
    "could not detect a terminal emulator that supports the Kitty Graphics Protocol with Unicode placeholders "
    "(e.g. Kitty, Ghostty, or iTerm2 >= 3.5.0). To run anyway, use --force-interactive."
  )


def is_preview_terminal_supported(
  *,
  env: Mapping[str, str] = os.environ,
  platform: str = sys.platform,
  is_atty: bool | None = None,
) -> tuple[bool, str | None]:
  if is_atty is None:
    is_atty = sys.stdout.isatty()
  if not is_atty:
    return False, "image preview requires a terminal (stdout is not a TTY)"

  # Tier A & B: Baseline Kitty graphics (a=T) supported
  if env.get("KITTY_WINDOW_ID") or env.get("KITTY_PID") or env.get("TERM") == "xterm-kitty":
    return True, None
  if env.get("GHOSTTY_RESOURCES_DIR") or env.get("TERM_PROGRAM") == "ghostty" or env.get("TERM") == "xterm-ghostty":
    return True, None
  if env.get("TERM_PROGRAM") == "iTerm.app":
    version = env.get("TERM_PROGRAM_VERSION")
    if not version or _parse_version(version) >= (3, 5, 0):
      return True, None
    return False, (
      "iTerm2 versions prior to 3.5.0 do not support the Kitty Graphics Protocol. "
      "Please update iTerm2. To preview anyway, use --force-preview."
    )
  if env.get("TERM_PROGRAM") == "WezTerm" or env.get("WEZTERM_EXECUTABLE") or env.get("WEZTERM_PANE"):
    return True, None
  if "KONSOLE_VERSION" in env:
    return True, None

  # Known unsupported terminals
  if "WT_SESSION" in env:
    return False, (
      "Windows Terminal does not support the Kitty Graphics Protocol. "
      "Consider using Ghostty or WezTerm. To preview anyway, use --force-preview."
    )
  if platform == "win32":
    return False, (
      "Windows Command Prompt and PowerShell do not support the Kitty Graphics Protocol. "
      "Consider using Ghostty or WezTerm. To preview anyway, use --force-preview."
    )
  if env.get("TERM_PROGRAM") == "Apple_Terminal":
    return False, (
      "Apple Terminal (Terminal.app) does not support the Kitty Graphics Protocol. "
      "Consider using Ghostty, Kitty, WezTerm, or iTerm2. To preview anyway, use --force-preview."
    )
  if env.get("TERM") == "alacritty" or env.get("ALACRITTY_LOG") or env.get("ALACRITTY_WINDOW_ID"):
    return False, (
      "Alacritty does not support the Kitty Graphics Protocol. "
      "Consider using Ghostty, Kitty, or WezTerm. To preview anyway, use --force-preview."
    )
  if env.get("TERM") in ("dumb", "linux"):
    return False, (
      f"terminal type '{env.get('TERM')}' does not support the Kitty Graphics Protocol. "
      "To preview anyway, use --force-preview."
    )
  if "TMUX" in env:
    return False, (
      "running inside tmux without outer terminal identification. "
      "If your terminal supports Kitty graphics (e.g. Kitty, Ghostty, WezTerm), ensure 'set -g allow-passthrough on' "
      "is set in your .tmux.conf and pass --force-preview."
    )

  return False, (
    "could not detect a terminal emulator that supports the Kitty Graphics Protocol "
    "(e.g. Kitty, Ghostty, WezTerm, Konsole, or iTerm2 >= 3.5.0). To preview anyway, use --force-preview."
  )


def check_interactive_terminal_or_exit(
  *,
  env: Mapping[str, str] = os.environ,
  platform: str = sys.platform,
  is_atty: bool | None = None,
) -> None:
  supported, reason = is_interactive_terminal_supported(env=env, platform=platform, is_atty=is_atty)
  if not supported:
    print(f"rclip: {reason}", file=sys.stderr)
    sys.exit(1)


def check_preview_terminal_or_exit(
  *,
  env: Mapping[str, str] = os.environ,
  platform: str = sys.platform,
  is_atty: bool | None = None,
) -> None:
  supported, reason = is_preview_terminal_supported(env=env, platform=platform, is_atty=is_atty)
  if not supported:
    print(f"rclip: {reason}", file=sys.stderr)
    sys.exit(1)
