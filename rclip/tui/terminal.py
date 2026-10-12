from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
import select
import struct
import sys
import termios
import tty
from typing import Any, TextIO


@dataclass(frozen=True)
class CellSize:
  width: int
  height: int


def get_cell_pixel_size(stream: TextIO = sys.stdout) -> CellSize:
  """Determine terminal cell pixel dimensions (width, height) via ioctl or env/default."""
  tgp_mod = sys.modules.get("textual_image.renderable.tgp")
  if tgp_mod and hasattr(tgp_mod, "get_cell_size"):
    try:
      cs = tgp_mod.get_cell_size()
      return CellSize(cs.width, cs.height)
    except Exception:
      pass

  try:
    if stream.isatty():
      rows, cols, xpixels, ypixels = struct.unpack(
        "HHHH", fcntl.ioctl(stream.fileno(), termios.TIOCGWINSZ, b"\x00" * 8)
      )
      if xpixels > 0 and cols > 0 and ypixels > 0 and rows > 0:
        return CellSize(max(1, xpixels // cols), max(1, ypixels // rows))
  except Exception:
    pass

  for w_var, h_var in [("CELL_WIDTH", "CELL_HEIGHT"), ("TEXTUAL_CELL_WIDTH", "TEXTUAL_CELL_HEIGHT")]:
    w, h = os.getenv(w_var), os.getenv(h_var)
    if w and h and w.isdigit() and h.isdigit():
      return CellSize(max(1, int(w)), max(1, int(h)))

  return CellSize(10, 20)


def get_terminal_size() -> tuple[int, int]:
  """Return (columns, rows) of current terminal window."""
  try:
    size = os.get_terminal_size()
    return max(10, size.columns), max(5, size.lines)
  except OSError:
    return 80, 24


# ANSI Escape Sequences
ESC = "\x1b"
ALT_SCREEN_ON = f"{ESC}[?1049h"
ALT_SCREEN_OFF = f"{ESC}[?1049l"
CURSOR_HIDE = f"{ESC}[?25l"
CURSOR_SHOW = f"{ESC}[?25h"
CLEAR_SCREEN = f"{ESC}[2J"
CURSOR_HOME = f"{ESC}[H"
CLEAR_LINE = f"{ESC}[2K"
AUTOWRAP_OFF = f"{ESC}[?7l"
AUTOWRAP_ON = f"{ESC}[?7h"

# Kitty Keyboard Protocol progressive enhancement (if supported)
KITTY_KEYBOARD_ON = f"{ESC}[>1u"
KITTY_KEYBOARD_OFF = f"{ESC}[<u"

# Colors & Text Styling
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
DIM = f"{ESC}[2m"
ITALIC = f"{ESC}[3m"
UNDERLINE = f"{ESC}[4m"

# Basic ANSI colors
FG_BLACK = f"{ESC}[30m"
FG_RED = f"{ESC}[31m"
FG_GREEN = f"{ESC}[32m"
FG_YELLOW = f"{ESC}[33m"
FG_BLUE = f"{ESC}[34m"
FG_MAGENTA = f"{ESC}[35m"
FG_CYAN = f"{ESC}[36m"
FG_WHITE = f"{ESC}[37m"
FG_BRIGHT_BLACK = f"{ESC}[90m"
FG_DEFAULT = f"{ESC}[39m"

BG_BLACK = f"{ESC}[40m"
BG_BRIGHT_BLACK = f"{ESC}[100m"
BG_DEFAULT = f"{ESC}[49m"


def cursor_to(row: int, col: int) -> str:
  """1-indexed ANSI cursor positioning."""
  return f"{ESC}[{max(1, row)};{max(1, col)}H"


def fg_rgb(r: int, g: int, b: int) -> str:
  return f"{ESC}[38;2;{r};{g};{b}m"


def bg_rgb(r: int, g: int, b: int) -> str:
  return f"{ESC}[48;2;{r};{g};{b}m"


class RawTerminal:
  """Context manager for entering and exiting raw terminal mode and alternate screen."""

  def __init__(self, in_stream: TextIO = sys.stdin, out_stream: TextIO = sys.stdout) -> None:
    self.in_stream = in_stream
    self.out_stream = out_stream
    self._old_attr: Any = None
    self._is_raw = False

  def enter(self) -> None:
    if self._is_raw:
      return
    if self.in_stream.isatty():
      try:
        self._old_attr = termios.tcgetattr(self.in_stream.fileno())
        tty.setraw(self.in_stream.fileno())
      except termios.error:
        self._old_attr = None
    self._is_raw = True
    self.out_stream.write(f"{ALT_SCREEN_ON}{CURSOR_HIDE}{AUTOWRAP_OFF}{KITTY_KEYBOARD_ON}{CLEAR_SCREEN}{CURSOR_HOME}")
    self.out_stream.flush()

  def exit(self) -> None:
    if not self._is_raw:
      return
    self.out_stream.write(f"{KITTY_KEYBOARD_OFF}{CURSOR_SHOW}{AUTOWRAP_ON}{ALT_SCREEN_OFF}")
    self.out_stream.flush()
    if self._old_attr is not None and self.in_stream.isatty():
      try:
        termios.tcsetattr(self.in_stream.fileno(), termios.TCSANOW, self._old_attr)
      except termios.error:
        pass
      self._old_attr = None
    self._is_raw = False

  def write(self, s: str) -> None:
    self.out_stream.write(s)

  def flush(self) -> None:
    self.out_stream.flush()

  def __enter__(self) -> RawTerminal:
    self.enter()
    return self

  def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
    self.exit()


_KITTY_KEY_MAP: dict[int, str] = {
  13: "enter",
  27: "escape",
  9: "tab",
  127: "backspace",
  8: "backspace",
  57352: "up",
  57353: "down",
  57354: "left",
  57355: "right",
  57356: "page_up",
  57357: "page_down",
  57358: "home",
  57359: "end",
  57360: "insert",
  57361: "delete",
  57399: "enter",
  57400: "escape",
}

_CTRL_MAP: dict[int, str] = {
  1: "ctrl+a",
  2: "ctrl+b",
  3: "ctrl+c",
  4: "ctrl+d",
  5: "ctrl+e",
  6: "ctrl+f",
  11: "ctrl+k",
  12: "ctrl+l",
  14: "ctrl+n",
  15: "ctrl+o",
  16: "ctrl+p",
  18: "ctrl+r",
  19: "ctrl+s",
  21: "ctrl+u",
  23: "ctrl+w",
  25: "ctrl+y",
  26: "ctrl+z",
}


def _parse_kitty_u(seq: str) -> str | None:
  """Parse Kitty keyboard protocol sequence: \\x1b[<key>;<mods>[:<event>]u."""
  body = seq[1:-1]
  parts = body.split(";")
  if not parts:
    return None

  key_sub = parts[0].split(":")
  if not key_sub[0].isdigit():
    return None
  key_code = int(key_sub[0])
  base_key = int(key_sub[2]) if len(key_sub) >= 3 and key_sub[2].isdigit() else key_code

  mods = 0
  event_type = 1
  if len(parts) > 1 and parts[1]:
    mod_sub = parts[1].split(":")
    if mod_sub[0].isdigit():
      mods = max(0, int(mod_sub[0]) - 1)
    if len(mod_sub) > 1 and mod_sub[1].isdigit():
      event_type = int(mod_sub[1])

  # event_type: 1 = press, 2 = repeat, 3 = release
  if event_type == 3:
    return None

  is_shift = bool(mods & 1)
  is_ctrl = bool(mods & 4)

  # Check text codepoint if provided (3rd parameter in Kitty keyboard protocol)
  if len(parts) > 2 and parts[2] and parts[2].isdigit():
    text_code = int(parts[2])
    if not is_ctrl and text_code >= 32:
      return chr(text_code)

  if is_ctrl:
    if key_code in _KITTY_KEY_MAP:
      return f"ctrl+{_KITTY_KEY_MAP[key_code]}"
    if key_code in _CTRL_MAP:
      return _CTRL_MAP[key_code]
    if 32 <= key_code <= 126:
      return f"ctrl+{chr(key_code).lower()}"
    if base_key in _CTRL_MAP:
      return _CTRL_MAP[base_key]
    if 32 <= base_key <= 126:
      return f"ctrl+{chr(base_key).lower()}"
    return None

  if key_code in _CTRL_MAP:
    return _CTRL_MAP[key_code]

  if key_code in _KITTY_KEY_MAP:
    base = _KITTY_KEY_MAP[key_code]
    if is_shift and key_code == 9:
      return "shift+tab"
    if is_shift and base in ("up", "down", "left", "right", "home", "end"):
      return f"shift+{base}"
    return base

  if is_shift and ord("a") <= key_code <= ord("z"):
    return chr(key_code).upper()

  try:
    return chr(key_code)
  except ValueError:
    return None


def _parse_escape_seq(seq: str) -> str | None:
  """Parse any ANSI CSI, SS3, or Kitty keyboard protocol sequence."""
  # SS3 sequences: OA (up), OB (down), OC (right), OD (left), OH (home), OF (end)
  if seq.startswith("O"):
    ss3_map = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end"}
    return ss3_map.get(seq[1:])

  if not seq.startswith("["):
    return None

  # Kitty keyboard protocol: \x1b[...u
  if seq.endswith("u"):
    return _parse_kitty_u(seq)

  # CSI sequences ending with A, B, C, D, H, F (arrows and home/end)
  final = seq[-1]
  if final in ("A", "B", "C", "D", "H", "F"):
    dir_map = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end"}
    base = dir_map[final]
    param_str = seq[1:-1]
    if not param_str:
      return base

    parts = param_str.split(";")
    mods = 0
    event_type = 1
    if len(parts) > 1 and parts[1]:
      mod_sub = parts[1].split(":")
      if mod_sub[0].isdigit():
        mods = max(0, int(mod_sub[0]) - 1)
      if len(mod_sub) > 1 and mod_sub[1].isdigit():
        event_type = int(mod_sub[1])
    elif len(parts) == 1 and ":" in parts[0]:
      sub = parts[0].split(":")
      if len(sub) > 1 and sub[1].isdigit():
        event_type = int(sub[1])

    # event_type: 3 = release
    if event_type == 3:
      return None

    is_ctrl = bool(mods & 4)
    is_shift = bool(mods & 1)
    if is_ctrl:
      return f"ctrl+{base}"
    if is_shift:
      return f"shift+{base}"
    return base

  # CSI sequences ending with ~ (VT keys: delete, page_up, page_down, etc.)
  if seq.endswith("~"):
    body = seq[1:-1]
    parts = body.split(";")
    vt_code = parts[0]
    mods = 0
    event_type = 1
    if len(parts) > 1 and parts[1]:
      mod_sub = parts[1].split(":")
      if mod_sub[0].isdigit():
        mods = max(0, int(mod_sub[0]) - 1)
      if len(mod_sub) > 1 and mod_sub[1].isdigit():
        event_type = int(mod_sub[1])
    elif ":" in vt_code:
      sub = vt_code.split(":")
      vt_code = sub[0]
      if len(sub) > 1 and sub[1].isdigit():
        event_type = int(sub[1])

    if event_type == 3:
      return None

    is_ctrl = bool(mods & 4)
    vt_map = {
      "1": "home",
      "2": "insert",
      "3": "delete",
      "4": "end",
      "5": "page_up",
      "6": "page_down",
      "7": "home",
      "8": "end",
    }
    base = vt_map.get(vt_code)
    if base and is_ctrl:
      return f"ctrl+{base}"
    return base

  if seq == "[Z":
    return "shift+tab"

  return None


def read_key(in_stream: TextIO = sys.stdin, timeout: float = 0.05) -> str | None:
  """Read a single key or ANSI key sequence from raw stdin with timeout."""
  if not in_stream.isatty():
    return None

  try:
    fd = in_stream.fileno()
    while True:
      rlist, _, _ = select.select([fd], [], [], timeout)
      if not rlist:
        return None

      raw = os.read(fd, 1)
      if not raw:
        return None

      code = raw[0]
      if code in _CTRL_MAP:
        return _CTRL_MAP[code]
      if code in (10, 13):
        return "enter"
      if code in (8, 127):
        return "backspace"
      if code == 9:
        return "tab"

      if code != 27:  # Not ESC
        needed = 0
        if (code & 0xE0) == 0xC0:
          needed = 1
        elif (code & 0xF0) == 0xE0:
          needed = 2
        elif (code & 0xF8) == 0xF0:
          needed = 3

        if needed == 0:
          return raw.decode("latin1")

        buf = bytearray(raw)
        while needed > 0:
          r, _, _ = select.select([fd], [], [], 0.02)
          if not r:
            break
          nxt = os.read(fd, 1)
          if not nxt:
            break
          buf.extend(nxt)
          needed -= 1
        return buf.decode("utf-8", errors="ignore")

      # Read remaining escape sequence with a short timeout directly from fd
      seq = ""
      while True:
        rlist, _, _ = select.select([fd], [], [], 0.02)
        if not rlist:
          break
        nxt = os.read(fd, 1)
        if not nxt:
          break
        char = chr(nxt[0])
        seq += char
        if seq == "O":
          # SS3 prefix: need to read the next character (e.g. 'A')
          continue
        if char.isalpha() or char in ("~", "@"):
          break

      if not seq:
        return "escape"

      parsed = _parse_escape_seq(seq)
      if parsed is not None:
        return parsed
      # Discard release events (e.g. event_type: 3) and check next queued event
      if ":3" in seq or ";3" in seq:
        timeout = 0.005
        continue

      return "escape"
  except (OSError, select.error):
    return None
