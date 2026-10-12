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
    self.out_stream.write(f"{ALT_SCREEN_ON}{CURSOR_HIDE}{CLEAR_SCREEN}{CURSOR_HOME}")
    self.out_stream.flush()

  def exit(self) -> None:
    if not self._is_raw:
      return
    self.out_stream.write(f"{CURSOR_SHOW}{ALT_SCREEN_OFF}")
    self.out_stream.flush()
    if self._old_attr is not None and self.in_stream.isatty():
      try:
        termios.tcsetattr(self.in_stream.fileno(), termios.TCSADRAIN, self._old_attr)
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


def read_key(in_stream: TextIO = sys.stdin, timeout: float = 0.05) -> str | None:
  """Read a single key or ANSI key sequence from raw stdin with timeout."""
  if not in_stream.isatty():
    return None

  try:
    rlist, _, _ = select.select([in_stream], [], [], timeout)
    if not rlist:
      return None

    first = in_stream.read(1)
    if not first:
      return None

    # Handle Ctrl keys
    code = ord(first)
    if code == 3:
      return "ctrl+c"
    if code == 6:
      return "ctrl+f"
    if code == 15:
      return "ctrl+o"
    if code == 16:
      return "ctrl+p"
    if code == 19:
      return "ctrl+s"
    if code == 25:
      return "ctrl+y"
    if code in (10, 13):
      return "enter"
    if code in (8, 127):
      return "backspace"
    if code == 9:
      return "tab"

    # Non-escape regular characters
    if first != ESC:
      return first

    # Read remaining escape sequence with a short timeout
    seq = ""
    while True:
      rlist, _, _ = select.select([in_stream], [], [], 0.02)
      if not rlist:
        break
      char = in_stream.read(1)
      if not char:
        break
      seq += char
      if char.isalpha() or char == "~":
        break

    if not seq:
      return "escape"

    # Arrow keys and standard sequences: \x1b[A, \x1b[B, \x1b[C, \x1b[D
    if seq == "[A" or seq == "OA":
      return "up"
    if seq == "[B" or seq == "OB":
      return "down"
    if seq == "[C" or seq == "OC":
      return "right"
    if seq == "[D" or seq == "OD":
      return "left"
    if seq in ("[H", "[1~"):
      return "home"
    if seq in ("[F", "[4~"):
      return "end"
    if seq == "[5~":
      return "page_up"
    if seq == "[6~":
      return "page_down"
    if seq == "[3~":
      return "delete"

    # Kitty keyboard protocol sequences: \x1b[<unicode_key>;<modifiers>u
    if seq.startswith("[") and seq.endswith("u"):
      parts = seq[1:-1].split(";")
      if parts:
        try:
          key_code = int(parts[0])
          return chr(key_code)
        except ValueError:
          pass

    return "escape"
  except (OSError, select.error):
    return None
