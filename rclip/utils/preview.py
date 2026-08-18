import base64
from io import BytesIO
import os
import sys
from PIL import Image

from rclip.utils.helpers import read_image


def _get_start_sequence():
  term_env_var = os.getenv("TERM")
  if term_env_var and (term_env_var.startswith("screen") or term_env_var.startswith("tmux")):
    return "\033Ptmux;\033\033]"
  return "\033]"


def _get_end_sequence():
  term_env_var = os.getenv("TERM")
  if term_env_var and (term_env_var.startswith("screen") or term_env_var.startswith("tmux")):
    return "\a\033\\"
  return "\a"


def iterm_sequence(command: str) -> str:
  return f"{_get_start_sequence()}1337;{command}{_get_end_sequence()}"


def _get_preview_dimensions(width_px: int, height_px: int) -> tuple[str, str]:
  if sys.stdout.isatty() and os.name != "nt":
    try:
      import fcntl
      import struct
      import termios

      rows, cols, terminal_width_px, terminal_height_px = struct.unpack(
        "HHHH",
        fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0)),
      )
      if rows > 0 and cols > 0 and terminal_width_px > 0 and terminal_height_px > 0:
        cell_width_px = terminal_width_px / cols
        cell_height_px = terminal_height_px / rows
        return (
          str(max(1, round(width_px / cell_width_px))),
          str(max(1, round(height_px / cell_height_px))),
        )
    except OSError:
      pass

  return f"{width_px}px", f"{height_px}px"


def preview(filepath: str, img_height_px: int):
  # preview images are displayed one at a time and the user opted into viewing them, so the
  # indexing memory cap doesn't apply; read as trusted like query images.
  with read_image(filepath, trusted=True) as img:
    if img_height_px >= img.height:
      width_px, height_px = img.width, img.height
    else:
      width_px, height_px = int(img_height_px * img.width / img.height), img_height_px
    img = img.resize((width_px, height_px), Image.LANCZOS)  # type: ignore
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="JPEG")
  img_bytes = buffer.getvalue()
  img_str = base64.b64encode(img_bytes).decode("utf-8")
  width, height = _get_preview_dimensions(width_px, height_px)
  print(
    iterm_sequence(
      f"File=inline=1;size={len(img_bytes)};preserveAspectRatio=1;width={width};height={height}:{img_str}"
    ),
  )
