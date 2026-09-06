import base64
from io import BytesIO
import os
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
    img.convert("RGB").save(buffer, format="PNG")
  img_str = base64.b64encode(buffer.getvalue()).decode("ascii")
  for offset in range(0, len(img_str), 4096):
    chunk = img_str[offset : offset + 4096]
    more = int(offset + 4096 < len(img_str))
    command = f"a=T,f=100,q=2,m={more}" if offset == 0 else f"q=2,m={more}"
    sequence = f"\033_G{command};{chunk}\033\\"
    if os.getenv("TERM", "").startswith(("screen", "tmux")):
      sequence = "\033Ptmux;" + sequence.replace("\033", "\033\033") + "\033\\"
    print(sequence, end="")
  print()
