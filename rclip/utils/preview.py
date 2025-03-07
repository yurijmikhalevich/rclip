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


def preview(filepath: str, img_height_px: int):
  with read_image(filepath) as img:
    if img_height_px >= img.height:
      width_px, height_px = img.width, img.height
    else:
      width_px, height_px = int(img_height_px * img.width / img.height), img_height_px
    img = img.resize((width_px, height_px), Image.LANCZOS)  # type: ignore
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="JPEG")
  img_bytes = buffer.getvalue()
  img_str = base64.b64encode(img_bytes).decode("utf-8")
  print(
    f"{_get_start_sequence()}1337;"
    f"File=inline=1;size={len(img_bytes)};preserveAspectRatio=1;"
    f"width={width_px}px;height={height_px}px:{img_str}{_get_end_sequence()}",
  )
