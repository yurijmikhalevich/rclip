import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import ImageOps

from rclip.utils import helpers


CLIPBOARD_NATIVE_EXTENSIONS = {"bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp"}


class ClipboardError(Exception):
  pass


def _kitten_executable() -> str:
  if executable := shutil.which("kitten"):
    return executable
  if installation_dir := os.getenv("KITTY_INSTALLATION_DIR"):
    executable = Path(installation_dir) / "kitten"
    if executable.is_file():
      return str(executable)
  raise ClipboardError("could not find Kitty's `kitten` executable")


def _run_clipboard_kitten(filepath: Path) -> None:
  completed = subprocess.run(
    [_kitten_executable(), "clipboard", str(filepath)],
    stdin=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
  )
  if completed.returncode:
    message = completed.stderr.strip() or f"kitten exited with status {completed.returncode}"
    raise ClipboardError(message)


def copy_image_to_clipboard(filepath: str) -> None:
  """Copy an image to Kitty's clipboard, converting uncommon formats to PNG."""
  if helpers.get_file_extension(filepath) in CLIPBOARD_NATIVE_EXTENSIONS:
    _run_clipboard_kitten(Path(filepath))
    return

  with tempfile.TemporaryDirectory(prefix="rclip-clipboard-") as temporary:
    converted = Path(temporary) / "image.png"
    with helpers.read_image(filepath) as opened:
      ImageOps.exif_transpose(opened).save(converted, "PNG")
    _run_clipboard_kitten(converted)
