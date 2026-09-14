from functools import cache
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import ImageOps

from rclip.utils import helpers


CLIPBOARD_NATIVE_EXTENSIONS = {"bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp"}


class TransferError(Exception):
  pass


def _kitten_executable() -> str:
  if executable := shutil.which("kitten"):
    return executable
  if installation_dir := os.getenv("KITTY_INSTALLATION_DIR"):
    executable = Path(installation_dir) / "kitten"
    if executable.is_file():
      return str(executable)
  raise TransferError("could not find Kitty's `kitten` executable")


def _run_kitten(command: list[str], timeout: float | None = None) -> str:
  with subprocess.Popen(
    command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
  ) as process:
    try:
      output, error = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
      process.terminate()
      try:
        process.communicate(timeout=3)
      except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
      raise TransferError("Kitty did not respond in time") from error
    if process.returncode:
      raise TransferError(error.strip() or f"kitten exited with status {process.returncode}")
    return output


def _supports_query_terminal(executable: str) -> bool:
  """Check whether this `kitten` has `query_terminal`, which kitty gained in 0.38."""
  try:
    _run_kitten([executable, "query_terminal", "--help"])
  except TransferError:
    return False
  return True


@cache
def _probe_kitty(executable: str) -> None:
  if not _supports_query_terminal(executable):
    return
  response = _run_kitten([executable, "query_terminal", "--wait-for", "1", "name"], timeout=2)
  if response.strip() != "name: kitty":
    raise TransferError("Image copy and download require Kitty")


def _require_kitty() -> str:
  executable = _kitten_executable()
  _probe_kitty(executable)
  return executable


def _run_clipboard_kitten(filepath: Path) -> None:
  _run_kitten([_require_kitty(), "clipboard", str(filepath)], timeout=30)


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


def _is_remote_session() -> bool:
  return bool(os.getenv("SSH_CONNECTION") or os.getenv("SSH_TTY"))


def download_image(filepath: str) -> None:
  """Download an original image through a remote Kitty terminal session."""
  _run_kitten([_require_kitty(), "transfer", str(Path(filepath)), "Downloads/"])
