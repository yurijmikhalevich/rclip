import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Literal

from PIL import ImageOps

from rclip.utils import helpers
from rclip.utils.preview import iterm_sequence


CLIPBOARD_NATIVE_EXTENSIONS = {"bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp"}
ITERM_TRANSFER_CHUNK_SIZE = 512 * 1024


class ClipboardError(Exception):
  pass


class DownloadError(Exception):
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


def _is_remote_session() -> bool:
  return bool(os.getenv("SSH_CONNECTION") or os.getenv("SSH_TTY"))


def _download_protocol() -> Literal["kitty", "iterm2"]:
  override = os.getenv("RCLIP_DOWNLOAD_PROTOCOL")
  if override == "kitty":
    return "kitty"
  if override == "iterm2":
    return "iterm2"
  if override:
    raise DownloadError("RCLIP_DOWNLOAD_PROTOCOL must be `kitty` or `iterm2`")
  if os.getenv("TERM") == "xterm-kitty" or os.getenv("KITTY_WINDOW_ID") or os.getenv("KITTY_PUBLIC_KEY"):
    return "kitty"
  if os.getenv("TERM_PROGRAM") == "iTerm.app" or os.getenv("LC_TERMINAL") == "iTerm2":
    return "iterm2"
  raise DownloadError("could not detect Kitty or iTerm2; set RCLIP_DOWNLOAD_PROTOCOL to `kitty` or `iterm2`")


def download_image(filepath: str) -> None:
  """Download an original image through a remote terminal session."""
  path = Path(filepath)
  if _download_protocol() == "kitty":
    completed = subprocess.run(
      [_kitten_executable(), "transfer", str(path), "Downloads/"],
      stderr=subprocess.PIPE,
      text=True,
    )
    if completed.returncode:
      message = completed.stderr.strip() or f"kitten exited with status {completed.returncode}"
      raise DownloadError(message)
    return

  name = base64.b64encode(path.name.encode()).decode("ascii")
  sys.stdout.write(iterm_sequence(f"MultipartFile=name={name};size={path.stat().st_size};inline=0"))
  with path.open("rb") as image:
    while chunk := image.read(ITERM_TRANSFER_CHUNK_SIZE):
      payload = base64.b64encode(chunk).decode("ascii")
      sys.stdout.write(iterm_sequence(f"FilePart={payload}"))
  sys.stdout.write(iterm_sequence("FileEnd"))
  sys.stdout.flush()
