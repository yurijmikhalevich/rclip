"""End-to-end tests that copy an image through a real Kitty terminal.

Run them inside Kitty, for example:

    xvfb-run -a kitty --config NONE uv run pytest tests/e2e/test_kitty_transfer.py

Without a Kitty terminal (and a `kitten` executable) the tests are skipped, so
`make test` stays runnable anywhere.
"""

from __future__ import annotations

import io
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from rclip.tui.transfer import _supports_query_terminal
from rclip.tui.transfer import copy_image_to_clipboard


KITTEN = shutil.which("kitten")
PROBES_TERMINAL = _supports_query_terminal(KITTEN) if KITTEN else False

pytestmark = pytest.mark.skipif(
  not (os.getenv("KITTY_WINDOW_ID") and KITTEN),
  reason="requires a Kitty terminal with a `kitten` executable",
)


def _clipboard_png() -> bytes:
  if xclip := shutil.which("xclip"):
    return subprocess.run(
      [xclip, "-selection", "clipboard", "-t", "image/png", "-o"], check=True, capture_output=True
    ).stdout
  if sys.platform == "darwin":
    data = subprocess.run(
      ["osascript", "-e", "the clipboard as «class PNGf»"], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(r"«data PNGf([0-9A-F]+)»", data)
    assert match, data
    return bytes.fromhex(match.group(1))
  pytest.skip("reading the clipboard requires xclip on Linux or osascript on macOS")


def make_image(path: Path) -> Image.Image:
  image = Image.new("RGB", (32, 32), (255, 0, 0))
  image.save(path)
  return image


def test_copy_image_reaches_the_system_clipboard(tmp_path: Path) -> None:
  source = tmp_path / "source.png"
  image = make_image(source)

  copy_image_to_clipboard(str(source))

  copied = Image.open(io.BytesIO(_clipboard_png())).convert("RGB")
  assert copied.size == image.size
  assert copied.tobytes() == image.tobytes()


@pytest.mark.skipif(sys.platform != "linux", reason="requires the util-linux `script` command")
@pytest.mark.skipif(not PROBES_TERMINAL, reason="a `kitten` older than 0.38 cannot probe the terminal")
def test_copy_fails_fast_in_a_terminal_that_is_not_kitty(tmp_path: Path) -> None:
  source = tmp_path / "source.png"
  make_image(source)
  code = f"from rclip.tui.transfer import copy_image_to_clipboard; copy_image_to_clipboard({str(source)!r})"

  # A plain pty answers neither Kitty's query protocol nor its clipboard requests.
  completed = subprocess.run(
    ["script", "-qec", f"stty cols 80 rows 24; {sys.executable} -c {shlex.quote(code)}", "/dev/null"],
    capture_output=True,
    text=True,
    timeout=60,
  )

  assert completed.returncode
  assert "TransferError" in completed.stdout + completed.stderr
