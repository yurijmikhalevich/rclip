import base64
from io import BytesIO
from random import Random

from PIL import Image
import pytest

from rclip.utils import preview as preview_module


@pytest.mark.parametrize("term", ["xterm-kitty", "xterm-256color", "tmux-256color"])
@pytest.mark.parametrize("height", [1, 50, 400])
def test_preview_transmits_resized_png_in_kitty_chunks(monkeypatch, capsys, term, height):
  original = Image.frombytes("RGB", (200, 100), Random(0).randbytes(60000))
  monkeypatch.setattr(preview_module, "read_image", lambda _filepath, **_kw: original.copy())
  monkeypatch.setenv("TERM", term)

  preview_module.preview("cat.jpg", height)

  output = capsys.readouterr().out
  assert output.endswith("\n")
  if term.startswith("tmux"):
    output = output.replace("\033Ptmux;", "").replace("\033\033", "\033").replace("\033\\\033\\", "\033\\")
  commands = output.removesuffix("\n").split("\033\\")
  assert commands.pop() == ""
  payload = ""
  for index, command in enumerate(commands):
    header, chunk = command.split(";", 1)
    more = int(index < len(commands) - 1)
    expected = f"\033_Ga=T,f=100,q=2,m={more}" if index == 0 else f"\033_Gq=2,m={more}"
    assert header == expected
    assert 0 < len(chunk) <= 4096
    payload += chunk
  image = Image.open(BytesIO(base64.b64decode(payload)))
  image.load()
  assert image.format == "PNG"
  assert image.size == (2 * min(height, 100), min(height, 100))
  if height == 400:
    assert len(commands) > 1
