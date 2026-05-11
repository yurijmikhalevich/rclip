import base64
from io import BytesIO

from PIL import Image

from rclip.utils import preview as preview_module


def test_preview_uses_terminal_cell_dimensions_when_available(monkeypatch, capsys):
  monkeypatch.setattr(preview_module, "read_image", lambda _filepath: Image.new("RGB", (200, 100), "red"))
  monkeypatch.setattr(preview_module, "_get_preview_dimensions", lambda _width, _height: ("10", "5"))

  preview_module.preview("cat.jpg", 50)

  output = capsys.readouterr().out.rstrip("\n")
  assert ";width=10;height=5:" in output

  metadata, encoded_image = output.split(":", 1)
  assert metadata.startswith("\033]1337;File=inline=1;size=")

  image = Image.open(BytesIO(base64.b64decode(encoded_image.removesuffix("\a"))))
  assert image.size == (100, 50)


def test_get_preview_dimensions_falls_back_to_pixels_when_tty_geometry_is_unavailable(monkeypatch):
  monkeypatch.setattr(preview_module.sys.stdout, "isatty", lambda: False)

  assert preview_module._get_preview_dimensions(120, 60) == ("120px", "60px")
