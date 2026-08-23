from io import BytesIO

from PIL import Image as PILImage
from PIL import ImageOps
from textual.app import RenderResult
from textual.geometry import Size
from textual_image.renderable import Image as TerminalRenderable
from textual_image.renderable import TGPImage as TGPRenderable
from textual_image.widget import Image as TerminalImage
from textual_image.widget import TGPImage

from rclip.utils import helpers


def prepare_image(filepath: str, size: tuple[int, int]) -> BytesIO:
  """Return an orientation-corrected JPEG no larger than ``size``."""
  with helpers.read_image(filepath) as opened:
    image = ImageOps.exif_transpose(opened)
    image.thumbnail(size, PILImage.Resampling.LANCZOS)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
      rgba = image.convert("RGBA")
      background = PILImage.new("RGBA", rgba.size, "#121212")
      background.alpha_composite(rgba)
      image = background
    prepared = BytesIO()
    image.convert("RGB").save(prepared, "JPEG", quality=80)
  prepared.seek(0)
  return prepared


class StableTGPImage(TGPImage, Renderable=TGPRenderable):
  """Keep a Kitty image alive until its source or rendered size changes."""

  _rendered_size: Size | None = None

  def render(self) -> RenderResult:
    if not self.image:
      return ""
    if self._rendered_size != self.content_size:
      self._discard_renderable()
    if self._renderable is None:
      self._renderable = self._Renderable(self.image, *self._get_styled_size())
    self._rendered_size = self.content_size
    return self._renderable

  def on_unmount(self) -> None:
    self._discard_renderable()

  def _discard_renderable(self) -> None:
    if self._renderable is not None:
      self._renderable.cleanup()
      self._renderable = None
    self._rendered_size = None


ImageWidget = StableTGPImage if TerminalRenderable is TGPRenderable else TerminalImage
