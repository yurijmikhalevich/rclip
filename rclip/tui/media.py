from io import BytesIO

from PIL import Image as PILImage
from PIL import ImageOps
from textual.app import RenderResult
from textual.geometry import Size
from textual_image.renderable import TGPImage as TGPRenderable
from textual_image.renderable.tgp import _send_tgp_message
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


class _TGPRenderable(TGPRenderable):
  def cleanup(self) -> None:
    # textual-image 0.12.0 omits the deletion selector and confuses image IDs with numbers.
    if self.terminal_image_id is not None:
      _send_tgp_message(a="d", d="I", i=self.terminal_image_id, q=2)
      self.terminal_image_id = None


class StableTGPImage(TGPImage, Renderable=_TGPRenderable):
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

  def refresh_image(self) -> None:
    self._discard_renderable()
    self.refresh()

  def on_unmount(self) -> None:
    self._discard_renderable()

  def _discard_renderable(self) -> None:
    if self._renderable is not None:
      self._renderable.cleanup()
      self._renderable = None
    self._rendered_size = None


ImageWidget = StableTGPImage
