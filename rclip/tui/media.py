import hashlib
import os
from pathlib import Path
import tempfile

from PIL import Image as PILImage
from PIL import ImageOps
from textual.app import RenderResult
from textual.geometry import Size
from textual_image.renderable import Image as TerminalRenderable
from textual_image.renderable import TGPImage as TGPRenderable
from textual_image.widget import Image as TerminalImage
from textual_image.widget import TGPImage

from rclip.utils import helpers


def _cache_path(filepath: str, cache_dir: Path, size: tuple[int, int]) -> Path:
  source = Path(filepath).resolve()
  key = hashlib.sha256(f"{source}\0{size[0]}x{size[1]}".encode()).hexdigest()
  return cache_dir / f"{key}.jpg"


def cache_image(filepath: str, cache_dir: Path, size: tuple[int, int]) -> Path:
  """Return an orientation-corrected display image no larger than ``size``."""
  source = Path(filepath)
  target = _cache_path(filepath, cache_dir, size)
  source_mtime = source.stat().st_mtime_ns
  if target.is_file() and target.stat().st_mtime_ns == source_mtime:
    return target

  cache_dir.mkdir(parents=True, exist_ok=True)
  temporary = tempfile.NamedTemporaryFile(prefix=f".{target.stem}-", suffix=".jpg", dir=cache_dir, delete=False)
  temporary.close()
  temporary_path = Path(temporary.name)
  try:
    with helpers.read_image(filepath) as opened:
      image = ImageOps.exif_transpose(opened)
      image.thumbnail(size, PILImage.Resampling.LANCZOS)
      if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = PILImage.new("RGBA", rgba.size, "#121212")
        background.alpha_composite(rgba)
        image = background
      image.convert("RGB").save(temporary_path, "JPEG", quality=80)
    os.utime(temporary_path, ns=(source_mtime, source_mtime))
    os.replace(temporary_path, target)
  finally:
    temporary_path.unlink(missing_ok=True)
  return target


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
