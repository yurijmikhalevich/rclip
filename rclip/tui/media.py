from base64 import b64encode
from collections import OrderedDict
from io import BytesIO
from itertools import count
from pathlib import Path
from typing import IO, Callable

from PIL import Image as PILImage
from PIL import ImageOps
from textual.app import RenderResult
from textual.geometry import Size
from textual_image._pixeldata import PixelData
from textual_image._utils import StrOrBytesPath
from textual_image.renderable import TGPImage as TGPRenderable
from textual_image.renderable.tgp import _send_tgp_message
from textual_image.widget import TGPImage

from rclip.utils import helpers


# Kitty keeps transmitted image data until it is deleted or its per-buffer quota silently evicts the
# oldest images, so keep our own cache well below that quota. The quota counts decoded pixels, which
# are far larger than the PNG payload sent over the wire.
_TGP_DECODED_BUDGET = 64 * 1024 * 1024

# textual-image hardcodes `compress_level=2` when encoding the PNG it sends to the terminal. Level 6
# saves ~10% of the wire bytes for ~20ms more encode time per image; level 9 saves only ~1% more for
# ~4x the encode time, which would stall the main thread while it renders.
_PNG_COMPRESS_LEVEL = 6


class _PreparedImage(BytesIO):
  """Prepared image data that remembers the file it was prepared from."""

  def __init__(self, source_path: str) -> None:
    super().__init__()
    self.source_path = source_path


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
    prepared = _PreparedImage(filepath)
    image.convert("RGB").save(prepared, "JPEG", quality=80)
  prepared.seek(0)
  return prepared


def _extract_source(image: object) -> str | int:
  if isinstance(image, (str, Path)):
    return str(image)
  source_path = getattr(image, "source_path", None)
  if isinstance(source_path, (str, Path)):
    return str(source_path)
  return id(image)


class _TransmittedImage:
  """One copy of an image that lives in the terminal, reusable by later placements."""

  def __init__(self, key: object, source: object | None, kitty_id: int, decoded: int) -> None:
    self.key = key
    # Identity-based sources are kept alive so their id() stays valid for the lifetime of the image.
    self.source = source
    self.kitty_id = kitty_id
    self.decoded = decoded
    self.refs = 0


class _ImageCache:
  """Keep transmitted Kitty image data alive for later placements.

  Kitty keeps an image's data until it is deleted or the per-buffer quota silently evicts the oldest
  image. Deleting only the placement (`d=i`) leaves the data in place, so showing the same image
  again needs a placement but no transmission. Images are keyed by renderable type, source,
  prepared size, and cell box, so previews, detail views, and differently-padded or sized widgets
  never share or distort each other's data.
  """

  def __init__(self, budget: int = _TGP_DECODED_BUDGET) -> None:
    self._by_key: dict[object, _TransmittedImage] = {}
    self._lru: OrderedDict[_TransmittedImage, None] = OrderedDict()
    self._decoded = 0
    self._budget = budget

  def acquire(
    self,
    key: object,
    source: object | None,
    decoded: int,
    transmit: Callable[[int], None],
  ) -> _TransmittedImage:
    image = self._by_key.get(key)
    if image is not None:
      return self._use(image)
    image = _TransmittedImage(key, source, next(TGPRenderable._image_id_counter), decoded)
    transmit(image.kitty_id)
    self._by_key[key] = image
    self._decoded += image.decoded
    self._use(image)
    self._evict()
    return image

  def release(self, image: _TransmittedImage) -> None:
    image.refs -= 1
    self._evict()

  def _use(self, image: _TransmittedImage) -> _TransmittedImage:
    image.refs += 1
    self._lru[image] = None
    self._lru.move_to_end(image)
    return image

  def _evict(self) -> None:
    for image in list(self._lru):
      if self._decoded <= self._budget:
        return
      if image.refs:
        continue
      del self._lru[image]
      self._decoded -= image.decoded
      del self._by_key[image.key]
      _send_tgp_message(a="d", d="I", i=image.kitty_id, q=2)


_IMAGE_CACHE = _ImageCache()


def _transmit_image(kitty_id: int, image_data: PixelData, size: tuple[int, int]) -> None:
  """Send image data to the terminal as a base64 PNG, framed the way textual-image frames it."""
  buffer = BytesIO()
  image_data.scaled(*size).pil_image.save(buffer, format="png", compress_level=_PNG_COMPRESS_LEVEL)
  payload = b64encode(buffer.getvalue()).decode("ascii")
  while payload:
    chunk, payload = payload[:4096], payload[4096:]
    _send_tgp_message(i=kitty_id, m=1 if payload else 0, f=100, payload=chunk, q=2)


class _TGPRenderable(TGPRenderable):
  """Render an image using Kitty image data shared through ``_IMAGE_CACHE``."""

  _placement_ids = count(1)

  def __init__(
    self,
    image: StrOrBytesPath | IO[bytes] | PILImage.Image,
    width: int | str | None = None,
    height: int | str | None = None,
  ) -> None:
    super().__init__(image, width, height)
    self._source_id = _extract_source(image)
    self._source_obj = image if not isinstance(self._source_id, str) else None
    self._size = (0, 0)
    self._transmitted: _TransmittedImage | None = None
    self._placement_id = next(_TGPRenderable._placement_ids)

  def _send_image_to_terminal(self, width: int, height: int) -> None:
    # Deferred to `_create_virtual_placement`, which also knows the placement size in cells.
    self._size = (width, height)

  def _create_virtual_placement(self, width: int, height: int) -> None:
    box = (width, height)
    key = (type(self), self._source_id, self._image_data.width, self._image_data.height, box)
    decoded = self._size[0] * self._size[1] * 4
    self._transmitted = _IMAGE_CACHE.acquire(key, self._source_obj, decoded, self._transmit)
    self.terminal_image_id = self._transmitted.kitty_id
    _send_tgp_message(a="p", i=self.terminal_image_id, p=self._placement_id, c=width, r=height, U=1, q=2)

  def _transmit(self, kitty_id: int) -> None:
    _transmit_image(kitty_id, self._image_data, self._size)

  def cleanup(self) -> None:
    if self.terminal_image_id is not None:
      _send_tgp_message(a="d", d="i", i=self.terminal_image_id, p=self._placement_id, q=2)
      self.terminal_image_id = None
    if self._transmitted is not None:
      _IMAGE_CACHE.release(self._transmitted)
      self._transmitted = None


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


class _CenteredTGPRenderable(_TGPRenderable):
  def _transmit(self, kitty_id: int) -> None:
    self._image_data = PixelData(
      ImageOps.pad(self._image_data.pil_image.convert("RGBA"), self._size, color=(0, 0, 0, 0))
    )
    super()._transmit(kitty_id)


class CenteredTGPImage(StableTGPImage, Renderable=_CenteredTGPRenderable):
  """Center an aspect-preserving image in its full widget area, at pixel precision."""
