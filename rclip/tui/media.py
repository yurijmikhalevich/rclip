from __future__ import annotations

from base64 import b64encode
from collections import OrderedDict
from io import BytesIO
from itertools import count
import os
from pathlib import Path
import sys
from threading import Event, Lock
from typing import Any, Callable, Iterator

from PIL import Image as PILImage
from PIL import ImageOps

from rclip.tui.terminal import CellSize, get_cell_pixel_size
from rclip.utils import helpers


_TGP_DECODED_BUDGET = 64 * 1024 * 1024
_PNG_COMPRESS_LEVEL = 6
_TERMINAL_WRITE_LOCK = Lock()

_PLACEHOLDER = 0x10EEEE
_IMAGE_ID_COUNTER = count(1)
_PLACEMENT_ID_COUNTER = count(1)

# fmt: off
_NUMBER_TO_DIACRITIC = [
    0x00305, 0x0030d, 0x0030e, 0x00310, 0x00312, 0x0033d, 0x0033e, 0x0033f, 0x00346, 0x0034a, 0x0034b, 0x0034c,
    0x00350, 0x00351, 0x00352, 0x00357, 0x0035b, 0x00363, 0x00364, 0x00365, 0x00366, 0x00367, 0x00368, 0x00369,
    0x0036a, 0x0036b, 0x0036c, 0x0036d, 0x0036e, 0x0036f, 0x00483, 0x00484, 0x00485, 0x00486, 0x00487, 0x00592,
    0x00593, 0x00594, 0x00595, 0x00597, 0x00598, 0x00599, 0x0059c, 0x0059d, 0x0059e, 0x0059f, 0x005a0, 0x005a1,
    0x005a8, 0x005a9, 0x005ab, 0x005ac, 0x005af, 0x005c4, 0x00610, 0x00611, 0x00612, 0x00613, 0x00614, 0x00615,
    0x00616, 0x00617, 0x00657, 0x00658, 0x00659, 0x0065a, 0x0065b, 0x0065d, 0x0065e, 0x006d6, 0x006d7, 0x006d8,
    0x006d9, 0x006da, 0x006db, 0x006dc, 0x006df, 0x006e0, 0x006e1, 0x006e2, 0x006e4, 0x006e7, 0x006e8, 0x006eb,
    0x006ec, 0x00730, 0x00732, 0x00733, 0x00735, 0x00736, 0x0073a, 0x0073d, 0x0073f, 0x00740, 0x00741, 0x00743,
    0x00745, 0x00747, 0x00749, 0x0074a, 0x007eb, 0x007ec, 0x007ed, 0x007ee, 0x007ef, 0x007f0, 0x007f1, 0x007f3,
    0x00816, 0x00817, 0x00818, 0x00819, 0x0081b, 0x0081c, 0x0081d, 0x0081e, 0x0081f, 0x00820, 0x00821, 0x00822,
    0x00823, 0x00825, 0x00826, 0x00827, 0x00829, 0x0082a, 0x0082b, 0x0082c, 0x0082d, 0x00951, 0x00953, 0x00954,
    0x00f82, 0x00f83, 0x00f86, 0x00f87, 0x0135d, 0x0135e, 0x0135f, 0x017dd, 0x0193a, 0x01a17, 0x01a75, 0x01a76,
    0x01a77, 0x01a78, 0x01a79, 0x01a7a, 0x01a7b, 0x01a7c, 0x01b6b, 0x01b6d, 0x01b6e, 0x01b6f, 0x01b70, 0x01b71,
    0x01b72, 0x01b73, 0x01cd0, 0x01cd1, 0x01cd2, 0x01cda, 0x01cdb, 0x01ce0, 0x01dc0, 0x01dc1, 0x01dc3, 0x01dc4,
    0x01dc5, 0x01dc6, 0x01dc7, 0x01dc8, 0x01dc9, 0x01dcb, 0x01dcc, 0x01dd1, 0x01dd2, 0x01dd3, 0x01dd4, 0x01dd5,
    0x01dd6, 0x01dd7, 0x01dd8, 0x01dd9, 0x01dda, 0x01ddb, 0x01ddc, 0x01ddd, 0x01dde, 0x01ddf, 0x01de0, 0x01de1,
    0x01de2, 0x01de3, 0x01de4, 0x01de5, 0x01de6, 0x01dfe, 0x020d0, 0x020d1, 0x020d4, 0x020d5, 0x020d6, 0x020d7,
    0x020db, 0x020dc, 0x020e1, 0x020e7, 0x020e9, 0x020f0, 0x02cef, 0x02cf0, 0x02cf1, 0x02de0, 0x02de1, 0x02de2,
    0x02de3, 0x02de4, 0x02de5, 0x02de6, 0x02de7, 0x02de8, 0x02de9, 0x02dea, 0x02deb, 0x02dec, 0x02ded, 0x02dee,
    0x02def, 0x02df0, 0x02df1, 0x02df2, 0x02df3, 0x02df4, 0x02df5, 0x02df6, 0x02df7, 0x02df8, 0x02df9, 0x02dfa,
    0x02dfb, 0x02dfc, 0x02dfd, 0x02dfe, 0x02dff, 0x0a66f, 0x0a67c, 0x0a67d, 0x0a6f0, 0x0a6f1, 0x0a8e0, 0x0a8e1,
    0x0a8e2, 0x0a8e3, 0x0a8e4, 0x0a8e5, 0x0a8e6, 0x0a8e7, 0x0a8e8, 0x0a8e9, 0x0a8ea, 0x0a8eb, 0x0a8ec, 0x0a8ed,
    0x0a8ee, 0x0a8ef, 0x0a8f0, 0x0a8f1, 0x0aab0, 0x0aab2, 0x0aab3, 0x0aab7, 0x0aab8, 0x0aabe, 0x0aabf, 0x0aac1,
    0x0fe20, 0x0fe21, 0x0fe22, 0x0fe23, 0x0fe24, 0x0fe25, 0x0fe26, 0x10a0f, 0x10a38, 0x1d185, 0x1d186, 0x1d187,
    0x1d188, 0x1d189, 0x1d1aa, 0x1d1ab, 0x1d1ac, 0x1d1ad, 0x1d242, 0x1d243, 0x1d244
]
# fmt: on


def _send_tgp_message(*, payload: str | None = None, **kwargs: int | str | None) -> None:
  ans = [
    "\x1b_G",
    ",".join(f"{k}={v}" for k, v in kwargs.items() if v is not None),
    f";{payload}" if payload else "",
    "\x1b\\",
  ]
  sequence = "".join(ans)
  if "TMUX" in os.environ:
    # Wrap in tmux DCS passthrough escape if inside tmux
    sequence = f"\x1bPtmux;\x1b{sequence}\x1b\\"
  with _TERMINAL_WRITE_LOCK:
    if not sys.__stdout__:
      raise RuntimeError("sys.__stdout__ is None")
    sys.__stdout__.write(sequence)
    sys.__stdout__.flush()


def _safe_send_tgp_message(*, payload: str | None = None, **kwargs: int | str | None) -> None:
  _send_tgp_message(payload=payload, **kwargs)


def prepare_image(filepath: str, size: tuple[int, int]) -> PILImage.Image:
  """Return an orientation-corrected RGB image no larger than ``size``."""
  with helpers.read_image(filepath) as opened:
    image = ImageOps.exif_transpose(opened)
    image.thumbnail(size, PILImage.Resampling.LANCZOS)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
      rgba = image.convert("RGBA")
      background = PILImage.new("RGBA", rgba.size, "#121212")
      background.alpha_composite(rgba)
      image = background
    rgb = image.convert("RGB")
    setattr(rgb, "source_path", filepath)
    return rgb


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
    self.source = source
    self.kitty_id = kitty_id
    self.decoded = decoded
    self.refs = 0
    self.ready = Event()
    self.error: BaseException | None = None


class _ImageCache:
  """Keep transmitted Kitty image data alive for later placements.

  Kitty keeps an image's data until it is deleted or the per-buffer quota silently evicts the oldest
  image. Deleting only the placement (`d=i`) leaves the data in place, so showing the same image
  again needs a placement but no transmission.
  """

  def __init__(self, budget: int = _TGP_DECODED_BUDGET) -> None:
    self._lock = Lock()
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
    with self._lock:
      image = self._by_key.get(key)
      if image is not None:
        self._use(image)
        should_transmit = False
      else:
        image = _TransmittedImage(key, source, next(_IMAGE_ID_COUNTER), decoded)
        self._by_key[key] = image
        self._decoded += image.decoded
        self._use(image)
        self._evict()
        should_transmit = True

    if should_transmit:
      try:
        transmit(image.kitty_id)
      except BaseException as error:
        with self._lock:
          self._by_key.pop(key, None)
          self._lru.pop(image, None)
          self._decoded -= image.decoded
        image.error = error
        image.ready.set()
        raise
      else:
        image.ready.set()
    else:
      image.ready.wait()
      if image.error is not None:
        raise image.error

    return image

  def release(self, image: _TransmittedImage) -> None:
    with self._lock:
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
      _safe_send_tgp_message(a="d", d="I", i=image.kitty_id, q=2)


_IMAGE_CACHE = _ImageCache()


def _transmit_image(kitty_id: int, image: PILImage.Image, size: tuple[int, int]) -> None:
  """Send image data to the terminal as a base64 PNG."""
  buffer = BytesIO()
  scaled = image.resize(size, PILImage.Resampling.LANCZOS)
  if scaled.mode in ("RGBA", "LA") and scaled.getextrema()[-1] == (255, 255):
    scaled = scaled.convert("RGB")
  scaled.save(buffer, format="png", compress_level=_PNG_COMPRESS_LEVEL)
  payload = b64encode(buffer.getvalue()).decode("ascii")
  chunks: list[str] = []
  while payload:
    chunk, payload = payload[:4096], payload[4096:]
    m = 1 if payload else 0
    seq = f"\x1b_Gi={kitty_id},m={m},f=100,q=2;{chunk}\x1b\\"
    if "TMUX" in os.environ:
      seq = f"\x1bPtmux;\x1b{seq}\x1b\\"
    chunks.append(seq)
  sequence = "".join(chunks)
  with _TERMINAL_WRITE_LOCK:
    if not sys.__stdout__:
      raise RuntimeError("sys.__stdout__ is None")
    sys.__stdout__.write(sequence)
    sys.__stdout__.flush()


def fit_image_to_cells(
  img_w: int,
  img_h: int,
  max_cols: int,
  max_rows: int,
  cell_size: CellSize | None = None,
) -> tuple[int, int, int, int]:
  """Fit an image into cell bounding box preserving aspect ratio.

  Returns (cols, rows, pixel_w, pixel_h).
  """
  if img_w <= 0 or img_h <= 0 or max_cols <= 0 or max_rows <= 0:
    return 0, 0, 0, 0
  if cell_size is None:
    cell_size = get_cell_pixel_size()

  max_pixel_w = max_cols * cell_size.width
  max_pixel_h = max_rows * cell_size.height
  img_aspect = img_w / img_h
  box_aspect = max_pixel_w / max_pixel_h

  if img_aspect > box_aspect:
    pixel_w = max_pixel_w
    pixel_h = max(1, round(pixel_w / img_aspect))
  else:
    pixel_h = max_pixel_h
    pixel_w = max(1, round(pixel_h * img_aspect))

  cols = min(max_cols, max(1, round(pixel_w / cell_size.width)))
  rows = min(max_rows, max(1, round(pixel_h / cell_size.height)))
  return cols, rows, pixel_w, pixel_h


def render_kitty_placeholders(image_id: int, cols: int, rows: int, placement_id: int | None = None) -> list[str]:
  """Generate Unicode placeholder character strings for each row of the placed Kitty image."""
  r_color = (image_id >> 16) & 255
  g_color = (image_id >> 8) & 255
  b_color = image_id & 255
  color_seq = f"\x1b[38;2;{r_color};{g_color};{b_color}m"
  id_char = chr(_NUMBER_TO_DIACRITIC[(image_id >> 24) & 255])
  ph = chr(_PLACEHOLDER)

  underline_seq = ""
  reset_seq = "\x1b[39m"
  if placement_id is not None:
    pr = (placement_id >> 16) & 255
    pg = (placement_id >> 8) & 255
    pb = placement_id & 255
    underline_seq = f"\x1b[58;2;{pr};{pg};{pb}m"
    reset_seq = "\x1b[39m\x1b[59m"

  lines: list[str] = []
  for r in range(rows):
    r_char = chr(_NUMBER_TO_DIACRITIC[r % len(_NUMBER_TO_DIACRITIC)])
    chars: list[str] = []
    for c in range(cols):
      c_char = chr(_NUMBER_TO_DIACRITIC[c % len(_NUMBER_TO_DIACRITIC)])
      chars.append(f"{ph}{r_char}{c_char}{id_char}")
    lines.append(f"{color_seq}{underline_seq}{''.join(chars)}{reset_seq}")
  return lines


def create_placement(image_id: int, placement_id: int, cols: int, rows: int) -> None:
  _safe_send_tgp_message(a="p", i=image_id, p=placement_id, c=cols, r=rows, U=1, q=2)


def delete_placement(image_id: int, placement_id: int) -> None:
  _safe_send_tgp_message(a="d", d="i", i=image_id, p=placement_id, q=2)


def next_placement_id() -> int:
  return next(_PLACEMENT_ID_COUNTER)


def prewarm_image(
  *args: object,
  cell_size: CellSize | None = None,
  **kwargs: object,
) -> int | None:
  """Pre-scale, encode, and transmit Kitty image data in the background before rendering."""
  widget_type = args[0] if len(args) >= 4 and isinstance(args[0], type) else StableTGPImage
  if len(args) >= 4 and isinstance(args[0], type):
    image = args[1]
    max_cols = int(args[2]) if isinstance(args[2], (int, float)) else 0
    max_rows = int(args[3]) if isinstance(args[3], (int, float)) else 0
  elif len(args) >= 3:
    image = args[0]
    max_cols = int(args[1]) if isinstance(args[1], (int, float)) else 0
    max_rows = int(args[2]) if isinstance(args[2], (int, float)) else 0
  else:
    return None

  styled_size = kwargs.get("styled_size")
  if isinstance(styled_size, tuple) and len(styled_size) == 2:
    sw, sh = styled_size
    if isinstance(sw, (int, float)):
      max_cols = int(sw)
    if isinstance(sh, (int, float)):
      max_rows = int(sh)

  if not isinstance(image, PILImage.Image) or max_cols <= 0 or max_rows <= 0:
    return None

  if cell_size is None:
    cell_size = get_cell_pixel_size()

  cols, rows, pixel_w, pixel_h = fit_image_to_cells(image.width, image.height, max_cols, max_rows, cell_size)
  if cols <= 0 or rows <= 0:
    return None
  box = (max_cols, max_rows)
  renderable_class = getattr(widget_type, "_Renderable", _TGPRenderable)
  source_id = _extract_source(image)
  key = (renderable_class, source_id, image.width, image.height, box)

  with _IMAGE_CACHE._lock:
    if key in _IMAGE_CACHE._by_key:
      return _IMAGE_CACHE._by_key[key].kitty_id

  decoded = max_cols * cell_size.width * max_rows * cell_size.height * 4

  def transmit(kitty_id: int) -> None:
    _transmit_image(kitty_id, image, (pixel_w, pixel_h))

  try:
    transmitted = _IMAGE_CACHE.acquire(key, image, decoded, transmit)
  except Exception:
    return None
  _IMAGE_CACHE.release(transmitted)
  return transmitted.kitty_id


class _TGPRenderable:
  def __init__(self, image: object, width: int | str | None = None, height: int | str | None = None) -> None:
    if isinstance(image, (str, Path)):
      self.image = helpers.read_image(str(image))
      setattr(self.image, "source_path", str(Path(image).resolve()))
    else:
      self.image = image
    self.width = width
    self.height = height
    self.terminal_image_id: int | None = None
    self._placement_id = next_placement_id()
    self._transmitted: _TransmittedImage | None = None

  def _create_virtual_placement(self, width: int | str | None = None, height: int | str | None = None) -> None:
    img = self.image
    if not isinstance(img, PILImage.Image):
      return
    cell_size = get_cell_pixel_size()
    w_arg = width if width is not None else self.width
    h_arg = height if height is not None else self.height
    max_w = int(w_arg) if isinstance(w_arg, (int, float)) else 24
    max_h = int(h_arg) if isinstance(h_arg, (int, float)) else 12
    cols, rows, pw, ph = fit_image_to_cells(img.width, img.height, max_w, max_h, cell_size)
    if cols <= 0 or rows <= 0:
      return
    box = (max_w, max_h)
    key = (self.__class__, _extract_source(img), img.width, img.height, box)
    if self._transmitted is not None and self._transmitted.key == key:
      return
    if self._transmitted is not None:
      _IMAGE_CACHE.release(self._transmitted)
      self._transmitted = None
    decoded = max_w * cell_size.width * max_h * cell_size.height * 4
    self._transmitted = _IMAGE_CACHE.acquire(
      key, img, decoded, lambda kid: _transmit_image(kid, img, (pw, ph))
    )
    self.terminal_image_id = self._transmitted.kitty_id
    create_placement(self.terminal_image_id, self._placement_id, cols, rows)

  def cleanup(self) -> None:
    if self.terminal_image_id is not None:
      delete_placement(self.terminal_image_id, self._placement_id)
      self.terminal_image_id = None
    if self._transmitted is not None:
      _IMAGE_CACHE.release(self._transmitted)
      self._transmitted = None

  def __rich_console__(self, console: object, options: object) -> Iterator[object]:
    self._create_virtual_placement(self.width, self.height)
    return iter([])


class _CenteredTGPRenderable(_TGPRenderable):
  def _create_virtual_placement(self, width: int | str | None = None, height: int | str | None = None) -> None:
    if not isinstance(self.image, PILImage.Image):
      return
    cell_size = get_cell_pixel_size()
    w_arg = width if width is not None else self.width
    h_arg = height if height is not None else self.height
    max_w = int(w_arg) if isinstance(w_arg, (int, float)) else 13
    max_h = int(h_arg) if isinstance(h_arg, (int, float)) else 5
    box_px_w = max_w * cell_size.width
    box_px_h = max_h * cell_size.height
    key = (_extract_source(self.image), self.image.width, self.image.height, (max_w, max_h), "centered")
    if self._transmitted is not None and self._transmitted.key == key:
      return
    if self._transmitted is not None:
      _IMAGE_CACHE.release(self._transmitted)
      self._transmitted = None
    padded = ImageOps.pad(self.image.convert("RGBA"), (box_px_w, box_px_h), color=(0, 0, 0, 0))
    self._transmitted = _IMAGE_CACHE.acquire(
      key, padded, box_px_w * box_px_h * 4, lambda kid: _transmit_image(kid, padded, (box_px_w, box_px_h))
    )
    self.terminal_image_id = self._transmitted.kitty_id
    create_placement(self.terminal_image_id, self._placement_id, max_w, max_h)

  def __rich_console__(self, console: object, options: object) -> Iterator[object]:
    self._create_virtual_placement(self.width, self.height)
    return iter([])


class StableTGPImage:
  _Renderable: type = _TGPRenderable
  display: bool = True
  region: Any = None
  parent: Any = None

  def __init__(self, image: object = None, classes: str | None = None) -> None:
    self._image = image
    self.classes = classes
    from types import SimpleNamespace

    self.content_size = SimpleNamespace(width=20, height=10)
    self.display = True
    self.region = None
    self.parent = None
    self._rendered_size: object = None
    self._renderable: Any = None

  @property
  def image(self) -> object:
    return self._image

  @image.setter
  def image(self, val: object) -> None:
    if self._image is not val:
      self._image = val
      self._discard_renderable()

  def render(self) -> object:
    if self._rendered_size != self.content_size:
      self._discard_renderable()
    if self._renderable is None and self._image is not None:
      self._renderable = self._Renderable(self._image)
    self._rendered_size = self.content_size
    return self._renderable

  def refresh_image(self) -> None:
    self._discard_renderable()

  def on_unmount(self) -> None:
    self._discard_renderable()

  def _discard_renderable(self) -> None:
    if self._renderable is not None:
      if hasattr(self._renderable, "cleanup"):
        self._renderable.cleanup()
      self._renderable = None
    self._rendered_size = None


class CenteredTGPImage(StableTGPImage):
  _Renderable: type = _CenteredTGPRenderable


ImageWidget = StableTGPImage

