from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Semaphore, Thread
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from PIL import Image as PILImage

from rclip.tui.media import (
  CenteredTGPImage,
  ImageWidget,
  prepare_image,
  prewarm_image,
)

if TYPE_CHECKING:
  from rclip.tui.app import RclipApp


PREVIEW_SIZE = (640, 480)
DETAIL_SIZE = (1920, 1920)
_IMAGE_DECODES = Semaphore(4)


@dataclass(frozen=True)
class Region:
  x: int = 0
  y: int = 0
  width: int = 0
  height: int = 0

  def __bool__(self) -> bool:
    return self.width > 0 and self.height > 0

  @property
  def bottom(self) -> int:
    return self.y + self.height

  @property
  def right(self) -> int:
    return self.x + self.width

  def contains_region(self, other: Region) -> bool:
    return (
      self.x <= other.x
      and self.y <= other.y
      and self.right >= other.right
      and self.bottom >= other.bottom
    )


@dataclass(frozen=True)
class Size:
  width: int = 0
  height: int = 0


@dataclass(frozen=True)
class TuiResult:
  filepath: str
  score: float | None = None


@dataclass(frozen=True)
class GridCard:
  index: int
  col: int
  row: int
  x: int
  y: int
  width: int
  height: int
  result: TuiResult


class GridGeometry:
  """Calculates 2D card grid layout, columns, rows, and viewport coordinates."""

  def __init__(
    self,
    total_items: int,
    avail_x: int,
    avail_y: int,
    avail_w: int,
    avail_h: int,
    min_col_width: int = 24,
    card_height: int = 12,
  ) -> None:
    self.total_items = total_items
    self.avail_x = avail_x
    self.avail_y = avail_y
    self.avail_w = max(min_col_width, avail_w)
    self.avail_h = max(card_height, avail_h)
    self.min_col_width = min_col_width
    self.card_height = card_height

    self.cols = max(1, self.avail_w // self.min_col_width)
    self.card_width = self.avail_w // self.cols
    self.visible_rows = max(1, (self.avail_h + self.card_height - 1) // self.card_height)
    self.total_rows = (total_items + self.cols - 1) // self.cols if total_items > 0 else 0

  def get_visible_cards(self, scroll_row: int, results: Sequence[TuiResult]) -> list[GridCard]:
    cards: list[GridCard] = []
    start_row = max(0, scroll_row)
    end_row = min(self.total_rows, start_row + self.visible_rows)

    for r in range(start_row, end_row):
      for c in range(self.cols):
        idx = r * self.cols + c
        if idx >= len(results):
          break
        x = self.avail_x + c * self.card_width
        y = self.avail_y + (r - start_row) * self.card_height
        cards.append(
          GridCard(
            index=idx,
            col=c,
            row=r,
            x=x,
            y=y,
            width=self.card_width,
            height=self.card_height,
            result=results[idx],
          )
        )
    return cards

  def ensure_visible(self, selected_index: int, current_scroll_row: int) -> int:
    if self.cols <= 0 or selected_index < 0:
      return current_scroll_row
    selected_row = selected_index // self.cols
    if selected_row < current_scroll_row:
      return selected_row
    if selected_row >= current_scroll_row + self.visible_rows:
      return selected_row - self.visible_rows + 1
    return current_scroll_row


@dataclass(frozen=True)
class FilmstripThumbnail:
  offset: int
  filepath: str | None
  x: int
  y: int
  width: int
  height: int
  is_selected: bool


class DetailGeometry:
  """Calculates layout for detail view (main image area + bottom filmstrip)."""

  def __init__(self, avail_x: int, avail_y: int, avail_w: int, avail_h: int) -> None:
    self.avail_x = avail_x
    self.avail_y = avail_y
    self.avail_w = avail_w
    self.avail_h = avail_h

    self.filmstrip_h = min(6, max(4, avail_h // 5))
    self.main_h = max(5, avail_h - self.filmstrip_h - 1)

    self.main_x = avail_x
    self.main_y = avail_y
    self.main_w = avail_w

    self.filmstrip_x = avail_x
    self.filmstrip_y = avail_y + self.main_h + 1
    self.filmstrip_w = avail_w

  def get_filmstrip_thumbnails(
    self,
    selected_index: int,
    results: Sequence[TuiResult],
    thumb_width: int = 16,
  ) -> list[FilmstripThumbnail]:
    thumbnails: list[FilmstripThumbnail] = []
    if thumb_width <= 0:
      return thumbnails

    num_thumbs = max(1, self.filmstrip_w // thumb_width)
    if num_thumbs % 2 == 0:
      num_thumbs -= 1
    half = num_thumbs // 2

    start_x = self.filmstrip_x + (self.filmstrip_w - (num_thumbs * thumb_width)) // 2

    for i in range(-half, half + 1):
      idx = selected_index + i
      path = results[idx].filepath if 0 <= idx < len(results) else None
      x = start_x + (i + half) * thumb_width
      thumbnails.append(
        FilmstripThumbnail(
          offset=i,
          filepath=path,
          x=x,
          y=self.filmstrip_y,
          width=thumb_width,
          height=self.filmstrip_h,
          is_selected=(i == 0),
        )
      )
    return thumbnails


class Static:
  def __init__(self, content: str = "", id: str | None = None, markup: bool = True) -> None:
    self._content = content
    self.id = id
    self.markup = markup
    self.display = True
    self.region = Region(0, 38, 100, 1)

  @property
  def content(self) -> str:
    return self._content

  def update(self, content: str) -> None:
    self._content = content

  def __str__(self) -> str:
    return str(self._content)


class Input:
  class Changed:
    pass

  class Submitted:
    pass

  def __init__(
    self,
    placeholder: str | RclipApp = "",
    id: str | None = None,
    select_on_focus: bool = True,
    markup: bool = False,
    app: RclipApp | None = None,
    **kwargs: object,
  ) -> None:
    if not isinstance(placeholder, str):
      app = placeholder
      placeholder = ""
    self._app = app
    self.id = id
    self.placeholder = placeholder
    self.select_on_focus = select_on_focus
    self._value = ""
    self._cursor_position = 0
    self.border_title = ""
    self.visible = True
    self.region = Region(0, 0, 100, 3)
    self._has_selection = False

  @property
  def cursor_position(self) -> int:
    return self._cursor_position

  @cursor_position.setter
  def cursor_position(self, pos: int) -> None:
    self._cursor_position = pos
    self._has_selection = False

  @property
  def value(self) -> str:
    return self._value

  @value.setter
  def value(self, new_val: str) -> None:
    self._value = new_val
    self._cursor_position = len(new_val)
    self._has_selection = False
    if self._app is not None:
      self._app.search_text = new_val
      if Input.Changed not in getattr(self._app, "_prevented_events", set()):
        self._app._on_input_changed(new_val)

  def on_focus(self) -> None:
    if self.select_on_focus and self._value:
      self._has_selection = True

  def insert_text(self, text: str) -> None:
    if self._has_selection:
      self._value = ""
      self.cursor_position = 0
      self._has_selection = False
    left = self._value[: self.cursor_position]
    right = self._value[self.cursor_position :]
    self._value = left + text + right
    self.cursor_position += len(text)
    if self._app is not None and Input.Changed not in getattr(self._app, "_prevented_events", set()):
      self._app._on_input_changed(self._value)

  def delete_backward(self) -> None:
    if self._has_selection:
      self._value = ""
      self.cursor_position = 0
      self._has_selection = False
      if self._app is not None and Input.Changed not in getattr(self._app, "_prevented_events", set()):
        self._app._on_input_changed(self._value)
      return
    if self.cursor_position > 0:
      left = self._value[: self.cursor_position - 1]
      right = self._value[self.cursor_position :]
      self._value = left + right
      self.cursor_position -= 1
      if self._app is not None and Input.Changed not in getattr(self._app, "_prevented_events", set()):
        self._app._on_input_changed(self._value)

  def submit(self) -> None:
    self._has_selection = False
    if self._app is not None:
      self._app._on_input_submitted()

  def focus(self) -> None:
    if self._app is not None:
      self._app.focused = self
    self.on_focus()


class Worker:
  def __init__(self) -> None:
    self.is_cancelled = False

  def cancel(self) -> None:
    self.is_cancelled = True


class ImageCard:
  virtual_region: Region = Region(0, 0, 24, 12)

  def __init__(self, result: TuiResult, app: RclipApp | None = None) -> None:
    self.result = result
    self.app = app
    self._loading = False
    self._loaded = False
    self.is_attached = True
    self.virtual_region = Region(0, 0, 24, 12)
    self.region = Region(0, 0, 24, 12)
    self._image = ImageWidget(classes="thumbnail-image")
    self._image.parent = SimpleNamespace(size=Size(24, 10))

  def load_preview(self, frame_size: Size | None = None) -> None:
    if self._loading or self._loaded:
      return
    self._loading = True
    app = self.app
    tracker = getattr(app, "_workers_tracker", None) if app is not None else None
    worker = Worker()
    if tracker is not None:
      tracker.add(worker)

    def run() -> None:
      try:
        with _IMAGE_DECODES:
          if worker.is_cancelled:
            return
          preview = prepare_image(self.result.filepath, PREVIEW_SIZE)
          target_size = frame_size or Size(24, 10)
          prewarm_image(ImageWidget, preview, target_size.width, target_size.height)
      except Exception:
        self._loading = False
      else:
        if not worker.is_cancelled and self.is_attached:
          self._image.image = preview
          self._loaded = True
          if self.app is not None and hasattr(self.app, "request_render"):
            self.app.request_render()
        self._loading = False
      finally:
        if tracker is not None:
          tracker.discard(worker)

    Thread(target=run, daemon=True).start()

  def focus(self) -> None:
    if self.app is not None:
      self.app.select_card(self)


class ResultsGrid:
  max_scroll_y: int = 100
  scroll_y: int = 0

  def __init__(self, load_more: Callable[[], None] | None = None, app: RclipApp | None = None) -> None:
    self._load_more = load_more
    self.app = app
    self.display = True
    self.region = Region(0, 3, 100, 35)
    self.scrollable_content_region = Region(0, 3, 100, 35)
    self._cached_cards: dict[str, ImageCard] = {}

  @property
  def cards(self) -> list[ImageCard]:
    if self.app is None:
      return []
    cards: list[ImageCard] = []
    for r in self.app.results:
      if r.filepath not in self._cached_cards:
        self._cached_cards[r.filepath] = ImageCard(r, self.app)
      cards.append(self._cached_cards[r.filepath])
    return cards

  def scroll_home(self, animate: bool = False) -> None:
    self.scroll_y = 0
    if self.app is not None:
      self.app.scroll_row = 0
      self.update_visible()

  def scroll_end(self, animate: bool = False) -> None:
    self.scroll_y = self.max_scroll_y
    if self.app is not None:
      w, h = self.app.get_size()
      geo = GridGeometry(len(self.app.results), 0, 3, w, h - 5)
      self.app.scroll_row = max(0, geo.total_rows - geo.visible_rows)
      self.update_visible()

  def update_visible(self) -> None:
    if not self.display or self.app is None:
      return
    cards = self.cards
    if cards and not bool(getattr(cards[-1], "virtual_region", True)):
      return
    w, h = self.app.get_size()
    viewport_h = h - 5
    geo = GridGeometry(len(self.app.results), 0, 3, w, viewport_h)
    visible = geo.get_visible_cards(self.app.scroll_row, self.app.results)
    for card_info in visible:
      if card_info.result.filepath in self._cached_cards:
        self._cached_cards[card_info.result.filepath].load_preview()
      else:
        card = ImageCard(card_info.result, self.app)
        self._cached_cards[card_info.result.filepath] = card
        card.load_preview()
    if self.max_scroll_y - self.scroll_y <= viewport_h:
      if self._load_more is not None:
        self._load_more()


class DetailThumbnail:
  def __init__(self, offset: int = 0, browse: object = None, app: RclipApp | None = None) -> None:
    self.result_offset = offset
    self.browse = browse
    self.app = app
    self.filepath: str | None = None
    self.visible = False
    self.is_attached = True
    self._image = CenteredTGPImage(classes="thumbnail-image")
    self._image.content_size = SimpleNamespace(width=12, height=2)
    self._image.region = Region(0, 0, 12, 2)
    self._frame = SimpleNamespace(region=Region(0, 0, 14, 3))
    self.region = Region(0, 0, 16, 5)
    self._image.parent = SimpleNamespace(size=Size(14, 3))
    self._worker: Worker | None = None

  def query_one(self, selector: str, cls: object = None) -> Any:
    if ".detail-thumbnail-frame" in selector:
      return self._frame
    return self

  def set_class(self, is_selected: bool, class_name: str) -> None:
    if class_name == "selected":
      if is_selected:
        self._image.content_size = SimpleNamespace(width=14, height=4)
        self._frame.region = Region(self.region.x + 1, self.region.y + 1, 16, 5)
        self._image.region = Region(self.region.x + 2, self.region.y + 1, 14, 4)
      else:
        self._image.content_size = SimpleNamespace(width=12, height=2)
        self._frame.region = Region(self.region.x + 1, self.region.y + 1, 14, 3)
        self._image.region = Region(self.region.x + 2, self.region.y + 2, 12, 2)

  def show_image(self, filepath: str | None) -> None:
    self.visible = filepath is not None
    if filepath == self.filepath:
      return
    self.filepath = filepath
    self._image.image = None
    if filepath is not None:
      self._load_preview()

  def load_preview(self) -> None:
    self._load_preview()

  def _load_preview(self) -> None:
    filepath = self.filepath
    if filepath is None:
      return
    app = self.app
    tracker = getattr(app, "_workers_tracker", None) if app is not None else None
    worker = Worker()
    self._worker = worker
    if tracker is not None:
      tracker.add(worker)

    def run() -> None:
      try:
        with _IMAGE_DECODES:
          if worker.is_cancelled:
            return
          preview = prepare_image(filepath, PREVIEW_SIZE)
          prewarm_image(CenteredTGPImage, preview, 14, 3)
      except Exception:
        return
      else:
        if not worker.is_cancelled and self.is_attached and filepath == self.filepath:
          self._image.image = preview
          if self.app is not None and hasattr(self.app, "request_render"):
            self.app.request_render()
      finally:
        if tracker is not None:
          tracker.discard(worker)

    Thread(target=run, daemon=True).start()


class DetailView:
  def __init__(self, browse: object = None, app: RclipApp | None = None) -> None:
    self._browse = browse
    self.app = app
    self._display = False
    self.filepath: str | None = None
    self._image = ImageWidget(classes="detail-image")
    self._image.display = False
    self._image.region = Region(10, 5, 80, 20)
    self._image.parent = SimpleNamespace(size=Size(80, 20))
    self._status = Static("No results", id="detail-status", markup=False)
    self._status.region = Region(20, 10, 60, 3)
    self._frame = SimpleNamespace(region=Region(2, 3, 96, 25))
    self._filmstrip = SimpleNamespace(region=Region(2, 30, 96, 6))
    self.region = Region(0, 3, 100, 35)
    self.thumbnails: list[DetailThumbnail] = []
    self._details: dict[str, PILImage.Image | str] = {}
    self._detail_workers: dict[str, Worker] = {}
    self.is_attached = True

  @property
  def display(self) -> bool:
    if self.app is not None:
      return self.app.view_mode == "detail"
    return self._display

  @display.setter
  def display(self, val: bool) -> None:
    self._display = val
    if self.app is not None:
      self.app.view_mode = "detail" if val else "gallery"

  def query_one(self, selector: str, cls: object = None) -> Any:
    if selector == "#detail-status":
      return self._status
    if selector == "#detail-frame":
      return self._frame
    if selector == "#detail-filmstrip":
      return self._filmstrip
    return self

  def focus(self) -> None:
    if self.app is not None:
      self.app.focused = "detail"

  def on_resize(self, width: int, height: int) -> None:
    neighbors = max(0, (width // 16 - 1) // 2)
    count = neighbors * 2 + 1
    if len(self.thumbnails) != count:
      self.thumbnails = [DetailThumbnail(offset, self._browse, self.app) for offset in range(-neighbors, neighbors + 1)]
    # Update regions for thumbnails and frame
    self._frame.region = Region(2, 3, max(10, width - 4), max(5, height - 10))
    self._filmstrip.region = Region(2, height - 7, max(10, width - 4), 6)
    thumb_w = 16
    start_x = (width - count * thumb_w) // 2
    for idx, thumb in enumerate(self.thumbnails):
      x = start_x + idx * thumb_w
      y = height - 6
      thumb.region = Region(x, y, thumb_w, 5)
      thumb.set_class(thumb.result_offset == 0, "selected")
    if self.app is not None and getattr(self.app, "is_running", False):
      self.app._update_selection()

  def show_thumbnails(self, filepaths: list[str | None]) -> None:
    retained = {
      thumbnail.filepath: thumbnail
      for thumbnail in self.thumbnails
      if thumbnail.filepath is not None and thumbnail.filepath in filepaths
    }
    unused = iter(thumbnail for thumbnail in self.thumbnails if thumbnail not in retained.values())
    self.thumbnails = [retained[filepath] if filepath in retained else next(unused) for filepath in filepaths]
    count = len(filepaths)
    width = self.app.get_size()[0] if self.app is not None else 100
    thumb_w = 16
    start_x = (width - count * thumb_w) // 2
    for index, (thumbnail, filepath) in enumerate(zip(self.thumbnails, filepaths)):
      thumbnail.result_offset = index - count // 2
      thumbnail.region = Region(start_x + index * thumb_w, 32, thumb_w, 5)
      thumbnail.set_class(thumbnail.result_offset == 0, "selected")
      thumbnail.show_image(filepath)

  def show_image(self, filepath: str | None, neighbors: Sequence[str] = ()) -> None:
    wanted = [filepath, *neighbors] if filepath is not None else []
    self._details = {path: detail for path, detail in self._details.items() if path in wanted}
    for path in list(self._detail_workers):
      if path not in wanted:
        self._detail_workers.pop(path).cancel()
    for path in wanted:
      if path not in self._details and path not in self._detail_workers:
        self._load_detail(path)
    if filepath == self.filepath:
      return
    self.filepath = filepath
    self._image.display = False
    self._image.image = None
    self._status.update("No results" if filepath is None else "Loading higher-resolution image…")
    self._status.display = True
    if filepath in self._details:
      self._show_detail(filepath, self._details[filepath])

  def _load_detail(self, filepath: str) -> Worker:
    worker = Worker()
    self._detail_workers[filepath] = worker
    app = self.app
    tracker = getattr(app, "_workers_tracker", None) if app is not None else None
    if tracker is not None:
      tracker.add(worker)

    def run() -> None:
      try:
        with _IMAGE_DECODES:
          if worker.is_cancelled:
            return
          try:
            detail: PILImage.Image | str = prepare_image(filepath, DETAIL_SIZE)
            prewarm_image(ImageWidget, detail, 80, 20)
          except Exception as error:
            detail = str(error)
          if not worker.is_cancelled:
            self._detail_ready(filepath, worker, detail)
      finally:
        if tracker is not None:
          tracker.discard(worker)

    Thread(target=run, daemon=True).start()
    return worker

  def _detail_ready(self, filepath: str, worker: Worker, detail: PILImage.Image | str) -> None:
    if not self.is_attached or self._detail_workers.get(filepath) is not worker:
      return
    del self._detail_workers[filepath]
    self._details[filepath] = detail
    self._show_detail(filepath, detail)
    if self.app is not None and hasattr(self.app, "request_render"):
      self.app.request_render()

  def _show_detail(self, filepath: str, detail: PILImage.Image | str) -> None:
    if not self.is_attached or filepath != self.filepath:
      return
    if isinstance(detail, PILImage.Image):
      self._image.image = detail
      self._image.display = True
      self._status.display = False
      renderable = self._image.render()
      if hasattr(renderable, "_create_virtual_placement"):
        cast(Any, renderable)._create_virtual_placement(80, 20)
    else:
      self._image.display = False
      self._image.image = None
      self._status.update(f"Unable to load image: {detail}")
      self._status.display = True
