from bisect import bisect_right
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Semaphore
from typing import Callable, Sequence, cast

from textual import events, work
from textual.app import ComposeResult
from textual.containers import CenterMiddle, Horizontal, ItemGrid, Vertical
from textual.worker import get_current_worker
from textual.widgets import Label, Static

from rclip.tui.media import CenteredTGPImage
from rclip.tui.media import ImageWidget
from rclip.tui.media import prepare_image


PREVIEW_SIZE = (640, 480)
DETAIL_SIZE = (1920, 1920)
_IMAGE_DECODES = Semaphore(4)


@dataclass(frozen=True)
class TuiResult:
  filepath: str
  score: float | None = None


class ImageCard(Static, can_focus=True):
  def __init__(self, result: TuiResult) -> None:
    super().__init__()
    self.result = result
    self._loading = False
    self._loaded = False
    self._image = ImageWidget(classes="thumbnail")

  def compose(self) -> ComposeResult:
    with CenterMiddle(classes="thumbnail-frame"):
      yield self._image
    filename = Path(self.result.filepath).name
    label = filename if self.result.score is None else f"{self.result.score:.3f}  {filename}"
    yield Label(label, classes="result-label", markup=False)

  def load_preview(self) -> None:
    if self._loading or self._loaded:
      return
    self._loading = True
    self._load_preview()

  @work(thread=True, exit_on_error=False)
  def _load_preview(self) -> None:
    with _IMAGE_DECODES:
      if get_current_worker().is_cancelled:
        return
      try:
        preview = prepare_image(self.result.filepath, PREVIEW_SIZE)
      except Exception:
        self.app.call_from_thread(self._preview_failed)
      else:
        self.app.call_from_thread(self._preview_ready, preview)

  def _preview_ready(self, preview: BytesIO) -> None:
    if self.is_attached:
      self._image.image = preview
    self._loading = False
    self._loaded = True

  def _preview_failed(self) -> None:
    self._loading = False
    self._loaded = True
    self.add_class("preview-failed")

  def on_click(self, event: events.Click) -> None:
    from rclip.tui.app import RclipApp

    if event.button == 1 and event.chain == 2 and isinstance(self.app, RclipApp):
      event.stop()
      self.focus()
      self.app.select_card(self)
      self.app.action_open_detail()

  def on_focus(self) -> None:
    from rclip.tui.app import RclipApp

    if isinstance(self.app, RclipApp):
      self.app.select_card(self)


class ResultsGrid(ItemGrid, can_focus=True):
  def __init__(self, load_more: Callable[[], None]) -> None:
    super().__init__(
      id="results",
      min_column_width=24,
      regular=False,
      stretch_height=False,
    )
    self._load_more = load_more

  @property
  def cards(self) -> Sequence[ImageCard]:
    # Textual types child collections as generic widgets; this grid mounts only image cards.
    return cast(Sequence[ImageCard], self.children)

  def update_visible(self) -> None:
    if not self.display:
      return
    viewport = self.scrollable_content_region
    cards = self.cards
    index = bisect_right(cards, self.scroll_y, key=lambda card: card.virtual_region.bottom)
    visible_bottom = self.scroll_y + viewport.height
    while index < len(cards):
      card = cards[index]
      if card.virtual_region.y >= visible_bottom:
        break
      card.load_preview()
      index += 1
    if self.max_scroll_y - self.scroll_y <= viewport.height:
      self._load_more()

  def watch_scroll_y(self, old_value: float, new_value: float) -> None:
    super().watch_scroll_y(old_value, new_value)
    self.call_after_refresh(self.update_visible)

  def on_resize(self, _event: events.Resize) -> None:
    self.call_after_refresh(self.update_visible)


class DetailThumbnail(Static):
  def __init__(self, offset: int, browse: Callable[[int], None]) -> None:
    super().__init__(classes="detail-thumbnail selected" if offset == 0 else "detail-thumbnail")
    self.visible = False
    self.result_offset = offset
    self.browse = browse
    self.filepath: str | None = None
    self._image = CenteredTGPImage(classes="thumbnail")

  def compose(self) -> ComposeResult:
    with CenterMiddle(classes="detail-thumbnail-frame"):
      yield self._image

  def show_image(self, filepath: str | None) -> None:
    self.visible = filepath is not None
    if filepath == self.filepath:
      return
    self.filepath = filepath
    self._image.image = None
    self._load_preview()

  @work(thread=True, exclusive=True, exit_on_error=False)
  def _load_preview(self) -> None:
    filepath = self.filepath
    if filepath is None:
      return
    with _IMAGE_DECODES:
      if get_current_worker().is_cancelled:
        return
      try:
        preview = prepare_image(filepath, PREVIEW_SIZE)
      except Exception:
        return
      self.app.call_from_thread(self._show_preview, filepath, preview)

  def _show_preview(self, filepath: str, preview: BytesIO) -> None:
    if self.is_attached and filepath == self.filepath:
      self._image.image = preview

  def on_click(self, event: events.Click) -> None:
    event.stop()
    if event.button == 1 and event.chain == 1 and self.filepath is not None:
      self.browse(self.result_offset)


class DetailView(Vertical, can_focus=True):
  def __init__(self, browse: Callable[[int], None]) -> None:
    super().__init__(id="detail")
    self.display = False
    self.filepath: str | None = None
    self._image = ImageWidget(classes="detail-image")
    self._image.display = False
    self._browse = browse
    self.thumbnails: list[DetailThumbnail] = []

  def compose(self) -> ComposeResult:
    with CenterMiddle(id="detail-frame"):
      yield self._image
      yield Static("No results", id="detail-status", markup=False)
    yield Horizontal(id="detail-filmstrip")

  def on_click(self, event: events.Click) -> None:
    from rclip.tui.app import RclipApp

    frame = self.query_one("#detail-frame")
    if (
      event.button == 1
      and event.chain == 2
      and event.widget is not None
      and (event.widget is frame or frame in event.widget.ancestors)
      and isinstance(self.app, RclipApp)
    ):
      event.stop()
      self.focus()
      self.app.action_toggle_view()

  async def on_resize(self, event: events.Resize) -> None:
    from rclip.tui.app import RclipApp

    neighbors = max(0, (event.size.width // 16 - 1) // 2)
    if len(self.thumbnails) == neighbors * 2 + 1:
      return
    filmstrip = self.query_one("#detail-filmstrip", Horizontal)
    await filmstrip.remove_children()
    self.thumbnails = [DetailThumbnail(offset, self._browse) for offset in range(-neighbors, neighbors + 1)]
    await filmstrip.mount(*self.thumbnails)
    if isinstance(self.app, RclipApp):
      self.app._update_selection()

  def show_thumbnails(self, filepaths: list[str | None]) -> None:
    retained = {
      thumbnail.filepath: thumbnail
      for thumbnail in self.thumbnails
      if thumbnail.filepath is not None and thumbnail.filepath in filepaths
    }
    unused = iter(thumbnail for thumbnail in self.thumbnails if thumbnail not in retained.values())
    self.thumbnails = [retained[filepath] if filepath in retained else next(unused) for filepath in filepaths]
    filmstrip = self.query_one("#detail-filmstrip", Horizontal)
    for index, (thumbnail, filepath) in enumerate(zip(self.thumbnails, filepaths)):
      thumbnail.result_offset = index - len(filepaths) // 2
      thumbnail.set_class(thumbnail.result_offset == 0, "selected")
      thumbnail.show_image(filepath)
      filmstrip.move_child(thumbnail, before=index)

  def show_image(self, filepath: str | None) -> None:
    if filepath == self.filepath:
      return
    self.filepath = filepath
    self._image.display = False
    self._image.image = None
    status = self.query_one("#detail-status", Static)
    status.update("No results" if filepath is None else "Loading higher-resolution image…")
    status.display = True
    if filepath is not None:
      self._load_detail()

  @work(thread=True, group="detail", exclusive=True, exit_on_error=False)
  def _load_detail(self) -> None:
    filepath = self.filepath
    if filepath is None:
      return
    with _IMAGE_DECODES:
      if get_current_worker().is_cancelled:
        return
      try:
        detail = prepare_image(filepath, DETAIL_SIZE)
      except Exception as error:
        self.app.call_from_thread(self._show_error, filepath, str(error))
      else:
        self.app.call_from_thread(self._show_detail, filepath, detail)

  def _show_detail(self, filepath: str, detail: BytesIO) -> None:
    if not self.is_attached or filepath != self.filepath:
      return
    self._image.image = detail
    self._image.display = True
    self.query_one("#detail-status", Static).display = False

  def _show_error(self, filepath: str, message: str) -> None:
    if self.is_attached and filepath == self.filepath:
      self.query_one("#detail-status", Static).update(f"Unable to load image: {message}")
