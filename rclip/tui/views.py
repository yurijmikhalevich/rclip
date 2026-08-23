from dataclasses import dataclass
from pathlib import Path

from textual import events, work
from textual.app import ComposeResult
from textual.containers import CenterMiddle, ItemGrid
from textual.screen import Screen
from textual.widgets import Label, Static

from rclip.tui.media import ImageWidget
from rclip.tui.media import cache_image


PREVIEW_SIZE = (640, 480)
DETAIL_SIZE = (1920, 1920)


@dataclass(frozen=True)
class TuiResult:
  filepath: str
  score: float | None = None


class ImageCard(Static, can_focus=True):
  def __init__(self, result: TuiResult, cache_dir: Path) -> None:
    super().__init__()
    self.result = result
    self.cache_dir = cache_dir
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
    try:
      preview = cache_image(self.result.filepath, self.cache_dir, PREVIEW_SIZE)
    except Exception:
      self.app.call_from_thread(self._preview_failed)
    else:
      self.app.call_from_thread(self._preview_ready, preview)

  def _preview_ready(self, preview: Path) -> None:
    if self.is_attached:
      self._image.image = preview
    self._loading = False
    self._loaded = True

  def _preview_failed(self) -> None:
    self._loading = False
    self._loaded = True
    self.add_class("preview-failed")

  def on_focus(self) -> None:
    from rclip.tui.app import RclipApp

    if isinstance(self.app, RclipApp):
      self.app.select_card(self)

  def on_click(self, event: events.Click) -> None:
    from rclip.tui.app import RclipApp

    if event.button == 1 and event.chain == 2 and isinstance(self.app, RclipApp):
      self.focus()
      self.app.select_card(self)
      self.app.action_view()


class ResultsGrid(ItemGrid):
  def __init__(self) -> None:
    super().__init__(
      id="results",
      min_column_width=24,
      regular=False,
      stretch_height=False,
    )

  def load_visible_previews(self) -> None:
    viewport = self.scrollable_content_region
    for card in self.query(ImageCard):
      if card.region.overlaps(viewport):
        card.load_preview()

  def watch_scroll_y(self, old_value: float, new_value: float) -> None:
    super().watch_scroll_y(old_value, new_value)
    self.call_after_refresh(self.load_visible_previews)

  def on_resize(self, _event: events.Resize) -> None:
    self.call_after_refresh(self.load_visible_previews)


class DetailScreen(Screen[None]):
  def __init__(self, filepath: str, cache_dir: Path) -> None:
    super().__init__()
    self.filepath = filepath
    self.cache_dir = cache_dir
    self._image = ImageWidget(classes="detail-image")

  def compose(self) -> ComposeResult:
    with CenterMiddle(id="detail-frame"):
      yield self._image
    yield Static("Loading higher-resolution image…", id="detail-status", markup=False)
    yield Static(self.filepath, id="detail-path", markup=False)
    yield Static(
      "h/l/Arrows Browse   Esc/Double-click Back   y Copy image   Y Copy path   q/Ctrl+C Quit",
      classes="hotkeys",
      markup=False,
    )

  def on_mount(self) -> None:
    self._load_detail()

  def show_image(self, filepath: str) -> None:
    self.filepath = filepath
    self._image.image = None
    status = self.query_one("#detail-status", Static)
    status.update("Loading higher-resolution image…")
    status.display = True
    self.query_one("#detail-path", Static).update(filepath)
    self._load_detail()

  def on_click(self, event: events.Click) -> None:
    from rclip.tui.app import RclipApp

    if event.button == 1 and event.chain == 2 and isinstance(self.app, RclipApp):
      event.stop()
      self.app.action_go_back()

  @work(thread=True, group="detail", exclusive=True, exit_on_error=False)
  def _load_detail(self) -> None:
    filepath = self.filepath
    try:
      detail = cache_image(filepath, self.cache_dir, DETAIL_SIZE)
    except Exception as error:
      self.app.call_from_thread(self._show_error, filepath, str(error))
    else:
      self.app.call_from_thread(self._show_detail, filepath, detail)

  def _show_detail(self, filepath: str, detail: Path) -> None:
    if not self.is_attached or filepath != self.filepath:
      return
    self._image.image = detail
    self.query_one("#detail-status", Static).display = False

  def _show_error(self, filepath: str, message: str) -> None:
    if self.is_attached and filepath == self.filepath:
      self.query_one("#detail-status", Static).update(f"Unable to load image: {message}")
