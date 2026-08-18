from __future__ import annotations

from asyncio import Lock as AsyncLock
import base64
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Literal, TextIO

from PIL import Image as PILImage
from PIL import ImageOps
from textual import events, work
from textual.app import App, ComposeResult, RenderResult
from textual.binding import Binding
from textual.containers import CenterMiddle, ItemGrid
from textual.geometry import Size
from textual.screen import Screen
from textual.timer import Timer
from textual.worker import get_current_worker
from textual.widgets import Input, Label, Static
from textual_image.renderable import Image as TerminalRenderable
from textual_image.renderable import TGPImage as TGPRenderable
from textual_image.widget import Image as TerminalImage
from textual_image.widget import TGPImage

from rclip.utils import helpers
from rclip.utils.preview import iterm_sequence

if TYPE_CHECKING:
  from rclip.main import RClip


RESULT_BATCH_SIZE = 100
PREVIEW_SIZE = (640, 480)
DETAIL_SIZE = (1920, 1920)
ITERM_TRANSFER_CHUNK_SIZE = 512 * 1024
CLIPBOARD_NATIVE_EXTENSIONS = {"bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp"}


class ClipboardError(Exception):
  pass


class DownloadError(Exception):
  pass


@dataclass(frozen=True)
class TuiResult:
  filepath: str
  score: float | None = None


def _cache_path(filepath: str, cache_dir: Path, size: tuple[int, int]) -> Path:
  source = Path(filepath).resolve()
  key = hashlib.sha256(f"{source}\0{size[0]}x{size[1]}".encode()).hexdigest()
  return cache_dir / f"{key}.jpg"


def _display_directory(directory: str) -> str:
  path = Path(directory)
  try:
    relative = path.relative_to(Path.home())
  except ValueError:
    return str(path)
  return str(Path("~") / relative)


def cache_image(filepath: str, cache_dir: Path, size: tuple[int, int]) -> Path:
  """Return an orientation-corrected display image no larger than ``size``."""
  source = Path(filepath)
  target = _cache_path(filepath, cache_dir, size)
  source_mtime = source.stat().st_mtime_ns
  if target.is_file() and target.stat().st_mtime_ns >= source_mtime:
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
      image.convert("RGB").save(temporary_path, "JPEG", quality=88)
    os.replace(temporary_path, target)
  finally:
    temporary_path.unlink(missing_ok=True)
  return target


def _kitten_executable() -> str:
  if executable := shutil.which("kitten"):
    return executable
  if installation_dir := os.getenv("KITTY_INSTALLATION_DIR"):
    executable = Path(installation_dir) / "kitten"
    if executable.is_file():
      return str(executable)
  raise ClipboardError("could not find Kitty's `kitten` executable")


def _run_clipboard_kitten(filepath: Path) -> None:
  completed = subprocess.run(
    [_kitten_executable(), "clipboard", str(filepath)],
    stdin=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
  )
  if completed.returncode:
    message = completed.stderr.strip() or f"kitten exited with status {completed.returncode}"
    raise ClipboardError(message)


def copy_image_to_clipboard(filepath: str) -> None:
  """Copy an image to Kitty's clipboard, converting uncommon formats to PNG."""
  if helpers.get_file_extension(filepath) in CLIPBOARD_NATIVE_EXTENSIONS:
    _run_clipboard_kitten(Path(filepath))
    return

  with tempfile.TemporaryDirectory(prefix="rclip-clipboard-") as temporary:
    converted = Path(temporary) / "image.png"
    with helpers.read_image(filepath) as opened:
      ImageOps.exif_transpose(opened).save(converted, "PNG")
    _run_clipboard_kitten(converted)


def _is_remote_session() -> bool:
  return bool(os.getenv("SSH_CONNECTION") or os.getenv("SSH_TTY"))


def _download_protocol() -> Literal["kitty", "iterm2"]:
  override = os.getenv("RCLIP_DOWNLOAD_PROTOCOL")
  if override == "kitty":
    return "kitty"
  if override == "iterm2":
    return "iterm2"
  if override:
    raise DownloadError("RCLIP_DOWNLOAD_PROTOCOL must be `kitty` or `iterm2`")
  if os.getenv("TERM") == "xterm-kitty" or os.getenv("KITTY_WINDOW_ID") or os.getenv("KITTY_PUBLIC_KEY"):
    return "kitty"
  if os.getenv("TERM_PROGRAM") == "iTerm.app" or os.getenv("LC_TERMINAL") == "iTerm2":
    return "iterm2"
  raise DownloadError("could not detect Kitty or iTerm2; set RCLIP_DOWNLOAD_PROTOCOL to `kitty` or `iterm2`")


def download_image(filepath: str, output: TextIO | None = None) -> None:
  """Download an original image through a remote terminal session."""
  if not _is_remote_session():
    raise DownloadError("image is already local")

  path = Path(filepath)
  if _download_protocol() == "kitty":
    completed = subprocess.run(
      [_kitten_executable(), "transfer", str(path), "Downloads/"],
      stderr=subprocess.PIPE,
      text=True,
    )
    if completed.returncode:
      message = completed.stderr.strip() or f"kitten exited with status {completed.returncode}"
      raise DownloadError(message)
    return

  stream = output or sys.stdout
  name = base64.b64encode(path.name.encode()).decode("ascii")
  stream.write(iterm_sequence(f"MultipartFile=name={name};size={path.stat().st_size};inline=0"))
  with path.open("rb") as image:
    while chunk := image.read(ITERM_TRANSFER_CHUNK_SIZE):
      payload = base64.b64encode(chunk).decode("ascii")
      stream.write(iterm_sequence(f"FilePart={payload}"))
  stream.write(iterm_sequence("FileEnd"))
  stream.flush()


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
    if isinstance(self.app, RclipApp):
      self.app.select_card(self)

  def on_click(self, event: events.Click) -> None:
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
    if self.max_scroll_y - self.scroll_y <= viewport.height:
      if isinstance(self.app, RclipApp):
        self.app.mount_more_results()

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
      "h/l/Arrows Browse   Esc/Double-click Back   y Copy   Y Copy path   d Download   q/Ctrl+C Quit",
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

  async def on_click(self, event: events.Click) -> None:
    if event.button == 1 and event.chain == 2 and isinstance(self.app, RclipApp):
      event.stop()
      await self.app.action_go_back()

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


class RclipApp(App[None]):
  TITLE = "rclip"
  CSS = """
  Screen {
    background: $background;
    color: $foreground;
    layout: vertical;
  }

  #search {
    height: 3;
    margin: 1 1 0 1;
    border: round $border-blurred;
    background: $surface;
  }

  #search:focus {
    border: round $accent;
  }

  #results {
    height: 1fr;
    grid-gutter: 1;
    grid-rows: 16;
    overflow-x: hidden;
    overflow-y: auto;
  }

  ImageCard {
    height: 16;
    layout: vertical;
    border: round $border-blurred;
    background: $surface;
    padding: 0 1;
  }

  ImageCard:focus {
    border: heavy $accent;
    background: $block-hover-background;
  }

  .thumbnail-frame {
    height: 13;
  }

  .thumbnail {
    width: auto;
    height: auto;
    max-width: 100%;
    max-height: 100%;
  }

  .result-label {
    height: 1;
    color: $text-muted;
    text-overflow: ellipsis;
  }

  ImageCard.preview-failed {
    border: round $error;
  }

  .hotkeys {
    height: 1;
    padding: 0 1;
    background: $surface;
    color: $text-muted;
    text-align: center;
  }

  #detail-frame {
    height: 1fr;
    padding: 1;
  }

  #detail-frame .detail-image {
    width: auto;
    height: auto;
    max-width: 100%;
    max-height: 100%;
  }

  #detail-status, #detail-path {
    height: 1;
    padding: 0 1;
    color: $text-muted;
    text-align: center;
    text-overflow: ellipsis;
  }
  """

  BINDINGS: ClassVar[list[Binding]] = [
    Binding("/", "focus_search", "Search", show=False),
    Binding("h,left", "move_left", "Left", show=False),
    Binding("j", "move_down", "Down", show=False),
    Binding("down", "move_down_or_focus", "Down", show=False),
    Binding("k", "move_up", "Up", show=False),
    Binding("up", "move_up_or_focus", "Up", show=False),
    Binding("l,right", "move_right", "Right", show=False),
    Binding("enter", "view", "View", show=False),
    Binding("escape", "go_back", "Back", show=False),
    Binding("y", "copy_image", "Copy image", show=False),
    Binding("Y", "copy_path", "Copy path", show=False),
    Binding("d", "download", "Download", show=False),
    Binding("q", "quit_navigation", "Quit", show=False),
    Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
  ]

  def __init__(
    self,
    rclip: RClip,
    working_directory: str,
    cache_dir: Path,
    initial_query: str | None = None,
    top_k: int | None = None,
    positive_queries: list[str] | None = None,
    negative_queries: list[str] | None = None,
  ) -> None:
    super().__init__()
    self.theme = os.getenv("TEXTUAL_THEME", "ansi-dark")
    self.rclip = rclip
    self.working_directory = working_directory
    self.cache_dir = cache_dir
    self.initial_query = initial_query or ""
    self.top_k = top_k
    self.positive_queries = positive_queries or []
    self.negative_queries = negative_queries or []
    self._search_timer: Timer | None = None
    self._search_lock = Lock()
    self._search_generation = 0
    self._mount_lock = AsyncLock()
    self._mount_requested = False
    self._focus_after_search = False
    self._ignore_initial_change = bool(initial_query)
    self._results: list[TuiResult] = []
    self._mounted_results = 0
    self._selected_index = 0

  def compose(self) -> ComposeResult:
    directory = _display_directory(self.working_directory)
    yield Input(value=self.initial_query, placeholder=f"Search images in {directory}…", id="search")
    yield ResultsGrid()
    yield Static(
      "/ Search   hjkl/Arrows Move   Enter View   y Copy   Y Copy path   d Download   q/Ctrl+C Quit",
      classes="hotkeys",
      markup=False,
    )

  def on_mount(self) -> None:
    self.query_one("#search", Input).focus()
    self._begin_search(self.initial_query.strip())

  def on_input_changed(self, event: Input.Changed) -> None:
    if event.input.id != "search":
      return
    if self._ignore_initial_change and event.value == self.initial_query:
      self._ignore_initial_change = False
      return
    if self._search_timer is not None:
      self._search_timer.stop()
      self._search_timer = None
    query = event.value.strip()
    self._search_timer = self.set_timer(0.25, lambda: self._begin_search(query))

  def on_input_submitted(self, event: Input.Submitted) -> None:
    if event.input.id != "search":
      return
    query = event.value.strip()
    if self._search_timer is not None:
      self._search_timer.stop()
      self._search_timer = None
    self._focus_after_search = True
    self._begin_search(query)

  def _begin_search(self, query: str) -> None:
    self._search_generation += 1
    self.query_one("#search", Input).border_title = "Searching…"
    self._search(query, self._search_generation)

  @work(thread=True, group="search", exclusive=True, exit_on_error=False)
  def _search(self, query: str, generation: int) -> None:
    worker = get_current_worker()
    try:
      with self._search_lock:
        if worker.is_cancelled:
          return
        if query:
          search_results = self.rclip.search(
            query,
            self.working_directory,
            self.top_k,
            self.positive_queries,
            self.negative_queries,
            cancel_event=worker.cancelled_event,
          )
          results: list[TuiResult] = []
          for result in search_results:
            if worker.is_cancelled:
              raise InterruptedError
            results.append(TuiResult(result.filepath, result.score))
        else:
          filepaths = self.rclip.list_images(self.working_directory, self.top_k, cancel_event=worker.cancelled_event)
          results = []
          for filepath in filepaths:
            if worker.is_cancelled:
              raise InterruptedError
            results.append(TuiResult(filepath))
    except InterruptedError:
      return
    except Exception as error:
      self.call_from_thread(self._show_search_error, generation, query, str(error))
    else:
      self.call_from_thread(self._show_results, generation, query, results)

  async def _show_results(self, generation: int, query: str, results: list[TuiResult]) -> None:
    search_input = self.query_one("#search", Input)
    if generation != self._search_generation or search_input.value.strip() != query:
      return
    grid = self.query_one(ResultsGrid)
    self.workers.cancel_group(self, "mount-results")
    self._mount_requested = False
    await grid.remove_children()
    if generation != self._search_generation:
      return
    self._results = results
    self._mounted_results = 0
    await self._mount_results_batch()
    if generation != self._search_generation:
      return
    grid.scroll_home(animate=False)
    self._selected_index = 0
    self.call_after_refresh(grid.load_visible_previews)
    focus_after_search = self._focus_after_search
    self._focus_after_search = False
    if focus_after_search and self._mounted_results:
      self.call_after_refresh(self.query(ImageCard).first().focus)
    search_input.border_title = None

  async def _mount_results_batch(self, through_index: int | None = None) -> None:
    async with self._mount_lock:
      if through_index is not None and through_index < self._mounted_results:
        return
      start = self._mounted_results
      required = -1 if through_index is None else through_index
      stop = min(len(self._results), max(start + RESULT_BATCH_SIZE, required + 1))
      if stop <= start:
        return
      cards = [ImageCard(result, self.cache_dir) for result in self._results[start:stop]]
      await self.query_one(ResultsGrid).mount(*cards)
      self._mounted_results = stop
      self.call_after_refresh(self.query_one(ResultsGrid).load_visible_previews)

  def mount_more_results(self) -> None:
    if self._mount_requested or self._mounted_results >= len(self._results):
      return
    self._mount_requested = True
    self._mount_more_results()

  @work(group="mount-results", exit_on_error=False)
  async def _mount_more_results(self) -> None:
    try:
      await self._mount_results_batch()
    finally:
      self._mount_requested = False

  def _show_search_error(self, generation: int, query: str, message: str) -> None:
    search_input = self.query_one("#search", Input)
    if generation != self._search_generation or search_input.value.strip() != query:
      return
    self._focus_after_search = False
    search_input.border_title = None
    self.notify(message, title="Search failed", severity="error")

  def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
    if isinstance(self.focused, Input) and action in {
      "copy_image",
      "copy_path",
      "download",
      "move_down",
      "move_left",
      "move_right",
      "move_up",
      "move_up_or_focus",
      "quit_navigation",
      "view",
    }:
      return False
    if isinstance(self.screen, DetailScreen):
      if action in {"move_left", "move_right"}:
        return len(self._results) > 1
      if action in {"focus_search", "move_down", "move_down_or_focus", "move_up", "move_up_or_focus", "view"}:
        return False
    if action == "focus_search":
      return not isinstance(self.focused, Input)
    if action in {"copy_image", "copy_path", "download"} and isinstance(self.screen, DetailScreen):
      return True
    if action in {
      "copy_image",
      "copy_path",
      "download",
      "move_down",
      "move_down_or_focus",
      "move_left",
      "move_right",
      "move_up",
      "move_up_or_focus",
      "view",
    }:
      return bool(self.query(ImageCard))
    return super().check_action(action, parameters)

  def select_card(self, card: ImageCard) -> None:
    cards = list(self.query(ImageCard))
    if card in cards:
      self._selected_index = cards.index(card)
      self.call_after_refresh(self.query_one(ResultsGrid).load_visible_previews)

  def _selected_card(self) -> ImageCard | None:
    if isinstance(self.focused, ImageCard):
      return self.focused
    cards = list(self.query(ImageCard))
    return cards[self._selected_index] if self._selected_index < len(cards) else None

  def _selected_filepath(self) -> str | None:
    if isinstance(self.screen, DetailScreen):
      return self.screen.filepath
    card = self._selected_card()
    return card.result.filepath if card else None

  async def _move(self, offset: int) -> None:
    if not self._results:
      return
    index = min(max(self._selected_index + offset, 0), len(self._results) - 1)
    await self._mount_results_batch(index)
    self._selected_index = index
    list(self.query(ImageCard))[index].focus()

  def _columns(self) -> int:
    cards = list(self.query(ImageCard))
    if not cards:
      return 1
    first_row = cards[0].virtual_region.y
    return max(1, sum(card.virtual_region.y == first_row for card in cards))

  def _move_detail(self, offset: int) -> None:
    index = min(max(self._selected_index + offset, 0), len(self._results) - 1)
    if index == self._selected_index:
      return
    self._selected_index = index
    if isinstance(self.screen, DetailScreen):
      self.screen.show_image(self._results[index].filepath)

  async def action_move_left(self) -> None:
    if isinstance(self.screen, DetailScreen):
      self._move_detail(-1)
    else:
      await self._move(-1)

  async def action_move_right(self) -> None:
    if isinstance(self.screen, DetailScreen):
      self._move_detail(1)
    else:
      await self._move(1)

  async def action_move_up(self) -> None:
    await self._move(-self._columns())

  async def action_move_down(self) -> None:
    await self._move(self._columns())

  async def action_move_up_or_focus(self) -> None:
    if isinstance(self.focused, ImageCard) and self._selected_index < self._columns():
      self.action_focus_search()
    else:
      await self.action_move_up()

  async def action_move_down_or_focus(self) -> None:
    if isinstance(self.focused, Input):
      if card := self._selected_card():
        card.focus()
    else:
      await self.action_move_down()

  def action_focus_search(self) -> None:
    self.query_one("#search", Input).focus()

  def action_view(self) -> None:
    if card := self._selected_card():
      self.push_screen(DetailScreen(card.result.filepath, self.cache_dir))

  async def action_go_back(self) -> None:
    if isinstance(self.screen, DetailScreen):
      selected_index = self._selected_index
      await self._mount_results_batch(selected_index)
      self.pop_screen()
      cards = list(self.query(ImageCard))
      if selected_index < len(cards):
        self.call_after_refresh(cards[selected_index].focus)
      return
    if isinstance(self.focused, Input):
      if card := self._selected_card():
        card.focus()
    else:
      self.action_focus_search()

  def action_copy_path(self) -> None:
    if filepath := self._selected_filepath():
      self.copy_to_clipboard(filepath)
      self.notify("Path copied", title=Path(filepath).name)

  def action_copy_image(self) -> None:
    if filepath := self._selected_filepath():
      self._copy_image(filepath)

  def action_download(self) -> None:
    if not (filepath := self._selected_filepath()):
      return
    if not _is_remote_session():
      self.notify(filepath, title="Image is already local")
      return
    try:
      with self.suspend():
        download_image(filepath)
    except Exception as error:
      self.notify(str(error), title="Unable to download image", severity="error")
    else:
      self.notify("Saved to ~/Downloads", title=Path(filepath).name)

  def action_quit_navigation(self) -> None:
    self.exit()

  @work(thread=True, group="clipboard", exclusive=True, exit_on_error=False)
  def _copy_image(self, filepath: str) -> None:
    try:
      copy_image_to_clipboard(filepath)
    except Exception as error:
      self.call_from_thread(self.notify, str(error), title="Unable to copy image", severity="error")
    else:
      self.call_from_thread(self.notify, "Image copied", title=Path(filepath).name)


def run_tui(
  rclip: RClip,
  working_directory: str,
  initial_query: str | None,
  top_k: int | None,
  positive_queries: list[str],
  negative_queries: list[str],
) -> None:
  cache_dir = helpers.get_app_datadir() / "previews"
  RclipApp(
    rclip,
    working_directory,
    cache_dir,
    initial_query,
    top_k,
    positive_queries,
    negative_queries,
  ).run()
