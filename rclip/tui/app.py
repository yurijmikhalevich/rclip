from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from io import StringIO
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, TypeVar, overload

from PIL import Image as PILImage

from rclip.model import Model
from rclip.tui.media import (
  _IMAGE_CACHE,
  _extract_source,
  _transmit_image,
  create_placement,
  delete_placement,
  fit_image_to_cells,
  next_placement_id,
  render_kitty_placeholders,
)
from rclip.tui.terminal import (
  BG_BRIGHT_BLACK,
  BOLD,
  CLEAR_SCREEN,
  DIM,
  FG_BRIGHT_BLACK,
  FG_CYAN,
  FG_DEFAULT,
  FG_RED,
  RESET,
  RawTerminal,
  cursor_to,
  get_terminal_size,
  read_key,
)
from rclip.tui.transfer import (
  _is_remote_session,
  _kitten_executable,
  _probe_kitty,
  copy_image_to_clipboard,
  download_image,
)
from rclip.tui.views import (
  DetailGeometry,
  DetailThumbnail,
  DetailView,
  GridGeometry,
  ImageCard,
  Input,
  Region,
  ResultsGrid,
  Static,
  TuiResult,
  Worker,
)

if TYPE_CHECKING:
  from rclip.main import RClip

_T = TypeVar("_T")


@dataclass(frozen=True)
class SearchResultsEvent:
  generation: int
  query: str
  results: list[TuiResult]
  next_cursor: RClip.ImageCursor | None
  append: bool


@dataclass(frozen=True)
class SearchErrorEvent:
  generation: int
  query: str
  message: str
  append: bool


@dataclass(frozen=True)
class PreviewReadyEvent:
  filepath: str
  preview: PILImage.Image


@dataclass(frozen=True)
class PreviewFailedEvent:
  filepath: str


@dataclass(frozen=True)
class DetailReadyEvent:
  filepath: str
  detail: PILImage.Image | str


type AppEvent = SearchResultsEvent | SearchErrorEvent | PreviewReadyEvent | PreviewFailedEvent | DetailReadyEvent


def _display_directory(directory: str) -> str:
  path = Path(directory)
  try:
    relative = path.relative_to(Path.home())
  except ValueError:
    return str(path)
  return str(Path("~") / relative)


@contextmanager
def _safe_suspend(app: object, message: str) -> Iterator[None]:
  """Suspend the TUI, guaranteeing that it is resumed even if the body raises."""
  failure: BaseException | None = None
  suspend_fn = getattr(app, "suspend", None)
  ctx = suspend_fn() if callable(suspend_fn) else nullcontext()
  with ctx:
    try:
      print(f"rclip: {message}", flush=True)
      yield
    except BaseException as error:
      failure = error
    print("\x1b\\", end="", flush=True)
  if failure is not None:
    raise failure


class _Workers:
  def __init__(self, app: RclipApp) -> None:
    self.app = app

  async def wait_for_complete(self) -> None:
    while True:
      if not self.app._workers_tracker and self.app._event_queue.empty():
        if self.app._search_timer is None or not self.app._search_timer.is_alive():
          break
      self.app.process_events()
      await asyncio.sleep(0.005)
    self.app.process_events()


class TestPilot:
  def __init__(self, app: RclipApp) -> None:
    self.app = app

  async def press(self, *keys: str) -> None:
    for key in keys:
      self.app.handle_key(key)
      self.app.process_events()
      await asyncio.sleep(0.005)

  async def click(self, target: object, times: int = 1) -> None:
    if isinstance(target, ImageCard):
      target.focus()
      if times == 2:
        self.app.action_open_detail()
    elif target == "#detail-frame":
      if times == 2:
        self.app.action_toggle_view()
    elif isinstance(target, DetailThumbnail):
      if times == 2 and target.filepath is not None:
        self.app.browse(target.result_offset)
    elif isinstance(target, str) and target.startswith("#"):
      if target == "#detail-frame" and times == 2:
        self.app.action_toggle_view()
    self.app.process_events()
    await asyncio.sleep(0.005)

  async def pause(self, delay: float = 0.005) -> None:
    self.app.process_events()
    await asyncio.sleep(delay)

  async def resize_terminal(self, width: int, height: int) -> None:
    self.app._on_terminal_resized(width, height)
    self.app.process_events()
    await asyncio.sleep(0.005)


class _RunTestContext:
  def __init__(self, app: RclipApp, size: tuple[int, int]) -> None:
    self.app = app
    self.size = size
    self.pilot = TestPilot(app)
    self._bg_task: asyncio.Task[None] | None = None
    self._thread: threading.Thread | None = None
    self._stop_event = threading.Event()

  def _start(self) -> TestPilot:
    self.app._on_terminal_resized(self.size[0], self.size[1])
    self.app.is_running = True
    self.app._begin_search("")
    return self.pilot

  def _stop(self) -> None:
    self.app.is_running = False
    self._stop_event.set()
    if self._bg_task is not None:
      self._bg_task.cancel()
      self._bg_task = None
    if self.app._search_timer is not None:
      self.app._search_timer.cancel()
      self.app._search_timer = None
    with self.app._search_lock:
      self.app._search_cancel_event.set()
    self.app._cleanup_all_placements()

  def __enter__(self) -> TestPilot:
    pilot = self._start()

    def loop() -> None:
      while not self._stop_event.is_set():
        self.app.process_events()
        time.sleep(0.005)

    self._thread = threading.Thread(target=loop, daemon=True)
    self._thread.start()
    return pilot

  def __exit__(self, *args: object) -> None:
    self._stop()
    if self._thread is not None:
      self._thread.join(timeout=0.1)

  async def __aenter__(self) -> TestPilot:
    pilot = self._start()

    async def loop() -> None:
      try:
        while self.app.is_running:
          self.app.process_events()
          await asyncio.sleep(0.005)
      except asyncio.CancelledError:
        pass

    self._bg_task = asyncio.create_task(loop())
    return pilot

  async def __aexit__(self, *args: object) -> None:
    self._stop()


class RclipApp:
  TITLE = "rclip"
  BROWSE_BATCH_SIZE = 25

  def __init__(
    self,
    rclip: RClip,
    working_directory: str,
    top_k: int = 100,
  ) -> None:
    self.rclip = rclip
    self.working_directory = working_directory
    self.search_batch_size = top_k
    self.theme = os.getenv("TEXTUAL_THEME", "ansi-dark")

    # State
    self.search_text = ""
    self.cursor_pos = 0
    self.selected_index = 0
    self.scroll_row = 0
    self.view_mode: str = "gallery"  # "gallery" | "detail"
    self._focus_target: str = "search"  # "search" | "gallery" | "detail"

    self.results: list[TuiResult] = []
    self._remaining_search_results: list[TuiResult] = []
    self._next_cursor: RClip.ImageCursor | None = None
    self._loading_more = False

    self._search_generation = 0
    self._search_lock = threading.Lock()
    self._search_cancel_event = threading.Event()
    self._search_timer: threading.Timer | None = None
    self.search_status: str | None = None

    # Components
    self._search_input = Input(self, id="search")
    self._grid = ResultsGrid(load_more=self._load_more, app=self)
    self._detail_view = DetailView(browse=self.browse, app=self)
    self._gallery_path = Static("", id="gallery-path", markup=False)
    self._hotkeys = Static("", markup=False)

    # Threading & worker tracking
    self._workers_tracker: set[Worker] = set()
    self.workers = _Workers(self)
    self._event_queue: queue.Queue[AppEvent] = queue.Queue()
    self._prevented_events: set[type] = set()
    self.is_running = False
    self._term: RawTerminal | None = None

    # Image caches & active placements: filepath -> (kitty_id, placement_id)
    self._previews: dict[str, PILImage.Image] = {}
    self._detail_images: dict[str, PILImage.Image | str] = {}
    self._loading_previews: set[str] = set()
    self._loading_detail: set[str] = set()
    self._active_placements: dict[str, tuple[int, int]] = {}
    self._active_detail_placements: dict[str, tuple[int, int]] = {}

    # Notification toast
    self._notification: tuple[str, float, str] | None = None

    self._dirty = True
    self._last_rendered_mode = "gallery"
    self._last_rendered_generation = 0

    # Overrideable size for testing
    self._terminal_size: tuple[int, int] | None = None

  def request_render(self) -> None:
    self._dirty = True

  @property
  def _selected_index(self) -> int:
    return self.selected_index

  @_selected_index.setter
  def _selected_index(self, val: int) -> None:
    self.selected_index = val

  @property
  def cards(self) -> list[ImageCard]:
    return self._grid.cards

  @property
  def focused(self) -> object:
    if self._focus_target == "search":
      return self._search_input
    if self.view_mode == "detail" and self._focus_target == "detail":
      return self._detail_view
    cards = self._grid.cards
    if cards and 0 <= self.selected_index < len(cards):
      return cards[self.selected_index]
    return self._search_input

  @focused.setter
  def focused(self, val: object) -> None:
    if val == "search" or val is self._search_input or getattr(val, "__name__", "") == "Input":
      self._focus_target = "search"
      self._search_input.on_focus()
    elif val == "detail" or val is self._detail_view or getattr(val, "__name__", "") == "DetailView":
      self._focus_target = "detail"
    elif val == "gallery" or isinstance(val, ImageCard):
      self._focus_target = "gallery"
      if isinstance(val, ImageCard):
        self.select_card(val)

  def select_card(self, card: ImageCard) -> None:
    cards = self._grid.cards
    for idx, c in enumerate(cards):
      if c is card or c.result.filepath == card.result.filepath:
        self.selected_index = idx
        self._focus_target = "gallery"
        self._update_selection()
        w, h = self.get_size()
        geo = GridGeometry(len(self.results), 0, 3, w, h - 5)
        self.scroll_row = geo.ensure_visible(self.selected_index, self.scroll_row)
        max_scroll_rows = max(1, geo.total_rows - geo.visible_rows)
        self._grid.scroll_y = int((self.scroll_row / max_scroll_rows) * self._grid.max_scroll_y) if max_scroll_rows > 0 else 0
        self._grid.update_visible()
        break

  def browse(self, offset: int) -> None:
    self.selected_index += offset
    self.selected_index = max(0, min(len(self.results) - 1, self.selected_index))
    self._on_selection_changed()

  @overload
  def query(self, target: type[_T]) -> list[_T]: ...
  @overload
  def query(self, target: object) -> list[Any]: ...
  def query(self, target: object) -> list[Any]:
    name = target.__name__ if isinstance(target, type) else str(target)
    if name == "ImageCard":
      return list(self._grid.cards)
    if name == "ResultsGrid":
      return [self._grid]
    if name == "DetailView":
      return [self._detail_view]
    if name == "Input" or target == "#search":
      return [self._search_input]
    if target == "#gallery-path":
      return [self._gallery_path]
    if target == ".hotkeys":
      return [self._hotkeys]
    return []

  @overload
  def query_one(self, target: type[_T], cls: object = None) -> _T: ...
  @overload
  def query_one(self, target: str, cls: type[_T]) -> _T: ...
  @overload
  def query_one(self, target: object, cls: object = None) -> Any: ...
  def query_one(self, target: object, cls: object = None) -> Any:
    items = self.query(target)
    if items:
      return items[0]
    if target == "#detail-frame":
      return self._detail_view.query_one("#detail-frame")
    if target == "#detail-filmstrip":
      return self._detail_view.query_one("#detail-filmstrip")
    raise KeyError(f"Widget not found: {target}")

  @contextmanager
  def prevent(self, *event_types: type) -> Iterator[None]:
    added = set(event_types)
    self._prevented_events.update(added)
    try:
      yield
    finally:
      self._prevented_events.difference_update(added)

  def post_message(self, event: object) -> None:
    key = getattr(event, "key", None)
    if key is not None:
      self.handle_key(key)

  def exit(self, *args: object, **kwargs: object) -> None:
    self.is_running = False

  def run_test(self, size: tuple[int, int] = (100, 40)) -> _RunTestContext:
    return _RunTestContext(self, size)

  def _selected_card_result(self) -> TuiResult | None:
    if 0 <= self.selected_index < len(self.results):
      return self.results[self.selected_index]
    return None

  def _selected_filepath(self) -> str | None:
    res = self._selected_card_result()
    return res.filepath if res else None

  @contextmanager
  def suspend(self, _message: str = "") -> Iterator[None]:
    if self._term is not None:
      self._term.exit()
    try:
      yield
    finally:
      if self._term is not None:
        self._term.enter()
        self.render()

  def notify(self, message: str, title: str | None = None, severity: str = "info") -> None:
    text = f"{title}: {message}" if title else message
    self._notification = (text, time.time() + 3.0, severity)

  def get_size(self) -> tuple[int, int]:
    if self._terminal_size is not None:
      return self._terminal_size
    return get_terminal_size()

  def set_terminal_size(self, width: int, height: int) -> None:
    self._terminal_size = (max(10, width), max(5, height))

  def _on_terminal_resized(self, width: int, height: int) -> None:
    self.set_terminal_size(width, height)
    self._grid.region = Region(0, 3, width, height - 5)
    self._search_input.region = Region(0, 0, width, 3)
    self._gallery_path.region = Region(0, height - 2, width, 1)
    self._hotkeys.region = Region(0, height - 1, width, 1)
    self._detail_view.region = Region(0, 3, width, height - 5)
    self._detail_view.on_resize(width, height)

  # ── Actions ──────────────────────────────────────────────────────────

  def action_focus_search(self) -> None:
    self._focus_target = "search"
    self._search_input.on_focus()

  def action_toggle_focus(self) -> None:
    if self._focus_target == "search":
      self._focus_target = self.view_mode
      self._on_selection_changed()
    else:
      self._focus_target = "search"
      self._search_input.on_focus()

  def action_escape(self) -> None:
    if self._focus_target == "search":
      self.action_toggle_focus()
    elif self.view_mode == "detail":
      self.action_toggle_view()

  def check_action(self, action: str, parameters: tuple[object, ...] = ()) -> bool:
    if action in {"copy_image", "copy_path", "download"}:
      return self._selected_filepath() is not None and (self.view_mode == "detail" or self._focus_target == "gallery")
    return True

  def _move_detail(self, offset: int) -> None:
    self._pending_detail_advance = False
    if not self.results:
      return
    if self.selected_index + offset >= len(self.results) and (
      self._loading_more or self._next_cursor or self._remaining_search_results
    ):
      self._pending_detail_advance = True
      self._load_more()
      return
    index = min(max(self.selected_index + offset, 0), len(self.results) - 1)
    if index == self.selected_index:
      return
    self.selected_index = index
    self._update_selection()

  def _move(self, offset: int) -> None:
    if not self.results:
      return
    index = min(max(self.selected_index + offset, 0), len(self.results) - 1)
    if index == self.selected_index:
      return
    self.selected_index = index
    self._update_selection()
    w, h = self.get_size()
    geo = GridGeometry(len(self.results), 0, 3, w, h - 5)
    self.scroll_row = geo.ensure_visible(self.selected_index, self.scroll_row)
    max_scroll_rows = max(1, geo.total_rows - geo.visible_rows)
    self._grid.scroll_y = int((self.scroll_row / max_scroll_rows) * self._grid.max_scroll_y) if max_scroll_rows > 0 else 0
    self._grid.update_visible()

  def action_move_left(self) -> None:
    if self.view_mode == "detail":
      self._move_detail(-1)
    else:
      self._move(-1)

  def action_move_right(self) -> None:
    if self.view_mode == "detail":
      self._move_detail(1)
    else:
      self._move(1)

  def action_move_up(self) -> None:
    self._move(-self._columns())

  def action_move_down(self) -> None:
    self._move(self._columns())

  def action_move_up_or_focus(self) -> None:
    if self.view_mode == "detail" or self.selected_index < self._columns():
      self.action_focus_search()
    else:
      self.action_move_up()

  def action_move_down_or_focus(self) -> None:
    if self._focus_target == "search":
      self.action_toggle_focus()
    elif self.view_mode != "detail":
      self.action_move_down()

  def action_toggle_view(self) -> None:
    self._pending_detail_advance = False
    self._cleanup_all_placements()
    if self.view_mode == "gallery":
      self.view_mode = "detail"
      if self._focus_target == "gallery":
        self._focus_target = "detail"
      for card in self._grid.cards:
        card._image.on_unmount()
    else:
      self.view_mode = "gallery"
      if self._focus_target == "detail":
        self._focus_target = "gallery"
      self._detail_view._image.on_unmount()
    self._on_selection_changed()

  def action_open_detail(self) -> None:
    if self.view_mode == "gallery":
      self.action_toggle_view()

  def action_copy_path(self) -> None:
    if filepath := self._selected_filepath():
      self.copy_to_clipboard(filepath)
      self.notify("Path copied", title=Path(filepath).name)

  def copy_to_clipboard(self, filepath: str) -> None:
    if sys.__stdout__ is not None:
      try:
        from base64 import b64encode

        encoded = b64encode(filepath.encode("utf-8")).decode("ascii")
        sys.__stdout__.write(f"\x1b]52;c;{encoded}\x07")
        sys.__stdout__.flush()
      except Exception:
        pass

  def _copy_image(self, filepath: str) -> None:
    try:
      _kitten_executable()
      with _safe_suspend(self, "pausing the TUI to copy the image…"):
        copy_image_to_clipboard(filepath)
    except Exception as error:
      self.notify(str(error), title="Unable to copy image", severity="error")
    else:
      self.notify("Image copied", title=Path(filepath).name)

  def action_copy_image(self) -> None:
    if filepath := self._selected_filepath():
      self._copy_image(filepath)

  def action_download(self) -> None:
    if filepath := self._selected_filepath():
      if not _is_remote_session():
        self.notify(filepath, title="Image is already local")
        return
      try:
        _kitten_executable()
        with _safe_suspend(self, "pausing the TUI to download the image…"):
          download_image(filepath)
      except Exception as error:
        self.notify(str(error), title="Unable to download image", severity="error")
      else:
        self.notify("Saved to ~/Downloads", title=Path(filepath).name)

  def action_quit(self) -> None:
    self.exit()

  # ── Selection & Layout Helpers ────────────────────────────────────────

  def _columns(self) -> int:
    w, _ = self.get_size()
    return max(1, w // 24)

  def _update_selection(self) -> None:
    if not self.is_running:
      return
    filepath = self._selected_filepath()
    if filepath is None:
      self._gallery_path.update("")
    else:
      res = self._selected_card_result()
      if not self.search_text and res is not None:
        self._gallery_path.update(filepath)
      elif res is not None and res.score is not None:
        self._gallery_path.update(f"{res.score:.3f}  {filepath}")
      else:
        self._gallery_path.update(filepath)

    if self.view_mode != "detail":
      return
    self._update_detail_view()
    self._load_more()

  def _update_detail_view(self) -> None:
    filepath = self._selected_filepath()
    neighbors: list[str] = []
    if filepath is not None:
      for offset in (-1, 1):
        idx = self.selected_index + offset
        if 0 <= idx < len(self.results):
          neighbors.append(self.results[idx].filepath)
    self._detail_view.show_image(filepath, neighbors)

    w, h = self.get_size()
    count = len(self._detail_view.thumbnails)
    if count == 0:
      self._detail_view.on_resize(w, h)
      count = len(self._detail_view.thumbnails)
    half = count // 2
    paths: list[str | None] = []
    for i in range(-half, half + 1):
      idx = self.selected_index + i
      paths.append(self.results[idx].filepath if 0 <= idx < len(self.results) else None)
    self._detail_view.show_thumbnails(paths)

  def _on_selection_changed(self) -> None:
    self._update_selection()

  def _can_load_more(self) -> bool:
    if self._loading_more:
      return False
    if self.search_text:
      return bool(self._remaining_search_results)
    return self._next_cursor is not None

  # ── Search & Browse Execution ─────────────────────────────────────────

  def _on_input_changed(self, new_query: str) -> None:
    self.search_text = new_query
    self._search_input._value = new_query
    self.cursor_pos = self._search_input.cursor_position
    if self._search_timer is not None:
      self._search_timer.cancel()
      self._search_timer = None
    self._search_timer = threading.Timer(0.25, lambda: self._begin_search(self.search_text.strip()))
    self._search_timer.daemon = True
    self._search_timer.start()

  def _on_input_submitted(self) -> None:
    if self._search_timer is not None:
      self._search_timer.cancel()
      self._search_timer = None
    self.search_text = self._search_input.value
    self._begin_search(self._search_input.value.strip())

  def _begin_search(self, query: str) -> None:
    self._search_generation += 1
    gen = self._search_generation
    self._search_cancel_event.set()
    self._search_cancel_event = threading.Event()
    cancel_event = self._search_cancel_event

    self._cleanup_all_placements()
    self._loading_more = False
    self._next_cursor = None
    self._remaining_search_results = []
    self.search_status = "Searching…"
    self._search_input.border_title = "Searching…"

    worker = Worker()
    self._workers_tracker.add(worker)

    def run() -> None:
      try:
        self._search_worker(query, gen, None, cancel_event)
      finally:
        self._workers_tracker.discard(worker)

    threading.Thread(target=run, daemon=True).start()

  def _load_more(self) -> None:
    if self._loading_more or (
      self.view_mode == "detail"
      and not getattr(self, "_pending_detail_advance", False)
      and self.selected_index + len(self._detail_view.thumbnails) // 2 < len(self.results)
    ):
      return
    query = self.search_text.strip()
    if query:
      if not self._remaining_search_results:
        return
      results = self._remaining_search_results[: self.search_batch_size]
      del self._remaining_search_results[: self.search_batch_size]
      self._loading_more = True
      self.search_status = "Loading…"
      self._search_input.border_title = "Loading…"
      self._event_queue.put(SearchResultsEvent(self._search_generation, query, results, None, True))
      return

    if self._next_cursor is None:
      return

    self._loading_more = True
    self.search_status = "Loading…"
    self._search_input.border_title = "Loading…"
    gen = self._search_generation
    cancel_event = self._search_cancel_event

    worker = Worker()
    self._workers_tracker.add(worker)

    def run() -> None:
      try:
        self._search_worker("", gen, self._next_cursor, cancel_event)
      finally:
        self._workers_tracker.discard(worker)

    threading.Thread(target=run, daemon=True).start()

  def _search_worker(
    self,
    query: str,
    generation: int,
    after: RClip.ImageCursor | None,
    cancel_event: threading.Event,
  ) -> None:
    try:
      if query and not Model.is_text_query(query):
        raise ValueError("interactive mode supports text queries only")
      with self._search_lock:
        if cancel_event.is_set():
          return
        if query:
          search_results = self.rclip.search(
            query,
            self.working_directory,
            top_k=None,
            cancel_event=cancel_event,
          )
          results = [TuiResult(r.filepath, r.score) for r in search_results]
          next_cursor = None
        else:
          page = self.rclip.list_images(
            self.working_directory,
            self.BROWSE_BATCH_SIZE,
            after=after,
            cancel_event=cancel_event,
          )
          results = [TuiResult(path) for path in page.filepaths]
          next_cursor = page.next_cursor
        if cancel_event.is_set():
          return
    except InterruptedError:
      return
    except Exception as error:
      self._event_queue.put(SearchErrorEvent(generation, query, str(error), after is not None))
    else:
      self._event_queue.put(SearchResultsEvent(generation, query, results, next_cursor, after is not None))

  def _handle_search_results(
    self,
    generation: int,
    query: str,
    results: list[TuiResult],
    next_cursor: RClip.ImageCursor | None,
    append: bool,
  ) -> None:
    if generation != self._search_generation or self.search_text.strip() != query:
      return

    self._loading_more = False
    self.search_status = None
    self._next_cursor = next_cursor

    if not append:
      self._grid.scroll_home(animate=False)
      self._gallery_path.update("")
      self.results = []
      self.selected_index = 0
      self.scroll_row = 0
      if query:
        self._remaining_search_results = results[self.search_batch_size :]
        self.results = results[: self.search_batch_size]
      else:
        self.results = results
      if not self.results:
        self.search_status = "No results"
        self._search_input.border_title = "No results"
      else:
        self._search_input.border_title = ""
    else:
      self.results.extend(results)

    if append and getattr(self, "_pending_detail_advance", False):
      self._pending_detail_advance = False
      if self.results and self.view_mode == "detail":
        self._move_detail(1)

    if self.view_mode == "detail":
      self._on_selection_changed()
    elif append or self._focus_target == "gallery":
      self._on_selection_changed()
    # Trigger preview loading for visible grid cards
    self._grid.update_visible()

  def _handle_search_error(self, generation: int, query: str, message: str, append: bool) -> None:
    if generation != self._search_generation or self.search_text.strip() != query:
      return
    self._loading_more = False
    self._pending_detail_advance = False
    self.search_status = None
    self._search_input.border_title = ""
    self.notify(message, title="Unable to load more images" if append else "Search failed", severity="error")

  def _cleanup_all_placements(self) -> None:
    for kitty_id, placement_id in self._active_placements.values():
      delete_placement(kitty_id, placement_id)
    self._active_placements.clear()

    for kitty_id, placement_id in self._active_detail_placements.values():
      delete_placement(kitty_id, placement_id)
    self._active_detail_placements.clear()

  def process_events(self) -> None:
    processed_any = False
    while True:
      try:
        ev = self._event_queue.get_nowait()
      except queue.Empty:
        break
      processed_any = True
      if isinstance(ev, SearchResultsEvent):
        self._handle_search_results(ev.generation, ev.query, ev.results, ev.next_cursor, ev.append)
      elif isinstance(ev, SearchErrorEvent):
        self._handle_search_error(ev.generation, ev.query, ev.message, ev.append)
    if processed_any:
      self._dirty = True

  # ── Key Handling ─────────────────────────────────────────────────────

  def handle_key(self, key: str) -> None:
    if key in ("ctrl+c", "ctrl+C", "^C", "\x03"):
      self.action_quit()
      return

    if key == "ctrl+f":
      self.action_focus_search()
      return

    if key == "ctrl+o":
      self.action_toggle_view()
      return

    if key == "ctrl+y":
      if self.check_action("copy_image"):
        self.action_copy_image()
      return

    if key == "ctrl+p":
      if self.check_action("copy_path"):
        self.action_copy_path()
      return

    if key == "ctrl+s":
      if self.check_action("download"):
        self.action_download()
      return

    if key in ("ctrl+q", "^Q"):
      return

    if self._focus_target == "search":
      self._handle_key_search(key)
    elif self.view_mode == "detail":
      self._handle_key_detail(key)
    else:
      self._handle_key_gallery(key)

  def _handle_key_search(self, key: str) -> None:
    if key == "down":
      self.action_move_down_or_focus()
      return

    if key == "escape":
      self.action_escape()
      return

    if key == "enter":
      self._search_input.submit()
      return

    if key == "left":
      if not self.search_text:
        self.action_move_down_or_focus()
        self.action_move_left()
        return
      self._search_input.cursor_left()
      return

    if key == "right":
      if not self.search_text:
        self.action_move_down_or_focus()
        self.action_move_right()
        return
      self._search_input.cursor_right()
      return

    if key in ("home", "ctrl+a"):
      self._search_input.cursor_home()
      return

    if key in ("end", "ctrl+e"):
      self._search_input.cursor_end()
      return

    if key == "backspace":
      self._search_input.delete_backward()
      return

    if key in ("delete", "ctrl+d"):
      self._search_input.delete_forward()
      return

    if key == "ctrl+k":
      self._search_input.delete_to_end()
      return

    if key == "ctrl+u":
      self._search_input.clear()
      return

    if key in ("tab", "shift+tab"):
      self.action_move_down_or_focus()
      return

    if key == "up":
      return

    if len(key) == 1 and ord(key) >= 32:
      self._search_input.insert_text(key)

  def _handle_key_gallery(self, key: str) -> None:
    if key in ("h", "left"):
      self.action_move_left()
      return

    if key in ("l", "right"):
      self.action_move_right()
      return

    if key in ("j", "down"):
      self.action_move_down()
      return

    if key == "k":
      self.action_move_up()
      return

    if key == "up":
      self.action_move_up_or_focus()
      return

    if key in ("enter", "o"):
      self.action_open_detail()
      return

    if key == "/":
      self.action_focus_search()
      return

    if key == "y":
      if self.check_action("copy_image"):
        self.action_copy_image()
      return

    if key == "p":
      if self.check_action("copy_path"):
        self.action_copy_path()
      return

    if key == "s":
      if self.check_action("download"):
        self.action_download()
      return

    if key == "escape":
      return

    if key in ("tab", "shift+tab"):
      self.action_focus_search()
      return

  def _handle_key_detail(self, key: str) -> None:
    if key in ("h", "left"):
      self._move_detail(-1)
      return

    if key in ("l", "right"):
      self._move_detail(1)
      return

    if key == "up":
      self.action_focus_search()
      return

    if key in ("escape", "o", "enter"):
      self.action_toggle_view()
      return

    if key == "/":
      self.action_focus_search()
      return

    if key == "y":
      if self.check_action("copy_image"):
        self.action_copy_image()
      return

    if key == "p":
      if self.check_action("copy_path"):
        self.action_copy_path()
      return

    if key == "s":
      if self.check_action("download"):
        self.action_download()
      return

    if key in ("tab", "shift+tab"):
      self.action_focus_search()
      return

  # ── Terminal Rendering ───────────────────────────────────────────────

  def render(self) -> None:
    if not self.is_running or self._term is None:
      return
    w, h = self.get_size()
    buf = StringIO()

    # Clear screen on mode switch or new search query generation
    if (
      self.view_mode != self._last_rendered_mode
      or self._search_generation != self._last_rendered_generation
    ):
      self._cleanup_all_placements()
      buf.write(CLEAR_SCREEN)
      self._last_rendered_mode = self.view_mode
      self._last_rendered_generation = self._search_generation

    # Search Bar (rows 1-3)
    is_search_focused = self._focus_target == "search"
    s_border = f"{FG_CYAN}{BOLD}" if is_search_focused else f"{FG_BRIGHT_BLACK}"

    buf.write(cursor_to(1, 1))
    st = f" [{self.search_status}]" if self.search_status else ""
    title_str = f" rclip search{st} "
    border_len = max(0, w - len(title_str) - 2)
    buf.write(f"{s_border}╭─{RESET}{BOLD}{title_str}{RESET}{s_border}{'─' * border_len}╮{RESET}")

    buf.write(cursor_to(2, 1))
    display_q = self.search_text
    inner_width = max(0, w - 4)
    if is_search_focused:
      if self._search_input._has_selection and display_q:
        rendered_text = f"\x1b[7m{display_q}\x1b[27m"
        text_len = len(display_q)
      else:
        cpos = max(0, min(len(display_q), self._search_input.cursor_position))
        before = display_q[:cpos]
        cursor_char = display_q[cpos] if cpos < len(display_q) else " "
        after = display_q[cpos + 1 :] if cpos < len(display_q) else ""
        rendered_text = f"{before}\x1b[7m{cursor_char}\x1b[27m{after}"
        text_len = len(display_q) if cpos < len(display_q) else len(display_q) + 1
    else:
      rendered_text = display_q
      text_len = len(display_q)

    pad_spaces = " " * max(0, inner_width - text_len)
    buf.write(f"{s_border}│{RESET} {rendered_text}{pad_spaces} {s_border}│{RESET}")

    buf.write(cursor_to(3, 1))
    buf.write(f"{s_border}╰{'─' * max(0, w - 2)}╯{RESET}")

    # Body (rows 4 to h - 2)
    if self.view_mode == "gallery":
      buf.write(self._render_gallery(w, h - 5))
    else:
      buf.write(self._render_detail(w, h - 5))

    # Gallery / Selection Path (row h - 1)
    buf.write(cursor_to(h - 1, 1))
    path_str = self._gallery_path.content[: w - 2]
    buf.write(f"{BOLD}{path_str}{RESET}{' ' * max(0, w - len(path_str))}")

    # Hotkeys Footer (row h) - capped at w - 1 to never trigger terminal edge wrap
    buf.write(cursor_to(h, 1))
    hotkeys_str = self._render_hotkeys(w)
    buf.write(f"{hotkeys_str}{' ' * max(0, w - 1 - len(hotkeys_str))}")

    self._term.write(buf.getvalue())
    self._term.flush()

  def _render_gallery(self, width: int, height: int) -> str:
    buf = StringIO()
    if not self.results:
      buf.write(cursor_to(4 + height // 2, max(1, width // 2 - 5)))
      buf.write(f"{DIM}No results{RESET}")
      return buf.getvalue()

    geo = GridGeometry(len(self.results), avail_x=1, avail_y=4, avail_w=width, avail_h=height)
    self.scroll_row = geo.ensure_visible(self.selected_index, self.scroll_row)
    cards = geo.get_visible_cards(self.scroll_row, self.results)

    visible_paths = {card.result.filepath for card in cards}
    for path, (kid, pid) in list(self._active_placements.items()):
      if path not in visible_paths:
        delete_placement(kid, pid)
        del self._active_placements[path]

    for card in cards:
      is_sel = card.index == self.selected_index
      b_color = f"{FG_CYAN}{BOLD}" if is_sel else f"{FG_BRIGHT_BLACK}"
      inner_w = card.width - 2
      inner_h = card.height - 2

      # Top border
      buf.write(cursor_to(card.y, card.x))
      buf.write(f"{b_color}╭{'─' * inner_w}╮{RESET}")

      cached = self._grid._cached_cards.get(card.result.filepath)
      if cached is None:
        cached = ImageCard(card.result, self)
        self._grid._cached_cards[card.result.filepath] = cached
        cached.load_preview()
      elif not cached._loading and not cached._loaded:
        cached.load_preview()

      img_obj = cached._image.image
      if isinstance(img_obj, PILImage.Image):
        cols, rows, pw, ph = fit_image_to_cells(img_obj.width, img_obj.height, inner_w, inner_h)
        if cols > 0 and rows > 0:
          source_id = _extract_source(img_obj)
          key = (source_id, img_obj.width, img_obj.height, (cols, rows))
          decoded = pw * ph * 4
          transmitted = _IMAGE_CACHE.acquire(
            key, img_obj, decoded, lambda kid: _transmit_image(kid, img_obj, (pw, ph))
          )
          if card.result.filepath in self._active_placements:
            old_kid, old_pid = self._active_placements[card.result.filepath]
            delete_placement(old_kid, old_pid)
          p_id = next_placement_id()
          self._active_placements[card.result.filepath] = (transmitted.kitty_id, p_id)
          create_placement(transmitted.kitty_id, p_id, cols, rows)
          _IMAGE_CACHE.release(transmitted)

          ph_lines = render_kitty_placeholders(transmitted.kitty_id, cols, rows, placement_id=p_id)
          pad_x = (inner_w - cols) // 2
          pad_y = (inner_h - rows) // 2
          for r_off in range(inner_h):
            buf.write(cursor_to(card.y + 1 + r_off, card.x))
            if pad_y <= r_off < pad_y + rows:
              line_str = ph_lines[r_off - pad_y]
              left_pad = " " * pad_x
              right_pad = " " * (inner_w - pad_x - cols)
              buf.write(f"{b_color}│{RESET}{left_pad}{line_str}{right_pad}{b_color}│{RESET}")
            else:
              buf.write(f"{b_color}│{' ' * inner_w}│{RESET}")
        else:
          for r_off in range(inner_h):
            buf.write(cursor_to(card.y + 1 + r_off, card.x))
            buf.write(f"{b_color}│{' ' * inner_w}│{RESET}")
      else:
        loading_text = "Loading…"
        pad_top = inner_h // 2
        for r_off in range(inner_h):
          buf.write(cursor_to(card.y + 1 + r_off, card.x))
          if r_off == pad_top:
            pad_left = (inner_w - len(loading_text)) // 2
            pad_right = inner_w - pad_left - len(loading_text)
            buf.write(f"{b_color}│{RESET}{' ' * pad_left}{DIM}{loading_text}{RESET}{' ' * pad_right}{b_color}│{RESET}")
          else:
            buf.write(f"{b_color}│{' ' * inner_w}│{RESET}")

      # Bottom border
      buf.write(cursor_to(card.y + card.height - 1, card.x))
      buf.write(f"{b_color}╰{'─' * inner_w}╯{RESET}")

    return buf.getvalue()

  def _render_detail(self, width: int, height: int) -> str:
    buf = StringIO()
    geo = DetailGeometry(avail_x=1, avail_y=4, avail_w=width, avail_h=height)

    # Clean up previous detail placements so images do not linger
    for kid, pid in self._active_detail_placements.values():
      delete_placement(kid, pid)
    self._active_detail_placements.clear()

    # Main Detail Frame Border
    f_color = f"{FG_BRIGHT_BLACK}"
    buf.write(cursor_to(geo.main_y, geo.main_x))
    buf.write(f"{f_color}╭{'─' * (geo.main_w - 2)}╮{RESET}")
    for r in range(1, geo.main_h - 1):
      buf.write(cursor_to(geo.main_y + r, geo.main_x))
      buf.write(f"{f_color}│{' ' * (geo.main_w - 2)}│{RESET}")
    buf.write(cursor_to(geo.main_y + geo.main_h - 1, geo.main_x))
    buf.write(f"{f_color}╰{'─' * (geo.main_w - 2)}╯{RESET}")

    filepath = self._selected_filepath()
    if filepath:
      detail_obj = self._detail_view._details.get(filepath)
      if isinstance(detail_obj, PILImage.Image):
        inner_mw = geo.main_w - 2
        inner_mh = geo.main_h - 2
        cols, rows, pw, ph = fit_image_to_cells(detail_obj.width, detail_obj.height, inner_mw, inner_mh)
        img_x = geo.main_x + 1 + (inner_mw - cols) // 2
        img_y = geo.main_y + 1 + (inner_mh - rows) // 2

        source_id = _extract_source(detail_obj)
        key = (source_id, detail_obj.width, detail_obj.height, (cols, rows))
        decoded = pw * ph * 4
        transmitted = _IMAGE_CACHE.acquire(
          key, detail_obj, decoded, lambda kid: _transmit_image(kid, detail_obj, (pw, ph))
        )
        p_id = next_placement_id()
        self._active_detail_placements[filepath] = (transmitted.kitty_id, p_id)
        create_placement(transmitted.kitty_id, p_id, cols, rows)
        _IMAGE_CACHE.release(transmitted)

        ph_lines = render_kitty_placeholders(transmitted.kitty_id, cols, rows, placement_id=p_id)
        for r_idx, ph_line in enumerate(ph_lines):
          buf.write(cursor_to(img_y + r_idx, img_x))
          buf.write(ph_line)
      elif isinstance(detail_obj, str):
        buf.write(cursor_to(geo.main_y + geo.main_h // 2, geo.main_x + 4))
        buf.write(f"{FG_RED}Unable to load image: {detail_obj}{RESET}")
      else:
        buf.write(cursor_to(geo.main_y + geo.main_h // 2, geo.main_x + geo.main_w // 2 - 15))
        buf.write(f"{DIM}Loading higher-resolution image…{RESET}")
    else:
      buf.write(cursor_to(geo.main_y + geo.main_h // 2, geo.main_x + geo.main_w // 2 - 5))
      buf.write(f"{DIM}No results{RESET}")

    # Filmstrip Thumbnails
    thumbs = geo.get_filmstrip_thumbnails(self.selected_index, self.results)
    for thumb in thumbs:
      t_border = f"{FG_CYAN}{BOLD}" if thumb.is_selected else f"{FG_BRIGHT_BLACK}"
      inner_tw = thumb.width - 2
      inner_th = thumb.height - 2
      buf.write(cursor_to(thumb.y, thumb.x))
      buf.write(f"{t_border}╭{'─' * inner_tw}╮{RESET}")

      t_img = None
      if thumb.filepath:
        for dt in self._detail_view.thumbnails:
          if dt.filepath == thumb.filepath and isinstance(dt._image.image, PILImage.Image):
            t_img = dt._image.image
            break
        if t_img is None:
          card = self._grid._cached_cards.get(thumb.filepath)
          if card and isinstance(card._image.image, PILImage.Image):
            t_img = card._image.image

      if isinstance(t_img, PILImage.Image):
        cols, rows, pw, ph = fit_image_to_cells(t_img.width, t_img.height, inner_tw, inner_th)
        if cols > 0 and rows > 0:
          source_id = _extract_source(t_img)
          key = (source_id, t_img.width, t_img.height, (cols, rows), "thumb")
          decoded = pw * ph * 4
          transmitted = _IMAGE_CACHE.acquire(
            key, t_img, decoded, lambda kid: _transmit_image(kid, t_img, (pw, ph))
          )
          p_id = next_placement_id()
          self._active_detail_placements[f"thumb_{thumb.offset}"] = (transmitted.kitty_id, p_id)
          create_placement(transmitted.kitty_id, p_id, cols, rows)
          _IMAGE_CACHE.release(transmitted)

          ph_lines = render_kitty_placeholders(transmitted.kitty_id, cols, rows, placement_id=p_id)
          pad_x = (inner_tw - cols) // 2
          pad_y = (inner_th - rows) // 2
          for r_off in range(inner_th):
            buf.write(cursor_to(thumb.y + 1 + r_off, thumb.x))
            if pad_y <= r_off < pad_y + rows:
              line_str = ph_lines[r_off - pad_y]
              left_pad = " " * pad_x
              right_pad = " " * (inner_tw - pad_x - cols)
              buf.write(f"{t_border}│{RESET}{left_pad}{line_str}{right_pad}{t_border}│{RESET}")
            else:
              buf.write(f"{t_border}│{' ' * inner_tw}│{RESET}")
        else:
          for r_off in range(inner_th):
            buf.write(cursor_to(thumb.y + 1 + r_off, thumb.x))
            buf.write(f"{t_border}│{' ' * inner_tw}│{RESET}")
      else:
        for r_off in range(inner_th):
          buf.write(cursor_to(thumb.y + 1 + r_off, thumb.x))
          buf.write(f"{t_border}│{' ' * inner_tw}│{RESET}")

      buf.write(cursor_to(thumb.y + thumb.height - 1, thumb.x))
      buf.write(f"{t_border}╰{'─' * inner_tw}╯{RESET}")

    return buf.getvalue()

  def _render_hotkeys(self, width: int) -> str:
    keys: list[tuple[str, str]]
    if self._focus_target == "search":
      keys = [("Down", "Browse"), ("^O", "Detail view" if self.view_mode == "gallery" else "Grid view")]
    elif self.view_mode == "detail":
      keys = [("/", "Search"), ("h/l/Arrows", "Browse"), ("Esc", "Grid view")]
    else:
      keys = [("/", "Search"), ("hjkl/Arrows", "Move"), ("o", "Detail view")]

    if self._selected_filepath():
      pfx = "^" if self._focus_target == "search" else ""
      keys.extend([(f"{pfx}y", "Copy"), (f"{pfx}p", "Copy path"), (f"{pfx}s", "Download")])

    keys.append(("^C", "Quit"))

    pills: list[str] = []
    for k, desc in keys:
      pill = f"{BG_BRIGHT_BLACK}{FG_DEFAULT} {k} {RESET} {desc}"
      pills.append(pill)

    return "   ".join(pills)

  # ── Main Run Loop ────────────────────────────────────────────────────

  def run(self) -> None:
    _probe_kitty.cache_clear()
    self.is_running = True
    self._dirty = True
    self._begin_search("")

    simulated_keys = [
      k.strip()
      for k in (os.getenv("TEXTUAL_PRESS") or os.getenv("RCLIP_PRESS") or "").split(",")
      if k.strip()
    ]

    with RawTerminal() as term:
      self._term = term
      last_size = self.get_size()
      while self.is_running:
        cur_size = self.get_size()
        if cur_size != last_size:
          last_size = cur_size
          self._dirty = True
          self._term.write(CLEAR_SCREEN)

        if simulated_keys:
          key = simulated_keys.pop(0)
          self.handle_key(key)
          self._dirty = True
        else:
          try:
            key = read_key(timeout=0.03)
          except KeyboardInterrupt:
            self.action_quit()
            break
          if key:
            self.handle_key(key)
            self._dirty = True

        self.process_events()
        if self._dirty:
          self.render()
          self._dirty = False

    self._cleanup_all_placements()


def run_tui(rclip: RClip, working_directory: str, top_k: int = 100) -> None:
  _probe_kitty.cache_clear()
  RclipApp(rclip, working_directory, top_k).run()
