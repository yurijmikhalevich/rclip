from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Sequence

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.timer import Timer
from textual.worker import get_current_worker
from textual.widgets import Input, Static

from rclip.model import Model
from rclip.tui.transfer import _is_remote_session
from rclip.tui.transfer import copy_image_to_clipboard
from rclip.tui.transfer import download_image
from rclip.tui.views import DetailScreen
from rclip.tui.views import ImageCard
from rclip.tui.views import ResultsGrid
from rclip.tui.views import TuiResult

if TYPE_CHECKING:
  from rclip.main import RClip


def _display_directory(directory: str) -> str:
  path = Path(directory)
  try:
    relative = path.relative_to(Path.home())
  except ValueError:
    return str(path)
  return str(Path("~") / relative)


class RclipApp(App[None]):
  TITLE = "rclip"
  CSS_PATH = "app.tcss"
  BROWSE_BATCH_SIZE = 25

  BINDINGS: ClassVar[list[Binding]] = [
    Binding("/", "focus_search", "Search", show=False),
    Binding("h,left", "move_left", "Left", show=False),
    # Making j or k switch focus would prevent typing that key in the search input.
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
    top_k: int = 100,
  ) -> None:
    super().__init__()
    self.theme = os.getenv("TEXTUAL_THEME", "ansi-dark")
    self.rclip = rclip
    self.working_directory = working_directory
    self.search_batch_size = top_k
    self._search_timer: Timer | None = None
    self._search_lock = Lock()
    self._search_generation = 0
    self._clipboard_lock = Lock()
    self._selected_index = 0
    self._next_cursor: RClip.ImageCursor | None = None
    self._remaining_search_results: list[TuiResult] = []
    self._loading_more = False
    self._pending_detail_advance = False

  def compose(self) -> ComposeResult:
    directory = _display_directory(self.working_directory)
    yield Input(placeholder=f"Search images in {directory}…", id="search")
    yield ResultsGrid(self._load_more)
    yield Static(
      "/ Search   hjkl/Arrows Move   Enter View   y Copy   Y Copy path   d Download   q/Ctrl+C Quit",
      classes="hotkeys",
      markup=False,
    )

  def on_mount(self) -> None:
    self.query_one("#search", Input).focus()
    self._begin_search("")

  def on_input_changed(self, event: Input.Changed) -> None:
    if event.input.id != "search":
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
    self._begin_search(query)

  def _begin_search(self, query: str) -> None:
    self._search_generation += 1
    self._pending_detail_advance = False
    self._loading_more = False
    self._next_cursor = None
    self._remaining_search_results = []
    self.query_one("#search", Input).border_title = "Searching…"
    self._search(query, self._search_generation, None)

  def _load_more(self) -> None:
    if self._loading_more or (isinstance(self.screen, DetailScreen) and not self._pending_detail_advance):
      return
    query = self.query_one("#search", Input).value.strip()
    if query:
      if not self._remaining_search_results:
        return
      results = self._remaining_search_results[: self.search_batch_size]
      del self._remaining_search_results[: self.search_batch_size]
      self._loading_more = True
      self.query_one("#search", Input).border_title = "Loading…"
      self.call_after_refresh(
        self._show_results,
        self._search_generation,
        query,
        results,
        None,
        True,
      )
      return
    if self._next_cursor is None:
      return
    self._loading_more = True
    self.query_one("#search", Input).border_title = "Loading…"
    self._search("", self._search_generation, self._next_cursor)

  @work(thread=True, group="search", exclusive=True, exit_on_error=False)
  def _search(self, query: str, generation: int, after: RClip.ImageCursor | None) -> None:
    worker = get_current_worker()
    append = after is not None
    next_cursor = None
    try:
      if query and not Model.is_text_query(query):
        raise ValueError("interactive mode supports text queries only")
      with self._search_lock:
        if worker.is_cancelled:
          return
        if query:
          search_results = self.rclip.search(
            query,
            self.working_directory,
            top_k=None,
            cancel_event=worker.cancelled_event,
          )
          results = [TuiResult(result.filepath, result.score) for result in search_results]
        else:
          page = self.rclip.list_images(
            self.working_directory,
            self.BROWSE_BATCH_SIZE,
            after=after,
            cancel_event=worker.cancelled_event,
          )
          results = [TuiResult(filepath) for filepath in page.filepaths]
          next_cursor = page.next_cursor
        if worker.is_cancelled:
          return
    except InterruptedError:
      return
    except Exception as error:
      self.call_from_thread(self._show_search_error, generation, query, str(error), append)
    else:
      self.call_from_thread(self._show_results, generation, query, results, next_cursor, append)

  async def _show_results(
    self,
    generation: int,
    query: str,
    results: list[TuiResult],
    next_cursor: RClip.ImageCursor | None,
    append: bool,
  ) -> None:
    search_input = self.query_one("#search", Input)
    if not self._is_current_search(generation, query):
      return
    grid = self.query_one(ResultsGrid)
    if not append:
      await grid.remove_children()
      if not self._is_current_search(generation, query):
        return
      if query:
        self._remaining_search_results = results[self.search_batch_size :]
        results = results[: self.search_batch_size]
    cards = [ImageCard(result) for result in results]
    if cards:
      await grid.mount(*cards)
    if not self._is_current_search(generation, query):
      return
    self._next_cursor = next_cursor
    self._loading_more = False
    if append and self._pending_detail_advance:
      self._pending_detail_advance = False
      if cards and isinstance(self.screen, DetailScreen):
        self._move_detail(1)
    if not append:
      grid.scroll_home(animate=False)
      self._selected_index = 0
    self.call_after_refresh(grid.update_visible)
    search_input.border_title = None if append or results else "No results"

  def _show_search_error(self, generation: int, query: str, message: str, append: bool) -> None:
    search_input = self.query_one("#search", Input)
    if not self._is_current_search(generation, query):
      return
    self._loading_more = False
    self._pending_detail_advance = False
    search_input.border_title = None
    self.notify(message, title="Unable to load more images" if append else "Search failed", severity="error")

  def _is_current_search(self, generation: int, query: str) -> bool:
    return generation == self._search_generation and self.query_one("#search", Input).value.strip() == query

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
    if isinstance(self.screen, DetailScreen) and action in {
      "focus_search",
      "move_down",
      "move_down_or_focus",
      "move_up",
      "move_up_or_focus",
      "view",
    }:
      return False
    if action == "focus_search":
      return not isinstance(self.focused, Input)
    return super().check_action(action, parameters)

  def select_card(self, card: ImageCard) -> None:
    cards = self._cards()
    if card in cards:
      self._selected_index = cards.index(card)
      self.call_after_refresh(self.query_one(ResultsGrid).update_visible)

  def _cards(self) -> Sequence[ImageCard]:
    return self.query_one(ResultsGrid).cards

  def _selected_card(self) -> ImageCard | None:
    if isinstance(self.focused, ImageCard):
      return self.focused
    cards = self._cards()
    return cards[self._selected_index] if self._selected_index < len(cards) else None

  def _selected_filepath(self) -> str | None:
    if isinstance(self.screen, DetailScreen):
      return self.screen.filepath
    card = self._selected_card()
    return card.result.filepath if card else None

  def _move(self, offset: int) -> None:
    cards = self._cards()
    if not cards:
      return
    index = min(max(self._selected_index + offset, 0), len(cards) - 1)
    self._selected_index = index
    cards[index].focus()

  def _columns(self) -> int:
    cards = self._cards()
    if not cards:
      return 1
    first_row = cards[0].virtual_region.y
    for index in range(1, len(cards)):
      if cards[index].virtual_region.y != first_row:
        return index
    return len(cards)

  def _move_detail(self, offset: int) -> None:
    self._pending_detail_advance = False
    cards = self._cards()
    if not cards:
      return
    if self._selected_index + offset >= len(cards) and (
      self._loading_more or self._next_cursor or self._remaining_search_results
    ):
      self._pending_detail_advance = True
      self._load_more()
      return
    index = min(max(self._selected_index + offset, 0), len(cards) - 1)
    if index == self._selected_index:
      return
    self._selected_index = index
    if isinstance(self.screen, DetailScreen):
      self.screen.show_image(cards[index].result.filepath)

  def action_move_left(self) -> None:
    if isinstance(self.screen, DetailScreen):
      self._move_detail(-1)
    else:
      self._move(-1)

  def action_move_right(self) -> None:
    if isinstance(self.screen, DetailScreen):
      self._move_detail(1)
    else:
      self._move(1)

  def action_move_up(self) -> None:
    self._move(-self._columns())

  def action_move_down(self) -> None:
    self._move(self._columns())

  def action_move_up_or_focus(self) -> None:
    if isinstance(self.focused, ImageCard) and self._selected_index < self._columns():
      self.action_focus_search()
    else:
      self.action_move_up()

  def action_move_down_or_focus(self) -> None:
    if isinstance(self.focused, Input):
      if card := self._selected_card():
        card.focus()
    else:
      self.action_move_down()

  def action_focus_search(self) -> None:
    self.query_one("#search", Input).focus()

  def action_view(self) -> None:
    if card := self._selected_card():
      self.push_screen(DetailScreen(card.result.filepath))

  def action_go_back(self) -> None:
    self._pending_detail_advance = False
    if isinstance(self.screen, DetailScreen):
      selected_index = self._selected_index
      self.pop_screen()
      cards = self._cards()
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
    worker = get_current_worker()
    with self._clipboard_lock:
      if worker.is_cancelled:
        return
      try:
        copy_image_to_clipboard(filepath)
      except Exception as error:
        if not worker.is_cancelled:
          self.call_from_thread(self.notify, str(error), title="Unable to copy image", severity="error")
      else:
        if not worker.is_cancelled:
          self.call_from_thread(self.notify, "Image copied", title=Path(filepath).name)


def run_tui(rclip: RClip, working_directory: str, top_k: int) -> None:
  RclipApp(rclip, working_directory, top_k).run()
