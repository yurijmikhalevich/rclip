from __future__ import annotations

from asyncio import Lock as AsyncLock
import os
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.timer import Timer
from textual.worker import get_current_worker
from textual.widgets import Input, Static

from rclip.model import Model
from rclip.tui.transfer import copy_image_to_clipboard
from rclip.tui.views import DetailScreen
from rclip.tui.views import ImageCard
from rclip.tui.views import ResultsGrid
from rclip.tui.views import TuiResult
from rclip.utils import helpers

if TYPE_CHECKING:
  from rclip.main import RClip


RESULT_BATCH_SIZE = 100


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
    top_k: int = 100,
  ) -> None:
    super().__init__()
    self.theme = os.getenv("TEXTUAL_THEME", "ansi-dark")
    self.rclip = rclip
    self.working_directory = working_directory
    self.cache_dir = cache_dir
    self.initial_query = initial_query or ""
    self.top_k = top_k
    self._search_timer: Timer | None = None
    self._search_lock = Lock()
    self._search_generation = 0
    self._clipboard_lock = Lock()
    self._mount_lock = AsyncLock()
    self._mount_requested = False
    self._ignore_initial_change = bool(initial_query)
    self._results: list[TuiResult] = []
    self._mounted_results = 0
    self._selected_index = 0

  def compose(self) -> ComposeResult:
    directory = _display_directory(self.working_directory)
    yield Input(value=self.initial_query, placeholder=f"Search images in {directory}…", id="search")
    yield ResultsGrid()
    yield Static(
      "/ Search   hjkl/Arrows Move   Enter View   y Copy image   Y Copy path   q/Ctrl+C Quit",
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
    self._begin_search(query)

  def _begin_search(self, query: str) -> None:
    self._search_generation += 1
    self._search(query, self._search_generation)

  @work(thread=True, group="search", exclusive=True, exit_on_error=False)
  def _search(self, query: str, generation: int) -> None:
    try:
      if query and not Model.is_text_query(query):
        raise ValueError("interactive mode supports text queries only")
      with self._search_lock:
        if query:
          search_results = self.rclip.search(query, self.working_directory, self.top_k)
          results = [TuiResult(result.filepath, result.score) for result in search_results]
        else:
          results = [
            TuiResult(filepath) for filepath in self.rclip.list_images(self.working_directory, self.top_k)
          ]
    except Exception as error:
      self.call_from_thread(self._show_search_error, generation, query, str(error))
    else:
      self.call_from_thread(self._show_results, generation, query, results)

  async def _show_results(self, generation: int, query: str, results: list[TuiResult]) -> None:
    if generation != self._search_generation or self.query_one("#search", Input).value.strip() != query:
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
    if generation != self._search_generation or self.query_one("#search", Input).value.strip() != query:
      return
    self.notify(message, title="Search failed", severity="error")

  def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
    if isinstance(self.focused, Input) and action in {
      "copy_image",
      "copy_path",
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
    if action in {"copy_image", "copy_path"} and isinstance(self.screen, DetailScreen):
      return True
    if action in {
      "copy_image",
      "copy_path",
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


def run_tui(
  rclip: RClip,
  working_directory: str,
  initial_query: str | None,
  top_k: int,
) -> None:
  cache_dir = helpers.get_app_datadir() / "previews"
  RclipApp(rclip, working_directory, cache_dir, initial_query, top_k).run()
