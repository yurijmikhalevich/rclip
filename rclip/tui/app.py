from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Sequence

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.timer import Timer
from textual.worker import get_current_worker
from textual.widgets import Input, Static

from rclip.model import Model
from rclip.tui.media import ImageWidget
from rclip.tui.transfer import _is_remote_session
from rclip.tui.transfer import copy_image_to_clipboard
from rclip.tui.transfer import download_image
from rclip.tui.views import DetailView
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


class RclipApp(App[None], inherit_bindings=False):
  TITLE = "rclip"
  ENABLE_COMMAND_PALETTE = False
  CSS_PATH = "app.tcss"
  BROWSE_BATCH_SIZE = 25

  BINDINGS: ClassVar[list[Binding]] = [
    Binding("ctrl+f,/", "focus_search", "Search", show=False),
    Binding("h,left", "move_left", "Left", show=False),
    # Making j or k switch focus would prevent typing that key in the search input.
    Binding("j", "move_down", "Down", show=False),
    Binding("down", "move_down_or_focus", "Down", show=False),
    Binding("k", "move_up", "Up", show=False),
    Binding("up", "move_up_or_focus", "Up", show=False),
    Binding("l,right", "move_right", "Right", show=False),
    Binding("ctrl+o,o", "toggle_view", "Toggle view", show=False),
    Binding("enter", "open_detail", "Open", show=False),
    Binding("escape", "escape", "Search/Browse", show=False),
    Binding("ctrl+y,y", "copy_image", "Copy image", show=False),
    Binding("ctrl+p,Y", "copy_path", "Copy path", show=False),
    Binding("ctrl+s,s", "download", "Download", show=False),
    Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
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
    self._detail = DetailView(self._move_detail)

  def get_css_variables(self) -> dict[str, str]:
    variables = super().get_css_variables()
    if self.current_theme.name == "ansi-dark":
      # Keep borders visible while respecting the terminal's ANSI palette.
      variables["border-blurred"] = "ansi_white"
      variables["border"] = "ansi_bright_green"
    return variables

  def compose(self) -> ComposeResult:
    directory = _display_directory(self.working_directory)
    yield Input(placeholder=f"Search images in {directory}…", id="search")
    with Horizontal(id="gallery-labels"):
      yield Static(f"{directory} · Including subfolders", id="search-scope", markup=False)
      yield Static("All images · Recently modified first", id="results-order", markup=False)
    yield ResultsGrid(self._load_more)
    yield self._detail
    yield Static("", id="gallery-path", markup=False)
    yield Static("", classes="hotkeys", markup=False)

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
    if self._loading_more or (
      self._detail.display
      and not self._pending_detail_advance
      and self._selected_index + len(self._detail.thumbnails) // 2 < len(self._cards())
    ):
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
      self.query_one("#gallery-path", Static).update("")
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
    self.query_one("#results-order", Static).update(
      "Search results · Most similar first" if query else "All images · Recently modified first"
    )
    if append and self._pending_detail_advance:
      self._pending_detail_advance = False
      if cards and self._detail.display:
        self._move_detail(1)
    if not append:
      grid.scroll_home(animate=False)
      self._selected_index = 0
    if self._detail.display:
      self._update_selection()
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
      "move_down",
      "move_left",
      "move_right",
      "move_up",
      "move_up_or_focus",
      "open_detail",
    }:
      return False
    if action in {"copy_image", "copy_path", "download"}:
      return self._selected_card() is not None and (self._detail.display or isinstance(self.focused, ImageCard))
    if action == "open_detail":
      return not self._detail.display and isinstance(self.focused, ImageCard)
    if self._detail.display and action in {
      "move_down",
      "move_up",
    }:
      return False
    if action == "focus_search":
      return not isinstance(self.focused, Input)
    return super().check_action(action, parameters)

  def select_card(self, card: ImageCard) -> None:
    cards = self._cards()
    if card in cards:
      self._selected_index = cards.index(card)
      self._update_selection()
      self.call_after_refresh(self.query_one(ResultsGrid).update_visible)

  def _cards(self) -> Sequence[ImageCard]:
    return self.query_one(ResultsGrid).cards

  def _selected_card(self) -> ImageCard | None:
    cards = self._cards()
    return cards[self._selected_index] if self._selected_index < len(cards) else None

  def _selected_filepath(self) -> str | None:
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
    self._update_selection()

  def _update_selection(self) -> None:
    card = self._selected_card()
    label = ""
    if card is not None:
      result = card.result
      label = result.filepath if result.score is None else f"{result.score:.3f}  {result.filepath}"
    self.query_one("#gallery-path", Static).update(label)
    self._update_hotkeys()
    if not self._detail.display:
      return
    self._detail.show_image(card.result.filepath if card else None)
    if not self._detail.thumbnails:
      return
    cards = self._cards()
    neighbors = len(self._detail.thumbnails) // 2
    self._detail.show_thumbnails([
      cards[index].result.filepath if 0 <= index < len(cards) else None
      for index in range(self._selected_index - neighbors, self._selected_index + neighbors + 1)
    ])
    self._load_more()

  def action_move_left(self) -> None:
    if self._detail.display:
      self._move_detail(-1)
    else:
      self._move(-1)

  def action_move_right(self) -> None:
    if self._detail.display:
      self._move_detail(1)
    else:
      self._move(1)

  def action_move_up(self) -> None:
    self._move(-self._columns())

  def action_move_down(self) -> None:
    self._move(self._columns())

  def action_move_up_or_focus(self) -> None:
    if self._detail.display or (isinstance(self.focused, ImageCard) and self._selected_index < self._columns()):
      self.action_focus_search()
    else:
      self.action_move_up()

  def action_move_down_or_focus(self) -> None:
    if isinstance(self.focused, Input):
      self.action_toggle_focus()
    elif not self._detail.display:
      self.action_move_down()

  def action_focus_search(self) -> None:
    self.query_one("#search", Input).focus()

  def action_toggle_view(self) -> None:
    search_focused = isinstance(self.focused, Input)
    self._pending_detail_advance = False
    grid = self.query_one(ResultsGrid)
    grid.display = self._detail.display
    self._detail.display = not grid.display
    # The terminal may have evicted images while the other view was active.
    for image in self.query(ImageWidget):
      image.refresh_image()
    self._update_selection()
    if search_focused:
      self.action_focus_search()
    elif self._detail.display:
      self._detail.focus()
    else:
      self.call_after_refresh((self._selected_card() or grid).focus)

  def action_toggle_focus(self) -> None:
    self._pending_detail_advance = False
    if not isinstance(self.focused, Input):
      self.action_focus_search()
    else:
      target = self._detail if self._detail.display else self._selected_card() or self.query_one(ResultsGrid)
      target.focus()

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

  def action_open_detail(self) -> None:
    self.action_toggle_view()

  def action_escape(self) -> None:
    if isinstance(self.focused, Input):
      self.action_toggle_focus()
    elif self._detail.display:
      self.action_toggle_view()

  def on_descendant_focus(self, event: events.DescendantFocus) -> None:
    self._update_hotkeys()

  def on_descendant_blur(self, event: events.DescendantBlur) -> None:
    self.call_after_refresh(self._update_hotkeys)

  def _update_hotkeys(self) -> None:
    search_focused = isinstance(self.focused, Input)
    keys = ["Down Browse"] if search_focused else ["/ Search"]
    if search_focused:
      keys.append("^O Grid view" if self._detail.display else "^O Detail view")
    elif self._detail.display:
      keys.append("h/l/Arrows Browse   Esc Grid view")
    else:
      keys.append("hjkl/Arrows Move   o Detail view")
    if self.check_action("copy_image", ()):
      keys.append("^Y Copy   ^P Copy path   ^S Download" if search_focused else "y Copy   Y Copy path   s Download")
    keys.append("^C Quit")
    self.query_one(".hotkeys", Static).update("   ".join(keys))

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
