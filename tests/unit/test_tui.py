import asyncio
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
from textual.geometry import Size
from textual.widgets import Input

from rclip import main as main_module
from rclip.main import RClip
from rclip.tui import ClipboardError
from rclip.tui import DetailScreen
from rclip.tui import ImageCard
from rclip.tui import RclipApp
from rclip.tui import StableTGPImage
from rclip.tui import _display_directory
from rclip.tui import cache_image
from rclip.tui import copy_image_to_clipboard
from rclip.utils.helpers import init_arg_parser


class FakeRclip(RClip):
  def __init__(self, results: list[RClip.SearchResult]) -> None:
    self.results = results
    self.searches: list[tuple[str, str, int | None, list[str], list[str]]] = []
    self.browses: list[tuple[str, int | None]] = []

  def search(
    self,
    query: str,
    directory: str,
    top_k: int | None = 10,
    positive_queries: list[str] = [],
    negative_queries: list[str] = [],
  ) -> list[RClip.SearchResult]:
    self.searches.append((query, directory, top_k, positive_queries, negative_queries))
    return self.results if top_k is None else self.results[:top_k]

  def list_images(self, directory: str, top_k: int | None = None) -> list[str]:
    self.browses.append((directory, top_k))
    filepaths = [result.filepath for result in self.results]
    return filepaths if top_k is None else filepaths[:top_k]


def make_image(path: Path, color: str = "red") -> Path:
  Image.new("RGB", (80, 60), color).save(path)
  return path


def test_interactive_cli_accepts_an_optional_query() -> None:
  parser = init_arg_parser()

  args = parser.parse_args(["--interactive"])
  assert args.interactive
  assert args.query is None

  args = parser.parse_args(["-i", "black cat", "--top", "25"])
  assert args.interactive
  assert args.query == "black cat"
  assert args.top == 25

  with pytest.raises(SystemExit):
    parser.parse_args(["--interactive", "--preview"])


def test_interactive_main_allows_database_use_from_search_worker(monkeypatch: pytest.MonkeyPatch) -> None:
  init_options: dict[str, object] = {}
  tui_arguments: list[object] = []
  resources = tuple(SimpleNamespace(close=lambda: None) for _ in range(3))

  def fake_init_rclip(**options: object):
    init_options.update(options)
    return resources

  monkeypatch.setattr(sys, "argv", ["rclip", "--interactive"])
  monkeypatch.setattr(main_module, "is_snap", lambda: False)
  monkeypatch.setattr(main_module, "init_rclip", fake_init_rclip)
  monkeypatch.setattr("rclip.tui.run_tui", lambda *args: tui_arguments.extend(args))

  main_module.main()

  assert init_options["allow_cross_thread_db"] is True
  assert tui_arguments[3] is None


def test_search_placeholder_shortens_the_home_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  home = tmp_path / "home"
  monkeypatch.setattr(Path, "home", lambda: home)

  assert _display_directory(str(home)) == "~"
  assert _display_directory(str(home / "photos")) == str(Path("~") / "photos")
  assert _display_directory(str(tmp_path / "elsewhere")) == str(tmp_path / "elsewhere")


def test_tui_uses_the_terminal_palette(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  monkeypatch.delenv("TEXTUAL_THEME", raising=False)
  assert RclipApp(FakeRclip([]), str(tmp_path), tmp_path / "cache").theme == "ansi-dark"

  monkeypatch.setenv("TEXTUAL_THEME", "ansi-light")
  assert RclipApp(FakeRclip([]), str(tmp_path), tmp_path / "cache").theme == "ansi-light"


def test_cache_image_reuses_a_small_display_image(tmp_path: Path) -> None:
  source = make_image(tmp_path / "source.jpg")

  first = cache_image(str(source), tmp_path / "cache", (32, 32))
  first_mtime = first.stat().st_mtime_ns
  second = cache_image(str(source), tmp_path / "cache", (32, 32))

  assert second == first
  assert second.stat().st_mtime_ns == first_mtime
  with Image.open(second) as preview:
    assert preview.width <= 32
    assert preview.height <= 32


def test_kitty_image_reuses_its_renderable_until_its_size_changes(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  class FakeRenderable:
    def __init__(self, *_args: object) -> None:
      self.cleaned = False

    def cleanup(self) -> None:
      self.cleaned = True

  monkeypatch.setattr(StableTGPImage, "_Renderable", FakeRenderable)
  image = StableTGPImage(make_image(tmp_path / "image.jpg"))

  first = image.render()
  assert isinstance(first, FakeRenderable)
  assert image.render() is first
  assert not first.cleaned

  image._rendered_size = Size(image.content_size.width + 1, image.content_size.height)
  second = image.render()
  assert isinstance(second, FakeRenderable)
  assert second is not first
  assert first.cleaned

  image.on_unmount()
  assert second.cleaned


def test_copy_image_keeps_common_formats_and_converts_others(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  copied: list[tuple[str, str]] = []

  def fake_copy(path: Path) -> None:
    with Image.open(path) as image:
      copied.append((path.suffix, image.format or ""))

  monkeypatch.setattr("rclip.tui._run_clipboard_kitten", fake_copy)
  jpeg = make_image(tmp_path / "image.jpg")
  ppm = make_image(tmp_path / "image.ppm")

  copy_image_to_clipboard(str(jpeg))
  copy_image_to_clipboard(str(ppm))

  assert copied == [(".jpg", "JPEG"), (".png", "PNG")]


def test_clipboard_kitten_failure_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  monkeypatch.setattr("rclip.tui._kitten_executable", lambda: "kitten")
  monkeypatch.setattr(
    subprocess,
    "run",
    lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, stderr="permission denied"),
  )

  with pytest.raises(ClipboardError, match="permission denied"):
    copy_image_to_clipboard(str(make_image(tmp_path / "image.jpg")))


def test_tui_search_navigation_detail_and_copy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  paths = [make_image(tmp_path / f"image-{index}.jpg", color) for index, color in enumerate(("red", "green"))]
  rclip = FakeRclip([RClip.SearchResult(str(path), 0.9 - index / 10) for index, path in enumerate(paths)])
  app = RclipApp(
    rclip,
    str(tmp_path),
    tmp_path / "cache",
    top_k=25,
    positive_queries=["bright"],
    negative_queries=["dark"],
  )
  copied: list[str] = []
  exits: list[bool] = []
  monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
  monkeypatch.setattr(app, "exit", lambda *args, **kwargs: exits.append(True))

  async def run() -> None:
    async with app.run_test(size=(100, 40)) as pilot:
      await pilot.press("c", "a", "t", "enter")
      await pilot.pause()
      await app.workers.wait_for_complete()
      await pilot.pause()

      cards = list(app.query(ImageCard))
      assert len(cards) == 2
      assert app.focused is cards[0]
      assert rclip.searches == [("cat", str(tmp_path), 25, ["bright"], ["dark"])]

      await pilot.press("up")
      assert isinstance(app.focused, Input)
      await pilot.press("up")
      assert isinstance(app.focused, Input)
      await pilot.press("down")
      assert app.focused is cards[0]

      await pilot.press("Y")
      assert copied == [str(paths[0])]

      await pilot.press("right")
      assert app.focused is cards[1]
      await pilot.press("enter")
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert isinstance(app.screen, DetailScreen)
      assert app.screen.filepath == str(paths[1])

      await pilot.press("left")
      await app.workers.wait_for_complete()
      assert app.screen.filepath == str(paths[0])
      await pilot.press("right")
      await app.workers.wait_for_complete()
      assert app.screen.filepath == str(paths[1])

      await pilot.click("#detail-frame", times=2)
      await pilot.pause()
      assert not isinstance(app.screen, DetailScreen)
      assert app.focused is cards[1]

      await pilot.click(cards[0], times=2)
      await pilot.pause()
      assert isinstance(app.screen, DetailScreen)
      await pilot.press("escape")
      await pilot.pause()

      await pilot.press("/")
      assert isinstance(app.focused, Input)
      await pilot.press("q")
      assert app.query_one(Input).value == "q"
      await pilot.press("ctrl+c")
      assert exits == [True]

  asyncio.run(run())


def test_tui_only_loads_visible_previews(tmp_path: Path) -> None:
  path = make_image(tmp_path / "image.jpg")
  rclip = FakeRclip([RClip.SearchResult(str(path), 1 - index / 100) for index in range(100)])
  app = RclipApp(rclip, str(tmp_path), tmp_path / "cache")

  async def run() -> None:
    async with app.run_test(size=(80, 24)) as pilot:
      await pilot.press("x", "enter")
      await pilot.pause()
      await app.workers.wait_for_complete()
      await pilot.pause()

      cards = list(app.query(ImageCard))
      loaded = sum(card._loaded for card in cards)
      assert 0 < loaded < len(cards)

  asyncio.run(run())


def test_empty_query_browses_and_mounts_results_in_batches(tmp_path: Path) -> None:
  path = make_image(tmp_path / "image.jpg")
  rclip = FakeRclip([RClip.SearchResult(str(path), 1 - index / 250) for index in range(250)])
  app = RclipApp(rclip, str(tmp_path), tmp_path / "cache")

  async def run() -> None:
    async with app.run_test(size=(80, 24)) as pilot:
      await app.workers.wait_for_complete()
      await pilot.pause()

      grid = app.query_one("#results")
      assert len(app.query(ImageCard)) == 100
      assert rclip.browses == [(str(tmp_path), None)]

      grid.scroll_end(animate=False)
      await pilot.pause()
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert len(app.query(ImageCard)) == 200

      grid.scroll_end(animate=False)
      await pilot.pause()
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert len(app.query(ImageCard)) == 250

  asyncio.run(run())


def test_interactive_top_limits_empty_browse(tmp_path: Path) -> None:
  path = make_image(tmp_path / "image.jpg")
  rclip = FakeRclip([RClip.SearchResult(str(path), 1 - index / 50) for index in range(50)])
  app = RclipApp(rclip, str(tmp_path), tmp_path / "cache", top_k=25)

  async def run() -> None:
    async with app.run_test() as pilot:
      await app.workers.wait_for_complete()
      await pilot.pause()

      assert len(app.query(ImageCard)) == 25
      assert rclip.browses == [(str(tmp_path), 25)]

  asyncio.run(run())
