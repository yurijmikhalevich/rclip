import asyncio
import base64
from contextlib import nullcontext
from io import BytesIO, StringIO
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Lock
from types import SimpleNamespace

from PIL import Image
from rich.console import Console
from textual_image.renderable import TGPImage as TGPRenderable
from textual_image._terminal import CellSize
import pytest
from textual.geometry import Size
from textual.widgets import Input, Static

from rclip import main as main_module
from rclip.main import RClip
from rclip.tui.app import RclipApp
from rclip.tui.app import _display_directory
from rclip.tui.media import CenteredTGPImage
from rclip.tui.media import StableTGPImage
from rclip.tui.media import prepare_image
from rclip.tui.transfer import TransferError
from rclip.tui.transfer import _download_protocol
from rclip.tui.transfer import copy_image_to_clipboard
from rclip.tui.transfer import download_image
from rclip.tui.views import DetailScreen
from rclip.tui.views import ImageCard
from rclip.tui.views import ResultsGrid
from rclip.utils.helpers import init_arg_parser


class FakeRclip(RClip):
  def __init__(self, results: list[RClip.SearchResult]) -> None:
    self.results = results
    self.searches: list[tuple[str, str, int | None, list[str], list[str]]] = []
    self.browses: list[tuple[str, int, RClip.ImageCursor | None]] = []

  def search(
    self,
    query: str,
    directory: str,
    top_k: int | None = 10,
    positive_queries: list[str] = [],
    negative_queries: list[str] = [],
    *,
    cancel_event: Event | None = None,
  ) -> list[RClip.SearchResult]:
    self.searches.append((query, directory, top_k, positive_queries, negative_queries))
    return self.results[:top_k]

  def list_images(
    self,
    directory: str,
    limit: int,
    *,
    after: RClip.ImageCursor | None = None,
    cancel_event: Event | None = None,
  ) -> RClip.ImagePage:
    self.browses.append((directory, limit, after))
    start = 0 if after is None else next(
      index + 1 for index, result in enumerate(self.results) if result.filepath == after.filepath
    )
    results = self.results[start : start + limit]
    next_cursor = None
    if start + limit < len(self.results):
      last_index = start + len(results) - 1
      next_cursor = RClip.ImageCursor(float(len(self.results) - last_index), results[-1].filepath)
    return RClip.ImagePage([result.filepath for result in results], next_cursor)


def make_image(path: Path, color: str = "red") -> Path:
  Image.new("RGB", (80, 60), color).save(path)
  return path


def test_interactive_cli_accepts_no_query() -> None:
  parser = init_arg_parser()

  args = parser.parse_args(["--interactive", "--top", "25"])
  assert args.interactive
  assert args.query is None
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
  assert tui_arguments == [resources[0], os.getcwd(), 100]


def test_noninteractive_main_requires_a_query(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(sys, "argv", ["rclip"])
  monkeypatch.setattr(main_module, "init_rclip", lambda **_options: pytest.fail("must reject before setup"))

  with pytest.raises(SystemExit):
    main_module.main()


def test_interactive_main_rejects_initial_query_before_setup(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(sys, "argv", ["rclip", "--interactive", "cat"])
  monkeypatch.setattr(main_module, "init_rclip", lambda **_options: pytest.fail("must reject before setup"))

  with pytest.raises(SystemExit):
    main_module.main()


@pytest.mark.parametrize("option", ["--add", "--subtract"])
def test_interactive_main_rejects_additional_queries_before_setup(option: str, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(sys, "argv", ["rclip", "--interactive", option, "cat"])
  monkeypatch.setattr(main_module, "init_rclip", lambda **_options: pytest.fail("must reject before setup"))

  with pytest.raises(SystemExit):
    main_module.main()


def test_interactive_main_rejects_search_batches_larger_than_100(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(sys, "argv", ["rclip", "--interactive", "--top", "101"])
  monkeypatch.setattr(main_module, "init_rclip", lambda **_options: pytest.fail("must reject before setup"))

  with pytest.raises(SystemExit):
    main_module.main()


def test_search_placeholder_shortens_the_home_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  home = tmp_path / "home"
  monkeypatch.setattr(Path, "home", lambda: home)

  assert _display_directory(str(home)) == "~"
  assert _display_directory(str(home / "photos")) == str(Path("~") / "photos")
  assert _display_directory(str(tmp_path / "elsewhere")) == str(tmp_path / "elsewhere")


def test_tui_uses_the_terminal_palette(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  monkeypatch.delenv("TEXTUAL_THEME", raising=False)
  assert RclipApp(FakeRclip([]), str(tmp_path)).theme == "ansi-dark"

  monkeypatch.setenv("TEXTUAL_THEME", "ansi-light")
  assert RclipApp(FakeRclip([]), str(tmp_path)).theme == "ansi-light"


def test_prepare_image_returns_a_small_jpeg_in_memory(tmp_path: Path) -> None:
  source = make_image(tmp_path / "source.jpg")
  prepared = prepare_image(str(source), (32, 32))

  assert list(tmp_path.iterdir()) == [source]
  with Image.open(prepared) as preview:
    assert preview.format == "JPEG"
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


@pytest.mark.parametrize("size", [(1200, 800), (800, 1200), (800, 800)])
def test_strip_preview_pixels_are_centered_and_keep_aspect_ratio(
  monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
  output = StringIO()
  monkeypatch.setattr(sys, "__stdout__", output)
  monkeypatch.setattr("textual_image.renderable.tgp.get_cell_size", lambda: CellSize(17, 33))
  renderable = CenteredTGPImage._Renderable(Image.new("RGB", size, "red"), 13, 5)
  Console(file=StringIO(), width=13).print(renderable)
  chunks = re.findall(r"\x1b_G[^;]*;([A-Za-z0-9+/=]+)\x1b\\", output.getvalue())
  with Image.open(BytesIO(base64.b64decode("".join(chunks)))) as image:
    assert image.size == (221, 165)
    bounds = image.getbbox()
    assert bounds is not None
    left, top, right, bottom = bounds
    assert abs(left - (image.width - right)) <= 1
    assert abs(top - (image.height - bottom)) <= 1
    assert abs((right - left) - (bottom - top) * size[0] / size[1]) <= 1
    assert left > 0 or top > 0


def test_kitty_image_cleanup_deletes_only_its_own_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  output = StringIO()
  monkeypatch.setattr(sys, "__stdout__", output)
  image = StableTGPImage(make_image(tmp_path / "image.jpg"))
  renderable = image.render()
  assert isinstance(renderable, TGPRenderable)
  Console(file=StringIO(), width=20).print(renderable)
  image_id = renderable.terminal_image_id
  assert image_id is not None
  output.seek(0)
  output.truncate()
  image.on_unmount()
  assert output.getvalue() == f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\"
  image.on_unmount()
  assert output.getvalue() == f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\"


def test_copy_image_keeps_common_formats_and_converts_others(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  copied: list[tuple[str, str]] = []

  def fake_copy(path: Path) -> None:
    with Image.open(path) as image:
      copied.append((path.suffix, image.format or ""))

  monkeypatch.setattr("rclip.tui.transfer._run_clipboard_kitten", fake_copy)
  jpeg = make_image(tmp_path / "image.jpg")
  ppm = make_image(tmp_path / "image.ppm")

  copy_image_to_clipboard(str(jpeg))
  copy_image_to_clipboard(str(ppm))

  assert copied == [(".jpg", "JPEG"), (".png", "PNG")]


def test_clipboard_kitten_failure_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  monkeypatch.setattr("rclip.tui.transfer._kitten_executable", lambda: "kitten")
  monkeypatch.setattr(
    subprocess,
    "run",
    lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, stderr="permission denied"),
  )

  with pytest.raises(TransferError, match="permission denied"):
    copy_image_to_clipboard(str(make_image(tmp_path / "image.jpg")))


def test_latest_clipboard_action_finishes_last(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  first_started = Event()
  release_first = Event()
  copied: list[str] = []
  notifications: list[str] = []

  def copy(filepath: str) -> None:
    copied.append(filepath)
    if filepath == "first":
      first_started.set()
      assert release_first.wait(2)

  app = RclipApp(FakeRclip([]), str(tmp_path))
  monkeypatch.setattr("rclip.tui.app.copy_image_to_clipboard", copy)
  monkeypatch.setattr(app, "notify", lambda message, **_options: notifications.append(message))

  async def run() -> None:
    async with app.run_test():
      await app.workers.wait_for_complete()
      app._copy_image("first")
      assert await asyncio.to_thread(first_started.wait, 1)
      app._copy_image("second")
      await asyncio.sleep(0.05)
      assert copied == ["first"]
      release_first.set()
      await app.workers.wait_for_complete()

  asyncio.run(run())

  assert copied == ["first", "second"]
  assert notifications == ["Image copied"]


def test_tui_rejects_image_query(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  rclip = FakeRclip([])
  app = RclipApp(rclip, str(tmp_path))
  notifications: list[str] = []
  monkeypatch.setattr(app, "notify", lambda message, **_options: notifications.append(message))

  async def run() -> None:
    async with app.run_test() as pilot:
      await app.workers.wait_for_complete()
      app.query_one(Input).value = "2:./cat.jpg"
      await pilot.pause(0.3)
      await app.workers.wait_for_complete()

  asyncio.run(run())

  assert not rclip.searches
  assert notifications == ["interactive mode supports text queries only"]


def test_tui_reports_searching_and_no_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  started = Event()
  release = Event()
  rclip = FakeRclip([])

  def list_images(
    _directory: str,
    _top_k: int,
    *,
    after: RClip.ImageCursor | None = None,
    cancel_event: Event | None = None,
  ) -> RClip.ImagePage:
    assert after is None
    assert cancel_event is not None
    started.set()
    assert release.wait(2)
    return RClip.ImagePage([], None)

  monkeypatch.setattr(rclip, "list_images", list_images)
  app = RclipApp(rclip, str(tmp_path))

  async def run() -> None:
    async with app.run_test():
      assert await asyncio.to_thread(started.wait, 1)
      assert app.query_one(Input).border_title == "Searching…"
      release.set()
      await app.workers.wait_for_complete()
      await asyncio.sleep(0)
      assert app.query_one(Input).border_title == "No results"

  asyncio.run(run())


def test_download_protocol_can_be_detected_or_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
  for name in (
    "KITTY_PUBLIC_KEY",
    "KITTY_WINDOW_ID",
    "LC_TERMINAL",
    "RCLIP_DOWNLOAD_PROTOCOL",
    "TERM",
    "TERM_PROGRAM",
  ):
    monkeypatch.delenv(name, raising=False)

  with pytest.raises(TransferError, match="could not detect"):
    _download_protocol()

  monkeypatch.setenv("TERM", "xterm-kitty")
  assert _download_protocol() == "kitty"

  monkeypatch.setenv("RCLIP_DOWNLOAD_PROTOCOL", "iterm2")
  assert _download_protocol() == "iterm2"

  monkeypatch.setenv("RCLIP_DOWNLOAD_PROTOCOL", "unknown")
  with pytest.raises(TransferError, match="must be `kitty` or `iterm2`"):
    _download_protocol()


def test_download_image_uses_kitty_transfer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  source = make_image(tmp_path / "image.jpg")
  commands: list[tuple[list[str], dict[str, object]]] = []

  def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
    commands.append((command, options))
    return subprocess.CompletedProcess(command, 0, stderr="")

  monkeypatch.setenv("SSH_CONNECTION", "client 1 server 2")
  monkeypatch.setenv("RCLIP_DOWNLOAD_PROTOCOL", "kitty")
  monkeypatch.setattr("rclip.tui.transfer._kitten_executable", lambda: "kitten")
  monkeypatch.setattr(subprocess, "run", run)

  download_image(str(source))

  assert commands == [(["kitten", "transfer", str(source), "Downloads/"], {"stderr": subprocess.PIPE, "text": True})]


def test_download_image_reports_missing_kitten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  monkeypatch.setenv("RCLIP_DOWNLOAD_PROTOCOL", "kitty")
  monkeypatch.delenv("KITTY_INSTALLATION_DIR", raising=False)
  monkeypatch.setattr("rclip.tui.transfer.shutil.which", lambda _: None)

  with pytest.raises(TransferError, match="could not find Kitty"):
    download_image(str(make_image(tmp_path / "image.jpg")))


def test_download_image_streams_the_original_to_iterm2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  source = make_image(tmp_path / "image with spaces.jpg")
  output = StringIO()
  monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
  monkeypatch.setenv("RCLIP_DOWNLOAD_PROTOCOL", "iterm2")
  monkeypatch.setenv("TERM", "xterm-256color")
  monkeypatch.setattr(sys, "stdout", output)

  download_image(str(source))

  sequences = output.getvalue()
  encoded_name = base64.b64encode(source.name.encode()).decode("ascii")
  assert f"1337;MultipartFile=name={encoded_name};size={source.stat().st_size};inline=0\a" in sequences
  assert sequences.endswith("1337;FileEnd\a")
  parts = re.findall(r"1337;FilePart=([A-Za-z0-9+/=]+)\a", sequences)
  assert b"".join(base64.b64decode(part) for part in parts) == source.read_bytes()


def test_tui_search_navigation_detail_and_copy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  paths = [make_image(tmp_path / f"image-{index}.jpg", color) for index, color in enumerate(("red", "green"))]
  rclip = FakeRclip([RClip.SearchResult(str(path), 0.9 - index / 10) for index, path in enumerate(paths)])
  app = RclipApp(
    rclip,
    str(tmp_path),
    top_k=25,
  )
  copied: list[str] = []
  downloaded: list[str] = []
  exits: list[bool] = []
  notifications: list[tuple[str, str | None]] = []
  monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
  monkeypatch.setattr(
    app,
    "notify",
    lambda message, **options: notifications.append((message, options.get("title"))),
  )
  monkeypatch.setattr(app, "suspend", nullcontext)
  monkeypatch.setattr("rclip.tui.app.download_image", downloaded.append)
  monkeypatch.setenv("SSH_CONNECTION", "client 1 server 2")
  monkeypatch.setattr(app, "exit", lambda *args, **kwargs: exits.append(True))

  async def run() -> None:
    async with app.run_test(size=(100, 40)) as pilot:
      await pilot.press("c", "a", "t", "enter")
      await pilot.pause()
      await app.workers.wait_for_complete()
      await pilot.pause()

      cards = list(app.query(ImageCard))
      assert len(cards) == 2
      assert isinstance(app.focused, Input)
      assert rclip.searches == [("cat", str(tmp_path), None, [], [])]
      gallery_path = app.query_one("#gallery-path", Static)
      assert str(gallery_path.content) == ""
      assert gallery_path.region.bottom == app.query_one(".hotkeys").region.y
      assert gallery_path.region.height == 1

      await pilot.press("down")
      assert app.focused is cards[0]
      assert str(gallery_path.content) == str(paths[0])

      await pilot.press("Y")
      assert copied == [str(paths[0])]
      await pilot.press("d")
      assert downloaded == [str(paths[0])]
      monkeypatch.delenv("SSH_CONNECTION")
      await pilot.press("d")
      assert downloaded == [str(paths[0])]
      assert notifications[-1] == (str(paths[0]), "Image is already local")

      await pilot.press("right")
      assert app.focused is cards[1]
      assert str(gallery_path.content) == str(paths[1])
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
      assert str(gallery_path.content) == str(paths[1])

      await pilot.click(cards[0], times=2)
      await pilot.pause()
      assert isinstance(app.screen, DetailScreen)
      await pilot.press("escape")
      await pilot.pause()
      assert not isinstance(app.screen, DetailScreen)

      await pilot.press("/")
      assert isinstance(app.focused, Input)
      await pilot.press("q")
      assert app.query_one(Input).value == "q"
      rclip.results = []
      await pilot.press("enter")
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert not app.query(ImageCard)
      assert str(gallery_path.content) == ""
      await pilot.press("ctrl+c")
      assert exits == [True]

  asyncio.run(run())


def test_gallery_resends_thumbnails_after_detail_browsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(sys, "__stdout__", StringIO())
  paths = [make_image(tmp_path / f"image-{index}.jpg") for index in range(6)]
  app = RclipApp(FakeRclip([RClip.SearchResult(str(path), 1) for path in paths]), str(tmp_path))

  async def run() -> None:
    async with app.run_test(size=(100, 40)) as pilot:
      await app.workers.wait_for_complete()
      await pilot.pause()
      cards = app.query_one(ResultsGrid).cards
      before = [card._image.render() for card in cards]
      assert all(isinstance(renderable, TGPRenderable) for renderable in before)
      await pilot.press("down", "enter", "right", "right", "left", "escape")
      await app.workers.wait_for_complete()
      await pilot.pause()
      for card, previous in zip(cards, before):
        assert isinstance(previous, TGPRenderable)
        assert previous.terminal_image_id is None
        current = card._image.render()
        assert isinstance(current, TGPRenderable)
        assert current is not previous
        assert current.terminal_image_id is not None
      await pilot.press("right", "left")
      assert all(card._image.image is not None for card in cards)

  asyncio.run(run())


def test_detail_loading_and_error_keep_layout_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  path = make_image(tmp_path / "image.jpg")
  missing = tmp_path / "missing.jpg"
  app = RclipApp(FakeRclip([RClip.SearchResult(str(item), 1) for item in [path, missing]]), str(tmp_path))
  started = Event()
  release = Event()

  def slow_prepare(filepath: str, size: tuple[int, int]):
    if size == (1920, 1920):
      started.set()
      assert release.wait(5)
    return prepare_image(filepath, size)

  monkeypatch.setattr("rclip.tui.views.prepare_image", slow_prepare)

  async def run() -> None:
    async with app.run_test(size=(100, 40)) as pilot:
      try:
        await app.workers.wait_for_complete()
        await pilot.press("down", "enter")
        assert await asyncio.to_thread(started.wait, 1)
        screen = app.screen
        assert isinstance(screen, DetailScreen)
        status = screen.query_one("#detail-status", Static)
        frame = screen.query_one("#detail-frame")
        filmstrip = screen.query_one("#detail-filmstrip")
        regions = frame.region, filmstrip.region
        assert status.display and not screen._image.display
        assert frame.region.contains_region(status.region)
      finally:
        release.set()
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert not status.display and screen._image.display
      assert (frame.region, filmstrip.region) == regions
      await pilot.press("right")
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert status.display and not screen._image.display
      assert (frame.region, filmstrip.region) == regions
      assert frame.region.contains_region(status.region)

  asyncio.run(run())


def test_filmstrip_reuses_loaded_neighbors_while_new_thumbnail_loads(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  paths = [str(make_image(tmp_path / f"image-{index}.jpg")) for index in range(7)]
  app = RclipApp(FakeRclip([RClip.SearchResult(path, 1) for path in paths]), str(tmp_path))
  previews: list[str] = []
  release = Event()

  def slow_prepare(filepath: str, size: tuple[int, int]):
    if size == (640, 480):
      previews.append(filepath)
      assert release.wait(5)
    return prepare_image(filepath, size)

  async def run() -> None:
    async with app.run_test(size=(100, 40)) as pilot:
      await app.workers.wait_for_complete()
      await pilot.press("down", "enter", "right", "right")
      await app.workers.wait_for_complete()
      screen = app.screen
      assert isinstance(screen, DetailScreen)
      loaded = {thumbnail.filepath: (thumbnail, thumbnail._image.image) for thumbnail in screen.thumbnails}
      monkeypatch.setattr("rclip.tui.views.prepare_image", slow_prepare)
      try:
        app.action_move_right()
        assert [thumbnail.filepath for thumbnail in screen.thumbnails] == paths[1:6]
        for thumbnail in screen.thumbnails[:-1]:
          previous, image = loaded[thumbnail.filepath]
          assert thumbnail is previous
          assert thumbnail._image.image is image
          assert image is not None
        assert screen.thumbnails[-1]._image.image is None
        await pilot.pause()
        assert previews == [paths[5]]
      finally:
        release.set()
      await app.workers.wait_for_complete()
      assert all(thumbnail._image.image is not None for thumbnail in screen.thumbnails)

  asyncio.run(run())


def test_detail_filmstrip_centers_selection_and_navigates(tmp_path: Path) -> None:
  paths = [str(make_image(tmp_path / f"image-{index}.jpg")) for index in range(5)]
  app = RclipApp(FakeRclip([RClip.SearchResult(path, 1) for path in paths]), str(tmp_path))

  async def run() -> None:
    async with app.run_test(size=(100, 40)) as pilot:
      await app.workers.wait_for_complete()
      await pilot.press("down", "enter")
      await app.workers.wait_for_complete()
      screen = app.screen
      assert isinstance(screen, DetailScreen)
      assert [thumbnail.filepath for thumbnail in screen.thumbnails] == [None, None, *paths[:3]]
      assert all(thumbnail._image.image is not None for thumbnail in screen.thumbnails[2:])
      assert all(not thumbnail.visible for thumbnail in screen.thumbnails[:2])
      assert all(thumbnail.visible for thumbnail in screen.thumbnails[2:])
      await pilot.click(screen.thumbnails[4])
      await app.workers.wait_for_complete()
      assert screen.filepath == paths[2]
      assert [thumbnail.filepath for thumbnail in screen.thumbnails] == paths
      await pilot.pause()
      selected_size = screen.thumbnails[2]._image.content_size
      selected_frame = screen.thumbnails[2].query_one(".detail-thumbnail-frame").region
      for thumbnail in [*screen.thumbnails[:2], *screen.thumbnails[3:]]:
        assert thumbnail._image.content_size.width < selected_size.width
        assert thumbnail._image.content_size.height < selected_size.height
        frame = thumbnail.query_one(".detail-thumbnail-frame").region
        assert 0 < selected_frame.width - frame.width <= 2
        assert selected_frame.height - frame.height == 2
        slot = thumbnail.region
        preview = thumbnail._image.region
        assert frame.y - slot.y == slot.bottom - frame.bottom == 1
        assert abs(preview.x * 2 + preview.width - (frame.x * 2 + frame.width)) <= 1
        assert abs(preview.y * 2 + preview.height - (frame.y * 2 + frame.height)) <= 1
      await pilot.press("right", "right")
      await app.workers.wait_for_complete()
      assert [thumbnail.filepath for thumbnail in screen.thumbnails] == [*paths[2:], None, None]
      assert all(thumbnail._image.image is None for thumbnail in screen.thumbnails[3:])
      assert all(not thumbnail.visible for thumbnail in screen.thumbnails[3:])
      for width, height, count in [(100, 40, 5), (81, 24, 5), (40, 16, 1), (160, 40, 9)]:
        await pilot.resize_terminal(width, height)
        await pilot.pause()
        assert len(screen.thumbnails) == count
        assert [thumbnail.filepath for thumbnail in screen.thumbnails] == [
          paths[index] if 0 <= index < len(paths) else None
          for index in range(4 - count // 2, 5 + count // 2)
        ]
        center = screen.thumbnails[count // 2].region
        assert abs(center.x * 2 + center.width - width) <= 1
        assert screen.query_one("#detail-frame").region.height > 0
      await pilot.press("h", "l", "left")
      await app.workers.wait_for_complete()
      assert screen.filepath == paths[3]
      assert screen.thumbnails[len(screen.thumbnails) // 2].filepath == paths[3]
      await pilot.press("escape")
      assert isinstance(app.focused, ImageCard)
      assert app.focused.result.filepath == paths[3]

  asyncio.run(run())


@pytest.mark.parametrize("query", ["", "cat"])
def test_detail_navigation_loads_more_results(tmp_path: Path, query: str) -> None:
  paths = [str(tmp_path / f"image-{index}.jpg") for index in range(51)]
  rclip = FakeRclip([RClip.SearchResult(path, 1) for path in paths])
  app = RclipApp(rclip, str(tmp_path), top_k=25)

  async def run() -> None:
    async with app.run_test(size=(80, 24)) as pilot:
      await app.workers.wait_for_complete()
      if query:
        await pilot.press(*query, "enter")
        await app.workers.wait_for_complete()
      await pilot.pause()
      assert len(app.query(ImageCard)) == 25
      await pilot.press("down", "enter")
      assert isinstance(app.screen, DetailScreen)
      for index, path in enumerate(paths[1:], 1):
        await pilot.press("right")
        if index == 22:
          await pilot.resize_terminal(160, 24)
        await app.workers.wait_for_complete()
        neighbors = 4 if index >= 22 else 2
        expected = [
          paths[neighbor] if 0 <= neighbor < len(paths) else None
          for neighbor in range(index - neighbors, index + neighbors + 1)
        ]
        async with asyncio.timeout(0.25):
          while app.screen.filepath != path or [thumbnail.filepath for thumbnail in app.screen.thumbnails] != expected:
            await asyncio.sleep(0.01)
      await pilot.press("right")
      assert app.screen.filepath == paths[-1]
      await pilot.press("left")
      assert app.screen.filepath == paths[-2]
      await pilot.press("escape")
      await pilot.pause()
      assert isinstance(app.focused, ImageCard)
      assert app.focused.result.filepath == paths[-2]
      assert [card.result.filepath for card in app.query(ImageCard)] == paths

  asyncio.run(run())
  assert len(rclip.browses) == (1 if query else 3)
  assert rclip.searches == ([(query, str(tmp_path), None, [], [])] if query else [])


@pytest.mark.parametrize("action", ["left", "right", "escape"])
def test_detail_navigation_handles_pending_advance(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
  paths = [str(tmp_path / f"image-{index}.jpg") for index in range(26)]
  rclip = FakeRclip([RClip.SearchResult(path, 1) for path in paths])
  list_images = rclip.list_images
  started = Event()
  release = Event()

  def slow_list_images(
    directory: str, limit: int, *, after: RClip.ImageCursor | None = None, cancel_event: Event | None = None
  ) -> RClip.ImagePage:
    if after is not None:
      started.set()
      assert release.wait(5)
    return list_images(directory, limit, after=after, cancel_event=cancel_event)

  monkeypatch.setattr(rclip, "list_images", slow_list_images)
  app = RclipApp(rclip, str(tmp_path))

  async def run() -> None:
    async with app.run_test(size=(80, 24)) as pilot:
      try:
        await app.workers.wait_for_complete()
        await pilot.press("down", "enter")
        await pilot.press(*(["right"] * 25))
        assert await asyncio.to_thread(started.wait, 1)
        await pilot.press("right", action)
      finally:
        release.set()
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert len(app.query(ImageCard)) == 26
      if action != "escape":
        assert isinstance(app.screen, DetailScreen)
        assert app.screen.filepath == paths[23 if action == "left" else 25]
      else:
        assert not isinstance(app.screen, DetailScreen)
        assert isinstance(app.focused, ImageCard)
        assert app.focused.result.filepath == paths[24]

  asyncio.run(run())
  assert len(rclip.browses) == 2


def test_new_search_interrupts_the_running_search_and_keeps_one_active(tmp_path: Path) -> None:
  class SlowRclip(FakeRclip):
    def __init__(self) -> None:
      super().__init__([])
      self.active = 0
      self.max_active = 0
      self.first_started = Event()
      self.second_started = Event()
      self.first_cancelled = Event()
      self.state_lock = Lock()

    def search(
      self,
      query: str,
      directory: str,
      top_k: int | None = 10,
      positive_queries: list[str] = [],
      negative_queries: list[str] = [],
      *,
      cancel_event: Event | None = None,
    ) -> list[RClip.SearchResult]:
      assert cancel_event is not None
      with self.state_lock:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
      try:
        if query == "first":
          self.first_started.set()
          if cancel_event.wait(2):
            self.first_cancelled.set()
            raise InterruptedError
        else:
          self.second_started.set()
        return []
      finally:
        with self.state_lock:
          self.active -= 1

  rclip = SlowRclip()
  app = RclipApp(rclip, str(tmp_path))

  async def run() -> None:
    async with app.run_test() as pilot:
      await app.workers.wait_for_complete()

      await pilot.press("f", "i", "r", "s", "t")
      await pilot.pause(0.3)
      assert await asyncio.to_thread(rclip.first_started.wait, 1)
      assert app.query_one(Input).border_title == "Searching…"

      await pilot.press("s")
      await pilot.pause(0.3)
      assert await asyncio.to_thread(rclip.second_started.wait, 1)
      await app.workers.wait_for_complete()
      await pilot.pause()

      assert rclip.first_cancelled.is_set()
      assert rclip.max_active == 1
      assert app.query_one(Input).border_title == "No results"

  asyncio.run(run())


def test_new_search_interrupts_loading_more(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  paths = [tmp_path / f"image-{index}.jpg" for index in range(26)]
  rclip = FakeRclip([RClip.SearchResult(str(path), 1 - index / 10) for index, path in enumerate(paths)])
  list_images = rclip.list_images
  page_started = Event()
  page_cancelled = Event()

  def slow_list_images(
    directory: str,
    limit: int,
    *,
    after: RClip.ImageCursor | None = None,
    cancel_event: Event | None = None,
  ) -> RClip.ImagePage:
    if after is None:
      return list_images(directory, limit, cancel_event=cancel_event)
    assert cancel_event is not None
    page_started.set()
    if cancel_event.wait(2):
      page_cancelled.set()
      raise InterruptedError
    raise AssertionError("loading more was not cancelled")

  monkeypatch.setattr(rclip, "list_images", slow_list_images)
  app = RclipApp(rclip, str(tmp_path), top_k=25)

  async def run() -> None:
    async with app.run_test() as pilot:
      await app.workers.wait_for_complete()
      app.query_one(ResultsGrid).scroll_end(animate=False)
      await pilot.pause()
      assert await asyncio.to_thread(page_started.wait, 1)

      search_input = app.query_one(Input)
      search_input.focus()
      search_input.value = "cat"
      await pilot.pause(0.3)
      await app.workers.wait_for_complete()
      await pilot.pause()

      assert page_cancelled.is_set()
      assert rclip.searches == [("cat", str(tmp_path), None, [], [])]
      assert [card.result.filepath for card in app.query(ImageCard)] == [str(path) for path in paths[:25]]

  asyncio.run(run())


def test_tui_only_loads_visible_previews(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  path = make_image(tmp_path / "image.jpg")
  filepaths = [str(tmp_path / f"image-{index}.jpg") for index in range(100)]
  rclip = FakeRclip([RClip.SearchResult(filepath, 1 - index / 100) for index, filepath in enumerate(filepaths)])
  app = RclipApp(rclip, str(tmp_path))
  loaded: list[str] = []
  lock = Lock()
  release = Event()
  four_started = Event()
  active = 0
  max_active = 0

  def prepare(filepath: str, _size: tuple[int, int]) -> Path:
    nonlocal active, max_active
    with lock:
      loaded.append(filepath)
      active += 1
      max_active = max(max_active, active)
      if active == 4:
        four_started.set()
    try:
      assert release.wait(2)
      return path
    finally:
      with lock:
        active -= 1

  monkeypatch.setattr("rclip.tui.views.prepare_image", prepare)

  async def run() -> None:
    async with app.run_test(size=(80, 24)) as pilot:
      assert await asyncio.to_thread(four_started.wait, 1)
      await asyncio.sleep(0.05)
      assert max_active == 4
      release.set()
      await app.workers.wait_for_complete()
      await asyncio.sleep(0)

      assert 0 < len(loaded) < len(app.query(ImageCard))

      grid = app.query_one(ResultsGrid)
      grid.scroll_end(animate=False)
      await pilot.pause()
      grid.update_visible()
      await pilot.pause()
      await app.workers.wait_for_complete()
      assert filepaths[24] in loaded

  asyncio.run(run())


def test_empty_browse_uses_bounded_batches(tmp_path: Path) -> None:
  results = [RClip.SearchResult(str(tmp_path / f"image-{index}.jpg"), 1 - index / 50) for index in range(50)]
  rclip = FakeRclip(results)
  app = RclipApp(rclip, str(tmp_path), top_k=10)

  async def run() -> None:
    async with app.run_test() as pilot:
      await app.workers.wait_for_complete()
      await pilot.pause()

      assert len(app.query(ImageCard)) == 25
      assert rclip.browses == [(str(tmp_path), 25, None)]

  asyncio.run(run())


def test_empty_browse_loads_more_as_keyboard_moves_down(tmp_path: Path) -> None:
  paths = [tmp_path / f"image-{index}.jpg" for index in range(26)]
  rclip = FakeRclip([RClip.SearchResult(str(path), 1 - index / 10) for index, path in enumerate(paths)])
  app = RclipApp(rclip, str(tmp_path), top_k=25)

  async def run() -> None:
    async with app.run_test(size=(80, 24)) as pilot:
      await app.workers.wait_for_complete()
      await pilot.pause()

      for _ in range(10):
        await pilot.press("down")
      await app.workers.wait_for_complete()
      await pilot.pause()

      assert [card.result.filepath for card in app.query(ImageCard)] == [str(path) for path in paths]

  asyncio.run(run())

  cursor = RClip.ImageCursor(2, str(paths[24]))
  assert rclip.browses == [
    (str(tmp_path), 25, None),
    (str(tmp_path), 25, cursor),
  ]


def test_search_loads_more_on_scroll(tmp_path: Path) -> None:
  paths = [tmp_path / f"image-{index}.jpg" for index in range(26)]
  rclip = FakeRclip([RClip.SearchResult(str(path), 1 - index / 10) for index, path in enumerate(paths)])
  app = RclipApp(rclip, str(tmp_path), top_k=25)

  async def run() -> None:
    async with app.run_test(size=(80, 24)) as pilot:
      await app.workers.wait_for_complete()

      search_input = app.query_one(Input)
      search_input.value = "cat"
      await pilot.pause(0.3)
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert [card.result.filepath for card in app.query(ImageCard)] == [str(path) for path in paths[:25]]

      await pilot.press("down")
      gallery_path = app.query_one("#gallery-path", Static)
      assert str(gallery_path.content) == str(paths[0])
      app.query_one(ResultsGrid).scroll_end(animate=False)
      async with asyncio.timeout(0.25):
        while len(app.query(ImageCard)) != 26:
          await asyncio.sleep(0.01)
      assert [card.result.filepath for card in app.query(ImageCard)] == [str(path) for path in paths]
      assert str(gallery_path.content) == str(paths[0])

  asyncio.run(run())

  assert rclip.searches == [("cat", str(tmp_path), None, [], [])]


@pytest.mark.parametrize("detail", [False, True])
def test_empty_browse_retries_loading_more_after_failure(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detail: bool
) -> None:
  paths = [tmp_path / f"image-{index}.jpg" for index in range(26)]
  rclip = FakeRclip([RClip.SearchResult(str(path), 1 - index / 10) for index, path in enumerate(paths)])
  app = RclipApp(rclip, str(tmp_path), top_k=25)
  list_images = rclip.list_images
  failed = False
  notifications: list[tuple[str, str | None]] = []

  def fail_second_page_once(
    directory: str,
    limit: int,
    *,
    after: RClip.ImageCursor | None = None,
    cancel_event: Event | None = None,
  ) -> RClip.ImagePage:
    nonlocal failed
    if after is not None and not failed:
      failed = True
      raise RuntimeError("page failed")
    return list_images(directory, limit, after=after, cancel_event=cancel_event)

  monkeypatch.setattr(rclip, "list_images", fail_second_page_once)
  monkeypatch.setattr(
    app,
    "notify",
    lambda message, **options: notifications.append((message, options.get("title"))),
  )

  async def run() -> None:
    async with app.run_test() as pilot:
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert [card.result.filepath for card in app.query(ImageCard)] == [str(path) for path in paths[:25]]

      grid = app.query_one(ResultsGrid)
      if detail:
        await pilot.press("down", "enter", *(["right"] * 23))
      else:
        grid.scroll_end(animate=False)
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert [card.result.filepath for card in app.query(ImageCard)] == [str(path) for path in paths[:25]]
      assert notifications == [("page failed", "Unable to load more images")]

      if detail:
        assert isinstance(app.screen, DetailScreen)
        assert app.screen.filepath == str(paths[23])
        await pilot.press("right")
      else:
        grid.scroll_home(animate=False)
        await pilot.pause()
        grid.scroll_end(animate=False)
      await app.workers.wait_for_complete()
      await pilot.pause()
      assert [card.result.filepath for card in app.query(ImageCard)] == [str(path) for path in paths]
      if detail:
        assert isinstance(app.screen, DetailScreen)
        assert app.screen.filepath == str(paths[24])

  asyncio.run(run())

  cursor = RClip.ImageCursor(2, str(paths[24]))
  assert rclip.browses == [
    (str(tmp_path), 25, None),
    (str(tmp_path), 25, cursor),
  ]
