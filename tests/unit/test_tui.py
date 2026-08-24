import asyncio
import base64
from contextlib import nullcontext
from io import StringIO
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Lock
from types import SimpleNamespace

from PIL import Image
import pytest
from textual.geometry import Size
from textual.widgets import Input

from rclip import main as main_module
from rclip.main import RClip
from rclip.tui.app import RclipApp
from rclip.tui.app import _display_directory
from rclip.tui.media import StableTGPImage
from rclip.tui.media import prepare_image
from rclip.tui.transfer import TransferError
from rclip.tui.transfer import _download_protocol
from rclip.tui.transfer import copy_image_to_clipboard
from rclip.tui.transfer import download_image
from rclip.tui.views import DetailScreen
from rclip.tui.views import ImageCard
from rclip.utils.helpers import init_arg_parser


class FakeRclip(RClip):
  def __init__(self, results: list[RClip.SearchResult]) -> None:
    self.results = results
    self.searches: list[tuple[str, str, int, list[str], list[str]]] = []
    self.browses: list[tuple[str, int]] = []

  def search(
    self,
    query: str,
    directory: str,
    top_k: int = 10,
    positive_queries: list[str] = [],
    negative_queries: list[str] = [],
  ) -> list[RClip.SearchResult]:
    self.searches.append((query, directory, top_k, positive_queries, negative_queries))
    return self.results[:top_k]

  def list_images(self, directory: str, top_k: int) -> list[str]:
    self.browses.append((directory, top_k))
    return [result.filepath for result in self.results[:top_k]]


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


def test_interactive_main_rejects_more_than_100_results_before_setup(monkeypatch: pytest.MonkeyPatch) -> None:
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

  def list_images(_directory: str, _top_k: int) -> list[str]:
    started.set()
    assert release.wait(2)
    return []

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
      assert rclip.searches == [("cat", str(tmp_path), 25, [], [])]

      await pilot.press("down")
      assert app.focused is cards[0]

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
      assert not isinstance(app.screen, DetailScreen)

      await pilot.press("/")
      assert isinstance(app.focused, Input)
      await pilot.press("q")
      assert app.query_one(Input).value == "q"
      await pilot.press("ctrl+c")
      assert exits == [True]

  asyncio.run(run())


def test_tui_only_loads_visible_previews(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  path = make_image(tmp_path / "image.jpg")
  rclip = FakeRclip([RClip.SearchResult(str(path), 1 - index / 100) for index in range(100)])
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
    async with app.run_test(size=(80, 24)):
      assert await asyncio.to_thread(four_started.wait, 1)
      await asyncio.sleep(0.05)
      assert max_active == 4
      release.set()
      await app.workers.wait_for_complete()
      await asyncio.sleep(0)

      assert 0 < len(loaded) < len(app.query(ImageCard))

  asyncio.run(run())


def test_interactive_top_limits_empty_browse(tmp_path: Path) -> None:
  path = make_image(tmp_path / "image.jpg")
  rclip = FakeRclip([RClip.SearchResult(str(path), 1 - index / 50) for index in range(50)])
  app = RclipApp(rclip, str(tmp_path), top_k=25)

  async def run() -> None:
    async with app.run_test() as pilot:
      await app.workers.wait_for_complete()
      await pilot.pause()

      assert len(app.query(ImageCard)) == 25
      assert rclip.browses == [(str(tmp_path), 25)]

  asyncio.run(run())
