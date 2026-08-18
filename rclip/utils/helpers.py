import argparse
import contextlib
import hashlib
import io
import os
import pathlib
import textwrap
import warnings
from typing import Optional, Union, cast
from PIL import Image, UnidentifiedImageError
import re
import numpy as np
import sys
from importlib.metadata import version
from urllib.parse import urlparse

from rclip.const import IMAGE_RAW_EXT, IS_LINUX, IS_MACOS, IS_WINDOWS
from rclip.utils import preprocess


MAX_DOWNLOAD_SIZE_BYTES = 50_000_000
DOWNLOAD_TIMEOUT_SECONDS = 60
WIN_ABSOLUTE_FILE_PATH_REGEX = re.compile(r"^[a-z]:\\", re.I)
DEFAULT_TERMINAL_TEXT_WIDTH = 100

# PIL's historic decompression-bomb limit; we never set our cap below this so that
# ordinary large photos (high-MP panoramas, scans) always index.
MIN_MAX_IMAGE_PIXELS = 89_478_485
# the memory-aware default caps a single decoded image at this fraction of total RAM,
# divided across the concurrent loader threads so that several big images can decode at once.
_MAX_IMAGE_PIXELS_MEMORY_FRACTION = 0.5
# a decoded pixel costs ~3 bytes (RGB), but decoding and our preprocessing keep extra copies
# around, so we budget conservatively.
_DECODED_BYTES_PER_PIXEL = 4

# "auto" means "compute the memory-aware default"; an int is an explicit cap; None disables the
# cap entirely (for power users indexing a trusted tree of gigapixel images).
AUTO_MAX_IMAGE_PIXELS = "auto"
MaxImagePixels = Union[str, int, None]

_image_loading_configured = False
_max_image_pixels: Optional[int] = MIN_MAX_IMAGE_PIXELS


class ImageTooLargeError(Exception):
  """Raised when an image's pixel count exceeds the configured limit, before it gets decoded."""

  def __init__(self, path: str, pixels: int, limit: int):
    self.path = path
    self.pixels = pixels
    self.limit = limit
    super().__init__(f"{path} has {pixels} pixels, which exceeds the limit of {limit} pixels")


def _get_total_memory_bytes() -> Optional[int]:
  """Best-effort total physical RAM, cross-platform; returns None if it can't be determined."""
  try:
    if IS_WINDOWS:
      import ctypes

      class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
          ("dwLength", ctypes.c_ulong),
          ("dwMemoryLoad", ctypes.c_ulong),
          ("ullTotalPhys", ctypes.c_ulonglong),
          ("ullAvailPhys", ctypes.c_ulonglong),
          ("ullTotalPageFile", ctypes.c_ulonglong),
          ("ullAvailPageFile", ctypes.c_ulonglong),
          ("ullTotalVirtual", ctypes.c_ulonglong),
          ("ullAvailVirtual", ctypes.c_ulonglong),
          ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

      stat = MEMORYSTATUSEX()
      stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
      # windll only exists on Windows; getattr keeps the type checker happy on other platforms.
      # GlobalMemoryStatusEx returns 0 on failure, in which case stat is left uninitialized.
      if not getattr(ctypes, "windll").kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
      return int(stat.ullTotalPhys)
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
  except (ValueError, OSError, AttributeError):
    return None


def compute_default_max_image_pixels(workers: int) -> int:
  """Picks a decompression-bomb cap from the machine's RAM so we attempt only images we can
  reasonably decode, never going below MIN_MAX_IMAGE_PIXELS so normal large photos still index."""
  total = _get_total_memory_bytes()
  if not total:
    return MIN_MAX_IMAGE_PIXELS
  estimate = int(total * _MAX_IMAGE_PIXELS_MEMORY_FRACTION / (max(1, workers) * _DECODED_BYTES_PER_PIXEL))
  return max(MIN_MAX_IMAGE_PIXELS, estimate)


def get_max_image_pixels() -> Optional[int]:
  return _max_image_pixels


def configure_max_image_pixels(value: MaxImagePixels, workers: int) -> None:
  """Resolves and applies the image pixel cap. ``value`` is "auto" (memory-aware default),
  an explicit int, or None to disable the cap. Also mirrors it onto PIL as a defense-in-depth
  backstop for formats that don't expose their size before decoding."""
  global _max_image_pixels
  if value == AUTO_MAX_IMAGE_PIXELS:
    _max_image_pixels = compute_default_max_image_pixels(workers)
  else:
    _max_image_pixels = cast(Optional[int], value)
  Image.MAX_IMAGE_PIXELS = _max_image_pixels


def _ensure_image_loading_configured() -> None:
  """Configure Pillow's safety and compatibility settings once per process."""
  global _image_loading_configured
  if _image_loading_configured:
    return
  from PIL import ImageFile

  setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
  # PIL only *warns* at MAX_IMAGE_PIXELS and *raises* at 2x; turn the warning into an error so we
  # get a single, clean threshold (the cap) with no raw PIL warning text leaking to the console.
  warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
  # apply the cap onto PIL even if configure_max_image_pixels was never called (e.g. URL queries)
  Image.MAX_IMAGE_PIXELS = _max_image_pixels
  _image_loading_configured = True


def _native_heif_path(path: str) -> Image.Image:
  from rclip.utils import native_heif

  try:
    return native_heif.decode_path(path, Image.MAX_IMAGE_PIXELS)
  except native_heif.NativeHeifTooLargeError as error:
    raise ImageTooLargeError(path, error.pixels, error.limit) from error
  except native_heif.NativeHeifError as error:
    unidentified = UnidentifiedImageError(str(error))
    unidentified.filename = path
    raise unidentified from error


def _native_heif_bytes(data: bytes, source: str) -> Image.Image:
  from rclip.utils import native_heif

  try:
    return native_heif.decode_bytes(data, Image.MAX_IMAGE_PIXELS)
  except native_heif.NativeHeifTooLargeError as error:
    raise ImageTooLargeError(source, error.pixels, error.limit) from error
  except native_heif.NativeHeifError as error:
    unidentified = UnidentifiedImageError(str(error))
    unidentified.filename = source
    raise unidentified from error


def __get_system_datadir() -> pathlib.Path:
  """
  Returns a parent directory path
  where persistent application data can be stored.

  - linux: ~/.local/share
  - macOS: ~/Library/Application Support
  - windows: C:/Users/<USER>/AppData/Roaming
  """

  home = pathlib.Path.home()

  if IS_WINDOWS:
    return home / "AppData/Roaming"
  elif IS_LINUX:
    return home / ".local/share"
  elif IS_MACOS:
    return home / "Library/Application Support"

  raise NotImplementedError(f'"{sys.platform}" is not supported')


def get_app_datadir() -> pathlib.Path:
  app_datadir = os.getenv("RCLIP_DATADIR")
  if app_datadir:
    app_datadir = pathlib.Path(app_datadir)
  else:
    app_datadir = __get_system_datadir() / "rclip"
  os.makedirs(app_datadir, exist_ok=True)
  return app_datadir


def get_model_cache_dir() -> pathlib.Path | None:
  model_cache_dir = os.getenv("RCLIP_MODEL_CACHE_DIR")
  if not model_cache_dir:
    return None

  model_cache_path = pathlib.Path(model_cache_dir)
  os.makedirs(model_cache_path, exist_ok=True)
  return model_cache_path


def positive_int_arg_type(arg: str) -> int:
  arg_int = int(arg)
  if arg_int < 1:
    raise argparse.ArgumentTypeError("should be >0")
  return arg_int


def max_image_megapixels_arg_type(arg: str) -> MaxImagePixels:
  """Parses the --max-image-megapixels value (in megapixels) into an internal pixel count;
  returns AUTO_MAX_IMAGE_PIXELS for "auto" and None for the "disable the limit" keywords.
  argparse also runs this on the string default, so "auto" must pass through cleanly."""
  if arg.lower() == AUTO_MAX_IMAGE_PIXELS:
    return AUTO_MAX_IMAGE_PIXELS
  if arg.lower() in ("none", "off", "disable", "disabled", "0"):
    return None
  try:
    megapixels = float(arg)
  except ValueError:
    raise argparse.ArgumentTypeError('should be a number, "auto", or "none" to disable the limit')
  if megapixels <= 0:
    raise argparse.ArgumentTypeError('should be >0, "auto", or "none" to disable the limit')
  return round(megapixels * 1_000_000)


def get_terminal_text_width() -> int:
  try:
    computed_width = min(DEFAULT_TERMINAL_TEXT_WIDTH, os.get_terminal_size().columns - 2)
    if computed_width < 20:
      return DEFAULT_TERMINAL_TEXT_WIDTH
    return computed_width
  except OSError:
    return DEFAULT_TERMINAL_TEXT_WIDTH


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
  def __init__(self, prog: str, indent_increment: int = 2, max_help_position: int = 24) -> None:
    text_width = get_terminal_text_width()
    super().__init__(prog, indent_increment, max_help_position, width=text_width)


def init_arg_parser() -> argparse.ArgumentParser:
  text_width = get_terminal_text_width()
  parser = argparse.ArgumentParser(
    formatter_class=HelpFormatter,
    prefix_chars="-+",
    description="rclip is a semantic photo search tool for the command line",
    epilog="hints:\n"
    + textwrap.fill(
      '- relative file path should be prefixed with ./, e.g. "./cat.jpg", not "cat.jpg"',
      initial_indent="  ",
      subsequent_indent="    ",
      width=text_width,
    )
    + "\n"
    + textwrap.fill(
      '- any query can be prefixed with a multiplier, e.g. "2:cat", "0.5:./cat-sleeps-on-a-chair.jpg";'
      " adding a multiplier is especially useful when combining image and text queries because"
      " image queries are usually weighted more than text ones",
      initial_indent="  ",
      subsequent_indent="    ",
      width=text_width,
    )
    + "\n\n"
    "get help:\n"
    "  https://github.com/yurijmikhalevich/rclip/discussions/new/choose\n\n",
  )
  version_str = f"rclip {version('rclip')}"
  parser.add_argument("--version", "-v", action="version", version=version_str, help=f'prints "{version_str}"')
  parser.add_argument(
    "query",
    nargs="?",
    help="a text query or a path/URL to an image file; optional with --interactive",
  )
  parser.add_argument(
    "--add",
    "-a",
    "+",
    metavar="QUERY",
    action="append",
    default=[],
    help='a text query or a path/URL to an image file to add to the "original" query, can be used multiple times',
  )
  parser.add_argument(
    "--subtract",
    "--sub",
    "-s",
    "-",
    metavar="QUERY",
    action="append",
    default=[],
    help='a text query or a path/URL to an image file to subtract from the "original" query,'
    " can be used multiple times",
  )
  parser.add_argument(
    "--top",
    "-t",
    type=positive_int_arg_type,
    help="number of top results to display; default: 10, or unlimited with --interactive",
  )
  display_mode_group = parser.add_mutually_exclusive_group()
  display_mode_group.add_argument(
    "--interactive",
    "-i",
    action="store_true",
    default=False,
    help="opens the interactive terminal UI; the query is optional",
  )
  display_mode_group.add_argument(
    "--preview",
    "-p",
    action="store_true",
    default=False,
    help="preview results in the terminal (supported in iTerm2, Konsole 22.04+, wezterm, Mintty, mlterm)",
  )
  display_mode_group.add_argument(
    "--filepath-only",
    "-f",
    action="store_true",
    default=False,
    help="outputs only filepaths",
  )
  parser.add_argument(
    "--preview-height",
    "-H",
    metavar="PREVIEW_HEIGHT_PX",
    action="store",
    type=int,
    default=400,
    help="preview height in pixels; default: 400",
  )
  parser.add_argument(
    "--no-indexing",
    "--skip-index",
    "--skip-indexing",
    "-n",
    action="store_true",
    default=False,
    help="allows to skip updating the index if no images were added, changed, or removed",
  )
  parser.add_argument(
    "--indexing-batch-size",
    "-b",
    type=positive_int_arg_type,
    default=8,
    help="the size of the image batch used when updating the search index;"
    " larger values may improve the indexing speed a bit on some hardware but will increase RAM usage; default: 8",
  )
  parser.add_argument(
    "--exclude-dir",
    action="append",
    help="dir to exclude from search, can be used multiple times;"
    ' adding this argument overrides the default of'
    ' ("@eaDir", "node_modules", ".git", "System Volume Information")',
  )
  parser.add_argument(
    "--include-hidden",
    action="store_true",
    default=False,
    help="index dot-prefixed hidden files and directories"
    " (e.g. .DS_Store, ._IMG_1234.JPG, .Spotlight-V100);"
    " by default they are skipped since they are usually OS metadata rather than user files",
  )
  parser.add_argument(
    "--experimental-raw-support",
    action="store_true",
    default=False,
    help="enables support for RAW images (ARW, CR2, and DNG are supported)",
  )
  parser.add_argument(
    "--max-image-megapixels",
    metavar="MP",
    type=max_image_megapixels_arg_type,
    default=AUTO_MAX_IMAGE_PIXELS,
    help="maximum size, in megapixels, an image may have to be indexed; larger images are skipped to"
    " avoid running out of memory on huge or maliciously crafted images;"
    ' pass "none" to disable the limit; default: chosen automatically based on available memory',
  )
  return parser


# See: https://meta.wikimedia.org/wiki/User-Agent_policy
def download_image(url: str, *, trusted: bool = False) -> Image.Image:
  import requests

  _ensure_image_loading_configured()
  headers = {"User-agent": "rclip - (https://github.com/yurijmikhalevich/rclip)"}
  check_size = requests.request("HEAD", url, headers=headers, timeout=60)
  if length := check_size.headers.get("Content-Length"):
    if int(length) > MAX_DOWNLOAD_SIZE_BYTES:
      raise ValueError(f"Avoiding download of large ({length} byte) file.")
  # a query image URL is chosen by the user, so bypass the cap when trusted (see read_image)
  limit_ctx = image_pixel_limit_disabled() if trusted else contextlib.nullcontext()
  response = requests.get(url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS)
  try:
    with limit_ctx:
      data = response.content
      try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
      except UnidentifiedImageError:
        # WIC and Image I/O accept filenames. If Pillow could not identify a URL response, retry
        # through the native codec using the bytes that requests has buffered.
        content_type = response.headers.get("Content-Type", "").partition(";")[0].lower()
        url_extension = get_file_extension(urlparse(url).path)
        if sys.platform not in ("darwin", "win32") or (
          url_extension != "heic" and content_type not in {"image/heic", "image/heif"}
        ):
          raise
        return _native_heif_bytes(data, url)
  finally:
    response.close()


def get_file_extension(path: str) -> str:
  return os.path.splitext(path)[1].lower()[1:]


def read_raw_image_file(path: str):
  import rawpy

  with rawpy.imread(path) as raw:
    # The embedded JPEG preview is usually large enough for the 256px model
    # input and far cheaper to decode than demosaicing the whole sensor.
    # Fall back to postprocess otherwise.
    try:
      thumb = raw.extract_thumb()
      # rawpy re-exports these from its compiled extension via
      # globals().update, so ty can't see them.
      if thumb.format == rawpy.ThumbFormat.JPEG:  # ty: ignore[unresolved-attribute]
        image = Image.open(io.BytesIO(thumb.data))
        if min(image.size) >= preprocess.IMAGE_SIZE:
          return image
    except rawpy.LibRawNoThumbnailError:  # ty: ignore[unresolved-attribute]
      pass
    rgb = raw.postprocess(half_size=True)
  return Image.fromarray(np.array(rgb))


def compute_file_hash(path: str) -> str:
  """Compute SHA-256 fingerprint of first 1MB + last 1KB of a file for rename/duplicate detection.

  This is O(1) with respect to file size.
  """
  hasher = hashlib.sha256()
  with open(path, "rb") as f:
    hasher.update(f.read(1024 * 1024))
    file_size = f.seek(0, 2)
    if file_size > 1024:
      f.seek(-1024, 2)
      hasher.update(f.read(1024))
  return hasher.hexdigest()


_BOMB_PIXELS_RE = re.compile(r"Image size \((\d+) pixels\)")


def parse_bomb_pixels(error: Exception) -> int:
  """Pulls the pixel count out of PIL's decompression-bomb message; 0 if it can't be parsed."""
  match = _BOMB_PIXELS_RE.search(str(error))
  return int(match.group(1)) if match else 0


@contextlib.contextmanager
def image_pixel_limit_disabled():
  """Temporarily lifts the decompression-bomb cap. Used for images the user explicitly chose as a
  query: they are trusted and decoded one at a time, so the indexing memory cap doesn't apply.
  Only safe on the single-threaded query path, since Image.MAX_IMAGE_PIXELS is process-global."""
  previous = Image.MAX_IMAGE_PIXELS
  Image.MAX_IMAGE_PIXELS = None
  try:
    yield
  finally:
    Image.MAX_IMAGE_PIXELS = previous


def read_image(query: str, *, trusted: bool = False) -> Image.Image:
  _ensure_image_loading_configured()
  path = str.removeprefix(query, "file://")
  # an explicit query image is chosen by the user and decoded one at a time, so the indexing memory
  # cap doesn't apply; read it as trusted and bypass the cap so any size the user points at works.
  limit_ctx = image_pixel_limit_disabled() if trusted else contextlib.nullcontext()
  try:
    with limit_ctx:
      file_ext = get_file_extension(path)
      if file_ext in IMAGE_RAW_EXT:
        image = read_raw_image_file(path)
      elif file_ext == "heic" and sys.platform in ("darwin", "win32"):
        image = _native_heif_path(path)
      else:
        # Image.open only reads the header and runs PIL's decompression-bomb check there, so oversized
        # images are rejected before we ever decode and allocate memory for them.
        image = Image.open(path)
  except UnidentifiedImageError as e:
    # by default the filename on the UnidentifiedImageError is None
    e.filename = path
    raise e
  except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
    # we turn PIL's warning into an error too (see _ensure_image_loading_configured), so this catches
    # both PIL tiers; re-raise as our own type so callers get a clean, friendly message.
    raise ImageTooLargeError(path, parse_bomb_pixels(e), _max_image_pixels or 0) from e
  return image


def is_http_url(path: str) -> bool:
  return path.startswith("https://") or path.startswith("http://")


def is_file_path(path: str) -> bool:
  return (
    path.startswith("/")
    or path.startswith("file://")
    or path.startswith("./")
    or WIN_ABSOLUTE_FILE_PATH_REGEX.match(path) is not None
  )
