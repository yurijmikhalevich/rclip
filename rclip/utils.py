import argparse
import os
import pathlib
from PIL import Image, UnidentifiedImageError
import re
import requests
import sys


MAX_DOWNLOAD_SIZE_BYTES = 50_000_000
DOWNLOAD_TIMEOUT_SECONDS = 60
WIN_ABSOLUTE_FILE_PATH_REGEX = re.compile(r'^[a-z]:\\', re.I)


def __get_system_datadir() -> pathlib.Path:
  '''
  Returns a parent directory path
  where persistent application data can be stored.

  - linux: ~/.local/share
  - macOS: ~/Library/Application Support
  - windows: C:/Users/<USER>/AppData/Roaming
  '''

  home = pathlib.Path.home()

  if sys.platform == 'win32':
    return home / 'AppData/Roaming'
  elif sys.platform.startswith('linux'):
    return home / '.local/share'
  elif sys.platform == 'darwin':
    return home / 'Library/Application Support'

  raise NotImplementedError(f'"{sys.platform}" is not supported')


def get_app_datadir() -> pathlib.Path:
  app_datadir = os.getenv('RCLIP_DATADIR')
  if app_datadir:
    app_datadir = pathlib.Path(app_datadir)
  else:
    app_datadir = __get_system_datadir() / 'rclip'
  os.makedirs(app_datadir, exist_ok=True)
  return app_datadir


def top_arg_type(arg: str) -> int:
  arg_int = int(arg)
  if arg_int < 1:
    raise argparse.ArgumentTypeError('number of results to display should be >0')
  return arg_int


def init_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    prefix_chars='-+',
    epilog='hints:\n'
    '  relative file path should be prefixed with ./, e.g. "./cat.jpg", not "cat.jpg"\n'
    '  any query can be prefixed with a multiplier, e.g. "2:cat", "0.5:./cat-sleeps-on-a-chair.jpg";'
    ' adding a multiplier is especially useful when combining image and text queries because'
    ' image queries are usually weighted more than text ones\n',
  )
  parser.add_argument('query', help='a text query or a path/URL to an image file')
  parser.add_argument('--add', '-a', '+', metavar='QUERY', action='append', default=[],
                      help='a text query or a path/URL to an image file to add to the "original" query,'
                      ' can be used multiple times')
  parser.add_argument('--subtract', '--sub', '-s', '-', metavar='QUERY', action='append', default=[],
                      help='a text query or a path/URL to an image file to add to the "original" query,'
                      ' can be used multiple times')
  parser.add_argument('--top', '-t', type=top_arg_type, default=10, help='number of top results to display')
  parser.add_argument('--filepath-only', '-f', action='store_true', default=False, help='outputs only filepaths')
  parser.add_argument(
    '--skip-index', '-n',
    action='store_true',
    default=False,
    help='don\'t attempt image indexing, saves time on consecutive runs on huge directories'
  )
  parser.add_argument(
    '--exclude-dir',
    action='append',
    help='dir to exclude from search, can be used multiple times;'
    ' adding this argument overrides the default of ("@eaDir", "node_modules", ".git");'
    ' WARNING: the default will be removed in v2'
  )
  return parser


def remove_prefix(string: str, prefix: str) -> str:
  '''
  Removes prefix from a string (if present) and returns a new string without a prefix
  TODO(yurij): replace with str.removeprefix once updated to Python 3.9+
  '''
  return string[len(prefix):] if string.startswith(prefix) else string


# See: https://meta.wikimedia.org/wiki/User-Agent_policy
def download_image(url: str) -> Image.Image:
  headers = {'User-agent': 'rclip - (https://github.com/yurijmikhalevich/rclip)'}
  check_size = requests.request('HEAD', url, headers=headers, timeout=60)
  if length := check_size.headers.get('Content-Length'):
      if int(length) > MAX_DOWNLOAD_SIZE_BYTES:
          raise ValueError(f"Avoiding download of large ({length} byte) file.")
  img = Image.open(requests.get(url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS).raw)
  return img


def read_image(query: str) -> Image.Image:
  path = remove_prefix(query, 'file://')
  try:
    img = Image.open(path)
  except UnidentifiedImageError as e:
    # by default the filename on the UnidentifiedImageError is None
    e.filename = path
    raise e
  return img


def is_http_url(path: str) -> bool:
  return path.startswith('https://') or path.startswith('http://')


def is_file_path(path: str) -> bool:
  return (
    path.startswith('/') or
    path.startswith('file://') or
    path.startswith('./') or
    WIN_ABSOLUTE_FILE_PATH_REGEX.match(path) is not None
  )
