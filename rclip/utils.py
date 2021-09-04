import argparse
import os
import pathlib
import sys

from rclip import config


def get_system_datadir() -> pathlib.Path:
  '''
  Returns a parent directory path
  where persistent application data can be stored.

  # linux: ~/.local/share
  # macOS: ~/Library/Application Support
  # windows: C:/Users/<USER>/AppData/Roaming
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
  app_datadir = os.getenv('DATADIR')
  if app_datadir:
    app_datadir = pathlib.Path(app_datadir)
  else:
    app_datadir = get_system_datadir() / config.NAME
  os.makedirs(app_datadir, exist_ok=True)
  return app_datadir


def top_arg_type(arg: str) -> int:
  arg_int = int(arg)
  if arg_int < 1:
    raise argparse.ArgumentTypeError('number of results to display should be >0')
  return arg_int


def init_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument('query')
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
    help='dir to exclude from search, can be specified multiple times;'
    ' adding this argument overrides the default of ("@eaDir", "node_modules", ".git");'
    ' WARNING: the default will be removed in v2'
  )
  return parser
