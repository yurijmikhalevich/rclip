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
