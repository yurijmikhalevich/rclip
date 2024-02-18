import os
import sys


def is_snap():
  return bool(os.getenv('SNAP'))


def check_snap_permissions(directory: str, package_name: str = "rclip"):
  try:
    any(os.scandir(directory))
  except PermissionError:
    homedir = os.getenv('SNAP_REAL_HOME')
    if not homedir:
      print(
        'SNAP_REAL_HOME environment variable is not set.'
        f' Please, report the issue to the {package_name} project on'
        f' GitHub https://github.com/yurijmikhalevich/{package_name}/issues.'
      )
      sys.exit(1)
    if directory == homedir or directory.startswith(homedir + os.sep):
      print(
        f'{package_name} doesn\'t have access to the current directory.'
        ' You can resolve this issue by running:'
        f'\n\n\tsudo snap connect {package_name}:home\n\n'
        f'This command will grant {package_name} the necessary access to the home directory.'
        ' Afterward, you can try again.'
      )
      sys.exit(1)
    if directory == '/media' or directory.startswith('/media' + os.sep):
      print(
        f'{package_name} doesn\'t have access to the current directory.'
        ' You can resolve this issue by running:'
        f'\n\n\tsudo snap connect {package_name}:removable-media\n\n'
        f'This command will grant {package_name} the necessary access to the "/media" directory.'
        ' Afterward, you can try again.'
      )
      sys.exit(1)
    print(
      f'Running {package_name} outside of the home or "/media" directories is not supported by its snap version.'
      f' If you want to use {package_name} outside of home or "/media",'
      f' file an issue in the {package_name} project on GitHub'
      f' https://github.com/yurijmikhalevich/{package_name}/issues,'
      f' describe your use case, and consider alternative {package_name} installation'
      f' options https://github.com/yurijmikhalevich/{package_name}#linux.'
    )
    sys.exit(1)
