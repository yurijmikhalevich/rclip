import os
import sys


def is_snap():
  return bool(os.getenv("SNAP"))


def check_snap_permissions(directory: str):
  try:
    any(os.scandir(directory))
  except PermissionError:
    homedir = os.getenv("SNAP_REAL_HOME")
    if not homedir:
      print(
        "SNAP_REAL_HOME environment variable is not set."
        " Please, report the issue to the rclip project on"
        " GitHub https://github.com/yurijmikhalevich/rclip/issues."
      )
      sys.exit(1)
    if directory == homedir or directory.startswith(homedir + os.sep):
      print(
        "rclip doesn't have access to the current directory."
        " You can resolve this issue by running:"
        "\n\n\tsudo snap connect rclip:home\n\n"
        "This command will grant rclip the necessary access to the home directory."
        " Afterward, you can try again."
      )
      sys.exit(1)
    if directory == "/media" or directory.startswith("/media" + os.sep):
      print(
        "rclip doesn't have access to the current directory."
        " You can resolve this issue by running:"
        "\n\n\tsudo snap connect rclip:removable-media\n\n"
        'This command will grant rclip the necessary access to the "/media" directory.'
        " Afterward, you can try again."
      )
      sys.exit(1)
    print(
      'Running rclip outside of the home or "/media" directories is not supported by its snap version.'
      ' If you want to use rclip outside of home or "/media",'
      " file an issue in the rclip project on GitHub https://github.com/yurijmikhalevich/rclip/issues,"
      " describe your use case, and consider alternative rclip installation"
      " options https://github.com/yurijmikhalevich/rclip#linux."
    )
    sys.exit(1)
