import os
import sys


def is_snap():
  return bool(os.getenv("SNAP"))


def print_snap_permission_error(directory: str, is_current_directory: bool = False):
  homedir = os.getenv("SNAP_REAL_HOME")
  if not homedir:
    print(
      "SNAP_REAL_HOME environment variable is not set."
      " Please, report the issue to the rclip project on"
      " GitHub https://github.com/yurijmikhalevich/rclip/issues."
    )
    return

  if directory == homedir or directory.startswith(homedir + os.sep):
    print(
      f"rclip doesn't have access to {'the current directory' if is_current_directory else directory}."
      " You can resolve this issue by running:"
      "\n\n\tsudo snap connect rclip:home\n\n"
      "This command will grant rclip the necessary access to the home directory."
      " Afterward, you can try again."
    )
  elif directory == "/media" or directory.startswith("/media" + os.sep):
    print(
      f"rclip doesn't have access to {'the current directory' if is_current_directory else directory}."
      " You can resolve this issue by running:"
      "\n\n\tsudo snap connect rclip:removable-media\n\n"
      'This command will grant rclip the necessary access to the "/media" directory.'
      " Afterward, you can try again."
    )
  else:
    print(
      'rclip installed with snap cannot access files outside of the home or "/media" directories.'
      ' If you want to use rclip outside of home or "/media",'
      " file an issue in the rclip project on GitHub https://github.com/yurijmikhalevich/rclip/issues,"
      " describe your use case, and consider alternative rclip installation"
      " options https://github.com/yurijmikhalevich/rclip#linux."
    )


def check_snap_permissions(directory: str, is_current_directory: bool = False):
  try:
    any(os.scandir(directory))
  except PermissionError:
    print_snap_permission_error(directory, is_current_directory)
    sys.exit(1)
