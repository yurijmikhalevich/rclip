import os
import sys


def is_snap():
  return bool(os.getenv("SNAP"))


def get_snap_permission_error(
  directory: str, 
  symlink_path: str | None,
  is_current_directory: bool = False,
) -> str:
  homedir = os.getenv("SNAP_REAL_HOME")
  if not homedir:
    return (
      "SNAP_REAL_HOME environment variable is not set."
      " Please, report the issue to the rclip project on"
      " GitHub https://github.com/yurijmikhalevich/rclip/issues."
    )

  directory_str = "the current directory" if is_current_directory else directory
  
  if symlink_path and symlink_path != directory:
    path_info = f"symlink {symlink_path} which points to {directory_str}"
  else:
    path_info = directory_str
  
  if directory == homedir or directory.startswith(homedir + os.sep):
    return (
      f"rclip doesn't have access to {path_info}."
      " You can resolve this issue by running:"
      "\n\n\tsudo snap connect rclip:home\n\n"
      "This command will grant rclip the necessary access to the home directory."
      " Afterward, you can try again."
    )
  if directory == "/media" or directory.startswith("/media" + os.sep):
    return (
      f"rclip doesn't have access to {path_info}."
      " You can resolve this issue by running:"
      "\n\n\tsudo snap connect rclip:removable-media\n\n"
      'This command will grant rclip the necessary access to the "/media" directory.'
      " Afterward, you can try again."
    )
  return (
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
    symlink_path = None
    realpath = directory
    
    if os.path.islink(directory):
      symlink_path = directory
      realpath = os.path.realpath(directory)
        
    print(get_snap_permission_error(realpath, symlink_path, is_current_directory))
    sys.exit(1)
  
