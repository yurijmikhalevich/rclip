import os
import re
from pathlib import Path

from rclip import fs

IMAGE_RE = re.compile(r"^.+\.(jpg|jpeg|png)$", re.I)
EXCLUDE_DIR_RE = re.compile(rf"^.+\{os.path.sep}(@eaDir|node_modules|\.git|System Volume Information)(\{os.path.sep}.+)?$")


def _touch(path: Path):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.touch()


def _walked_names(tmp_path: Path, skip_hidden: bool = True):
  return sorted(entry.name for entry in fs.walk(str(tmp_path), EXCLUDE_DIR_RE, IMAGE_RE, skip_hidden))


def test_walk_skips_dot_files_by_default(tmp_path: Path):
  _touch(tmp_path / "photo.jpg")
  _touch(tmp_path / ".DS_Store")
  _touch(tmp_path / "._photo.jpg")  # AppleDouble sidecar, same extension as a real image

  assert _walked_names(tmp_path) == ["photo.jpg"]


def test_walk_skips_dot_directories_by_default(tmp_path: Path):
  _touch(tmp_path / "photo.jpg")
  _touch(tmp_path / ".Spotlight-V100" / "hidden.jpg")
  _touch(tmp_path / ".Trashes" / "deleted.jpg")

  assert _walked_names(tmp_path) == ["photo.jpg"]


def test_walk_skips_system_volume_information_by_default(tmp_path: Path):
  _touch(tmp_path / "photo.jpg")
  _touch(tmp_path / "System Volume Information" / "metadata.jpg")

  assert _walked_names(tmp_path) == ["photo.jpg"]


def test_walk_include_hidden_indexes_dot_files_and_dirs(tmp_path: Path):
  _touch(tmp_path / "photo.jpg")
  _touch(tmp_path / ".DS_Store")
  _touch(tmp_path / ".hidden_dir" / "hidden.jpg")

  # .DS_Store doesn't match the image regex regardless, but the sidecar-like hidden dir now gets walked
  assert _walked_names(tmp_path, skip_hidden=False) == ["hidden.jpg", "photo.jpg"]
