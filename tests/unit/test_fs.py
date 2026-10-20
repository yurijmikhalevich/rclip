import os
import re
import sys
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("ancestor", [".worktrees", ".git", "node_modules"])
def test_walk_exclusions_are_relative_to_root(tmp_path: Path, ancestor: str):
  root = tmp_path / ancestor / "checkout"
  _touch(root / "photo.jpg")
  _touch(root / "nested" / "nested.jpg")
  _touch(root / ".hidden.jpg")
  _touch(root / "node_modules" / "excluded.jpg")

  assert _walked_names(root) == ["nested.jpg", "photo.jpg"]


def test_walk_reports_and_skips_unreadable_directories(tmp_path: Path, monkeypatch):
  _touch(tmp_path / "photo.jpg")
  _touch(tmp_path / "locked" / "hidden.jpg")
  _touch(tmp_path / "locked" / "nested" / "deeper.jpg")
  scandir = os.scandir

  def deny_locked_dir(path):
    if os.fspath(path) == str(tmp_path / "locked"):
      raise PermissionError(13, "Permission denied", path)
    return scandir(path)

  monkeypatch.setattr(os, "scandir", deny_locked_dir)
  errors: list[str] = []
  entries = fs.walk(str(tmp_path), EXCLUDE_DIR_RE, IMAGE_RE, on_error=lambda dir, error: errors.append(dir))

  assert sorted(entry.name for entry in entries) == ["photo.jpg"]
  assert errors == [str(tmp_path / "locked")]
  with pytest.raises(PermissionError):
    _walked_names(tmp_path)


cannot_lock_dirs = pytest.mark.skipif(
  sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions that apply to the current user"
)


@cannot_lock_dirs
def test_walk_reports_and_skips_symlinks_to_inaccessible_targets(tmp_path: Path):
  root, locked = tmp_path / "root", tmp_path / "locked"
  _touch(root / "a.jpg")
  _touch(root / "z.jpg")
  _touch(locked / "sub" / "hidden.jpg")
  _touch(locked / "linked.jpg")
  (root / "b_dir_link").symlink_to(locked / "sub")
  (root / "c_file_link.jpg").symlink_to(locked / "linked.jpg")
  locked.chmod(0)
  try:
    errors: list[str] = []
    entries = fs.walk(str(root), EXCLUDE_DIR_RE, IMAGE_RE, on_error=lambda path, error: errors.append(path))

    assert sorted(entry.name for entry in entries) == ["a.jpg", "z.jpg"]
    assert sorted(errors) == [str(root / "b_dir_link"), str(root / "c_file_link.jpg")]
    with pytest.raises(PermissionError):
      _walked_names(root)

    def reraise(path: str, error: OSError):
      errors.append(path)
      raise error

    errors.clear()
    with pytest.raises(PermissionError):
      list(fs.walk(str(root), EXCLUDE_DIR_RE, IMAGE_RE, on_error=reraise))
    # an error re-raised for an entry isn't reported again as an error reading its directory
    assert len(errors) == 1 and errors[0] != str(root)
  finally:
    locked.chmod(0o755)
