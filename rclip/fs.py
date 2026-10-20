import os
from typing import Callable, Optional, Pattern

COUNT_FILES_UPDATE_EVERY = 10_000


def count_files(
  directory: str,
  exclude_dir_re: Pattern[str],
  file_re: Pattern[str],
  on_change: Callable[[int], None],
  skip_hidden: bool = True,
) -> None:
  prev_update_count = 0
  count = 0
  # Unreadable directories are reported by the indexing walk, so the count skips them silently
  for _ in walk(directory, exclude_dir_re, file_re, skip_hidden, on_error=lambda dir, error: None):
    count += 1
    if count - prev_update_count >= COUNT_FILES_UPDATE_EVERY:
      on_change(count)
      prev_update_count = count
  on_change(count)


def walk(
  directory: str,
  exclude_dir_re: Pattern[str],
  file_re: Pattern[str],
  skip_hidden: bool = True,
  on_error: Optional[Callable[[str, OSError], None]] = None,
):
  """Walks through a directory recursively and yields files that match the given regex.
  Directories that can't be read are passed to `on_error` and skipped, or raise if it isn't set."""
  dirs_to_process = [directory]
  while dirs_to_process:
    dir = dirs_to_process.pop()
    try:
      with os.scandir(dir) as it:
        for entry in it:
          if skip_hidden and entry.name.startswith("."):
            continue
          if entry.is_dir():
            if not exclude_dir_re.match(os.path.join(".", os.path.relpath(entry.path, directory))):
              dirs_to_process.append(entry.path)
          elif entry.is_file() and file_re.match(entry.name):
            yield entry
    except OSError as error:
      if on_error is None:
        raise
      on_error(dir, error)
