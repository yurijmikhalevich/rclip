import os
from typing import Callable, Iterator, Optional, Pattern

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
  Directories and entries that can't be read, such as a symlink to an inaccessible target, are passed to
  `on_error` and skipped, or raise if it isn't set."""

  def report(path: str, error: OSError) -> None:
    if on_error is None:
      raise error
    on_error(path, error)

  dirs_to_process = [directory]
  while dirs_to_process:
    dir = dirs_to_process.pop()
    for entry in _scandir(dir, report):
      if skip_hidden and entry.name.startswith("."):
        continue
      try:
        is_dir = entry.is_dir()
        is_file = not is_dir and entry.is_file()
      except OSError as error:
        report(entry.path, error)
        continue
      if is_dir:
        if not exclude_dir_re.match(os.path.join(".", os.path.relpath(entry.path, directory))):
          dirs_to_process.append(entry.path)
      elif is_file and file_re.match(entry.name):
        yield entry


def _scandir(dir: str, report: Callable[[str, OSError], None]) -> Iterator[os.DirEntry[str]]:
  """Yields the entries of `dir`, passing an error reading it to `report`. Errors raised while handling
  an entry stay with the caller, since they aren't raised inside this generator."""
  try:
    with os.scandir(dir) as it:
      yield from it
  except OSError as error:
    report(dir, error)
