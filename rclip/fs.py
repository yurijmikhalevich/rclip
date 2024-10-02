import os
from typing import Callable, Pattern

COUNT_FILES_UPDATE_EVERY = 10_000
PRIORITIZED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def count_files(
  directory: str,
  exclude_dir_re: Pattern[str],
  file_re: Pattern[str],
  on_change: Callable[[int], None]
) -> None:
  prev_update_count = 0
  count = 0
  for _ in walk(directory, exclude_dir_re, file_re):
    count += 1
    if count - prev_update_count >= COUNT_FILES_UPDATE_EVERY:
      on_change(count)
      prev_update_count = count
  on_change(count)


def walk(
  directory: str,
  exclude_dir_re: Pattern[str],
  file_re: Pattern[str],
):
  '''Walks through a directory recursively and yields files that match the given regex'''
  dirs_to_process = [directory]
  while dirs_to_process:
    dir = dirs_to_process.pop()
    with os.scandir(dir) as it:
      base_names = {}
      sorted_entries = sorted(it, key=(
        lambda entry: os.path.splitext(entry.name)[1].lower() not in PRIORITIZED_IMAGE_EXTENSIONS
      ))
      for entry in sorted_entries:
        if entry.is_dir():
          if not exclude_dir_re.match(entry.path):
            dirs_to_process.append(entry.path)
        elif entry.is_file() and file_re.match(entry.name):
          base_name = os.path.splitext(entry.name)[0]
          if base_name not in base_names:
            base_names[base_name] = entry.name
            yield entry
