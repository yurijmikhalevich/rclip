import os
from typing import Callable, Pattern

COUNT_FILES_UPDATE_EVERY = 1000


def count_files(
  directory: str,
  exclude_dir_re: Pattern[str],
  file_re: Pattern[str],
  on_change: Callable[[int], None]
) -> None:
  prev_update_count = 0
  count = 0
  for root, _, files in os.walk(directory):
    if exclude_dir_re.match(root):
      continue
    count += len(list(f for f in files if file_re.match(f)))
    if count - prev_update_count >= COUNT_FILES_UPDATE_EVERY:
      on_change(count)
      prev_update_count = count
  on_change(count)
