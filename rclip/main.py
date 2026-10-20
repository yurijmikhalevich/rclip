import heapq
import itertools
import os
import re
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, TypedDict, cast

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import PIL
import PIL.Image

from rclip import db, fs, model
from rclip.const import IMAGE_EXT, IMAGE_RAW_EXT
from rclip.utils.preprocess import IMAGE_SIZE, preprocess
from rclip.utils.preview import preview
from rclip.utils.snap import check_snap_permissions, is_snap, get_snap_permission_error
from rclip.utils import helpers, terminal


class ImageMeta(TypedDict):
  modified_at: float
  size: int


PathMetaVector = Tuple[str, ImageMeta, str, model.FeatureVector]


def get_image_meta(entry: os.DirEntry[str]) -> ImageMeta:
  stat = entry.stat()
  return ImageMeta(modified_at=stat.st_mtime, size=stat.st_size)


def is_image_meta_equal(image: db.ImageState, meta: ImageMeta) -> bool:
  return meta["modified_at"] == image["modified_at"] and meta["size"] == image["size"]


def _too_large_message(path: str, pixels: int, limit: int) -> str:
  size = f" ({pixels / 1_000_000:.0f} MP)" if pixels else ""
  return (
    f"skipping {path}: it is too large to process{size};"
    f" the limit is {limit / 1_000_000:.0f} MP."
    ' Raise or disable it with "--max-image-megapixels" if you want to index this image'
  )


def _read_and_preprocess(path: str) -> Tuple[str, str, npt.NDArray[np.float32]]:
  """Reads an image, computes its hash, and runs the CLIP preprocessing on it. Runs on the loader
  threads so that the decoding and resizing happen in parallel and only the
  model forward pass is left for the consumer."""
  file_hash = helpers.compute_file_hash(path)
  preprocessed = preprocess(helpers.read_image(path, min_side=IMAGE_SIZE))
  return path, file_hash, preprocessed


class RClip:
  EXCLUDE_DIRS_DEFAULT = ["@eaDir", "node_modules", ".git", "System Volume Information"]
  DB_IMAGES_BEFORE_COMMIT = 50_000
  # how many indexing batches to keep loading ahead of the model; preprocessed
  # images are small (a few hundred KB each), so a few batches of look-ahead
  # cost little memory.
  LOOKAHEAD_BATCHES = 3
  MAX_IMAGE_LOADING_WORKERS = 16
  SEARCH_BATCH_SIZE = 4096

  class SearchResult(NamedTuple):
    filepath: str
    score: float

  class ImageCursor(NamedTuple):
    modified_at: float
    filepath: str

  class ImagePage(NamedTuple):
    filepaths: list[str]
    next_cursor: "RClip.ImageCursor | None"

  def __init__(
    self,
    model_instance: model.Model,
    database: db.DB,
    indexing_batch_size: int,
    exclude_dirs: Optional[List[str]],
    enable_raw_support: bool = False,
    max_image_pixels: helpers.MaxImagePixels = helpers.AUTO_MAX_IMAGE_PIXELS,
    include_hidden: bool = False,
  ):
    self._model = model_instance
    self._db = database
    self._indexing_batch_size = indexing_batch_size
    self._enable_raw_support = enable_raw_support
    self._skip_hidden = not include_hidden

    supported_image_ext = IMAGE_EXT + (IMAGE_RAW_EXT if enable_raw_support else [])
    self._image_regex = re.compile(f"^.+\\.({'|'.join(supported_image_ext)})$", re.I)

    excluded_dirs = "|".join(re.escape(dir) for dir in exclude_dirs or self.EXCLUDE_DIRS_DEFAULT)
    self._exclude_dir_regex = re.compile(f"^.+\\{os.path.sep}({excluded_dirs})(\\{os.path.sep}.+)?$")

    self._image_loading_executor: Optional[ThreadPoolExecutor] = None
    self._image_loading_workers = max(1, min(self.MAX_IMAGE_LOADING_WORKERS, os.cpu_count() or 1))
    helpers.configure_max_image_pixels(max_image_pixels, self._image_loading_workers)

  def _get_image_loading_executor(self) -> ThreadPoolExecutor:
    if self._image_loading_executor is None:
      self._image_loading_executor = ThreadPoolExecutor(max_workers=self._image_loading_workers)
    return self._image_loading_executor

  def _is_excluded(self, filepath: str, directory: str) -> bool:
    relative_path = os.path.relpath(filepath, directory)
    if self._skip_hidden and any(part.startswith(".") for part in relative_path.split(os.path.sep)):
      return True
    return self._exclude_dir_regex.match(os.path.join(".", os.path.dirname(relative_path))) is not None

  def _shutdown_image_loading_executor(self) -> None:
    if self._image_loading_executor is None:
      return
    self._image_loading_executor.shutdown(wait=True)
    self._image_loading_executor = None

  def close(self) -> None:
    self._shutdown_image_loading_executor()

  def _load_images(
    self, items: Iterable[Tuple[str, ImageMeta]]
  ) -> Iterator[Tuple[str, ImageMeta, str, npt.NDArray[np.float32]]]:
    """Reads and preprocesses images on the loader threads, yielding them as soon as each one is ready.
    One slow image doesn't hold back the ones queued behind it, so the yield order may differ from `items`."""
    helpers._ensure_image_loading_configured()
    executor = self._get_image_loading_executor()
    # keep a few full batches in flight so they preprocess while the model
    # processes the current one.
    max_in_flight = max(self.LOOKAHEAD_BATCHES * self._indexing_batch_size, self._image_loading_workers)
    items_iter = iter(items)
    in_flight: Dict[Future[Tuple[str, str, npt.NDArray[np.float32]]], Tuple[str, ImageMeta]] = {}

    def submit_next() -> None:
      item = next(items_iter, None)
      if item is not None:
        path, meta = item
        in_flight[executor.submit(_read_and_preprocess, path)] = (path, meta)

    for _ in range(max_in_flight):
      submit_next()

    while in_flight:
      done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
      for future in done:
        path, meta = in_flight.pop(future)
        submit_next()  # refill so the window stays full while the consumer is busy
        try:
          _, file_hash, preprocessed = future.result()
          yield path, meta, file_hash, preprocessed
        except helpers.ImageTooLargeError as ex:
          print(_too_large_message(path, ex.pixels, ex.limit), file=sys.stderr)
        except (PIL.Image.DecompressionBombError, PIL.Image.DecompressionBombWarning) as ex:
          # backstop for formats whose true size only surfaces while decoding in the worker
          print(
            _too_large_message(path, helpers.parse_bomb_pixels(ex), helpers.get_max_image_pixels() or 0),
            file=sys.stderr,
          )
        except MemoryError:
          print(f"skipping {path}: ran out of memory while processing it", file=sys.stderr)
        except PIL.UnidentifiedImageError:
          print(f"skipping {path}: it is not a readable image", file=sys.stderr)
        except Exception as ex:
          print(f"skipping {path}: {ex}", file=sys.stderr)

  def _index_images(self, items: Iterable[Tuple[str, ImageMeta]], seen: set[str]) -> None:
    """Indexes `items`, adding every stored path to `seen`."""
    paths: List[str] = []
    metas: List[ImageMeta] = []
    hashes: List[str] = []
    images: List[npt.NDArray[np.float32]] = []
    renamed_items: List[Tuple[str, ImageMeta, str, bytes]] = []

    def flush() -> None:
      if images:
        self._store_image_features(paths, metas, hashes, images, seen)
        paths.clear()
        metas.clear()
        hashes.clear()
        images.clear()
      for path, meta, file_hash, vector in renamed_items:
        self._db.upsert_image(
          db.NewImage(filepath=path, modified_at=meta["modified_at"], size=meta["size"], vector=vector, hash=file_hash),
          commit=False,
        )
        seen.add(path)
      renamed_items.clear()

    for path, meta, file_hash, image in self._load_images(items):
      existing_images_with_hash = self._db.get_images_by_hash(file_hash, meta["size"])
      if existing_images_with_hash:
        vector = existing_images_with_hash[0]["vector"]
        renamed_items.append((path, meta, file_hash, vector))
      else:
        paths.append(path)
        metas.append(meta)
        hashes.append(file_hash)
        images.append(image)
      if len(images) >= self._indexing_batch_size:
        flush()
    flush()

  def _store_image_features(
    self,
    paths: List[str],
    metas: List[ImageMeta],
    hashes: List[str],
    images: List[npt.NDArray[np.float32]],
    seen: set[str],
  ) -> None:
    try:
      features = self._model.compute_preprocessed_image_features(images, for_indexing=True)
    except Exception as ex:
      print("error computing features:", ex, file=sys.stderr)
      return
    for path, meta, file_hash, vector in cast(Iterable[PathMetaVector], zip(paths, metas, hashes, features)):
      self._db.upsert_image(
        db.NewImage(
          filepath=path, modified_at=meta["modified_at"], size=meta["size"], vector=vector.tobytes(), hash=file_hash
        ),
        commit=False,
      )
      seen.add(path)

  def _does_processed_image_exist_for_raw(self, raw_path: str) -> bool:
    """Check if there is a processed image alongside the raw one; doesn't support mixed-case extensions,
    e.g. it won't detect the .JpG image, but will detect .jpg or .JPG"""

    image_path = os.path.splitext(raw_path)[0]
    for ext in IMAGE_EXT:
      if os.path.isfile(image_path + "." + ext):
        return True
      if os.path.isfile(image_path + "." + ext.upper()):
        return True
    return False

  def _iter_images_to_index(
    self, directory: str, pbar: tqdm, seen: set[str], unreadable_dirs: list[str]
  ) -> Iterator[Tuple[str, ImageMeta]]:
    """Walks the directory and yields (path, meta) for every image that needs (re)indexing, adding the
    ones already up to date in the database to `seen` and the directories it couldn't read to `unreadable_dirs`.
    Advances the progress bar once per scanned file."""

    def on_unreadable_dir(dir: str, error: OSError):
      # under snap, the caller explains how to give rclip access to the directory a symlink points to
      if is_snap() and isinstance(error, PermissionError) and os.path.islink(dir):
        raise error
      print(f"cannot read {dir}: {error.strerror or error}; keeping its images in the index as they are", file=sys.stderr)
      unreadable_dirs.append(dir)

    images_processed = 0
    for entry in fs.walk(directory, self._exclude_dir_regex, self._image_regex, self._skip_hidden, on_unreadable_dir):
      filepath = entry.path

      if self._enable_raw_support:
        file_ext = helpers.get_file_extension(filepath)
        if file_ext in IMAGE_RAW_EXT and self._does_processed_image_exist_for_raw(filepath):
          images_processed += 1
          pbar.update()
          continue

      try:
        meta = get_image_meta(entry)
      except Exception as ex:
        print(f"error getting fs metadata for {filepath}:", ex, file=sys.stderr)
        continue

      if not images_processed % self.DB_IMAGES_BEFORE_COMMIT:
        self._db.commit()
      images_processed += 1
      pbar.update()

      image = self._db.get_image_state(filepath)
      if image and is_image_meta_equal(image, meta):
        if image["deleted"]:
          self._db.restore_image(filepath, commit=False)
        seen.add(filepath)
        continue

      yield filepath, meta

  def ensure_index(self, directory: str):
    print(
      "checking images in the current directory for changes;"
      ' use "--no-indexing" to skip this if no images were added, changed, or removed',
      file=sys.stderr,
    )

    seen: set[str] = set()
    unreadable_dirs: list[str] = []
    with tqdm(total=None, unit="images") as pbar:

      def update_total_images(count: int):
        pbar.total = count
        pbar.refresh()

      counter_thread = threading.Thread(
        target=fs.count_files,
        args=(directory, self._exclude_dir_regex, self._image_regex, update_total_images, self._skip_hidden),
      )
      counter_thread.start()

      self._index_images(self._iter_images_to_index(directory, pbar, seen, unreadable_dirs), seen)

      self._db.commit()
      counter_thread.join()

    self._mark_missing_images_as_deleted(directory, seen, unreadable_dirs)
    print("", file=sys.stderr)

  def _mark_missing_images_as_deleted(self, directory: str, seen: set[str], unreadable_dirs: list[str]) -> None:
    """Marks indexed images under `directory` that weren't found during this run as deleted.
    Excluded paths and unreadable directories and entries weren't inspected, so their cached state is kept."""
    unreadable_paths = set(unreadable_dirs)
    unreadable_prefixes = tuple(os.path.join(dir, "") for dir in unreadable_dirs)
    self._db.mark_images_as_deleted(
      [
        row["filepath"]
        for row in self._db.get_filepaths_by_dir_path(directory)
        if row["filepath"] not in seen
        and row["filepath"] not in unreadable_paths
        and not row["filepath"].startswith(unreadable_prefixes)
        and not self._is_excluded(row["filepath"], directory)
      ]
    )

  def search(
    self,
    query: str,
    directory: str,
    top_k: int | None = 10,
    positive_queries: List[str] = [],
    negative_queries: List[str] = [],
    *,
    cancel_event: threading.Event | None = None,
  ) -> List[SearchResult]:
    positive_queries = [query] + positive_queries
    helpers.raise_if_cancelled(cancel_event)
    positive_features = self._model.compute_features_for_queries(positive_queries)
    helpers.raise_if_cancelled(cancel_event)
    negative_features = self._model.compute_features_for_queries(negative_queries)
    helpers.raise_if_cancelled(cancel_event)
    query_features = positive_features - negative_features

    # exclude images that were part of the query from the results
    exclude_files = {
      os.path.abspath(query) for query in positive_queries + negative_queries if helpers.is_file_path(query)
    }

    def is_excluded(filepath: str) -> bool:
      return filepath in exclude_files or self._is_excluded(filepath, directory)

    candidates: List[RClip.SearchResult] = []
    rows = iter(self._db.get_image_vectors_by_dir_path(directory))
    while batch := list(itertools.islice(rows, self.SEARCH_BATCH_SIZE)):
      helpers.raise_if_cancelled(cancel_event)
      filepaths = [row[0] for row in batch]
      vectors = np.frombuffer(b"".join(row[1] for row in batch), np.float32).reshape(len(batch), -1)
      similarities = query_features @ vectors.T
      candidates.extend(self._best_results(filepaths, similarities, top_k, is_excluded))
      helpers.raise_if_cancelled(cancel_event)

    if top_k is None:
      results = sorted(candidates, key=lambda result: (-result.score, result.filepath))
    else:
      results = heapq.nsmallest(top_k, candidates, key=lambda result: (-result.score, result.filepath))
    helpers.raise_if_cancelled(cancel_event)
    return results

  @staticmethod
  def _best_results(
    filepaths: List[str],
    similarities: npt.NDArray[np.floating],
    top_k: int | None,
    is_excluded: Callable[[str], bool],
  ) -> List[SearchResult]:
    """Returns the `top_k` best results that aren't excluded, along with the ones tied with the worst of them,
    so that the final ranking can break ties by filepath. Walking the results from the best one down means only
    those few candidates are checked for exclusion, rather than every indexed image."""
    results: List[RClip.SearchResult] = []
    for index in np.argsort(-similarities, kind="stable"):
      score = float(similarities[index])
      if top_k is not None and len(results) >= top_k and (not results or score < results[-1].score):
        break
      if not is_excluded(filepaths[index]):
        results.append(RClip.SearchResult(filepaths[index], score))
    return results

  def list_images(
    self,
    directory: str,
    limit: int,
    *,
    after: ImageCursor | None = None,
    cancel_event: threading.Event | None = None,
  ) -> ImagePage:
    results: list[str] = []
    last_cursor: RClip.ImageCursor | None = None
    for row in self._db.get_image_filepaths_by_dir_path(directory, after):
      helpers.raise_if_cancelled(cancel_event)
      filepath = row["filepath"]
      if self._is_excluded(filepath, directory):
        continue
      if len(results) == limit:
        return RClip.ImagePage(results, last_cursor)
      results.append(filepath)
      last_cursor = RClip.ImageCursor(float(row["modified_at"]), filepath)
    return RClip.ImagePage(results, None)


def init_rclip(
  working_directory: str,
  indexing_batch_size: int,
  exclude_dir: Optional[List[str]] = None,
  no_indexing: bool = False,
  enable_raw_support: bool = False,
  max_image_pixels: helpers.MaxImagePixels = helpers.AUTO_MAX_IMAGE_PIXELS,
  include_hidden: bool = False,
  allow_cross_thread_db: bool = False,
):
  datadir = helpers.get_app_datadir()
  db_path = datadir / "db.sqlite3"

  database = db.DB(
    db_path,
    allow_vector_cache_reset=not no_indexing,
    allow_cross_thread=allow_cross_thread_db,
  )
  model_instance = model.Model()
  model_instance.ensure_downloaded()
  rclip = RClip(
    model_instance=model_instance,
    database=database,
    indexing_batch_size=indexing_batch_size,
    exclude_dirs=exclude_dir,
    enable_raw_support=enable_raw_support,
    max_image_pixels=max_image_pixels,
    include_hidden=include_hidden,
  )

  if not no_indexing:
    try:
      rclip.ensure_index(working_directory)
    except PermissionError as e:
      if is_snap() and e.filename is not None and os.path.islink(e.filename):
        symlink_path = e.filename
        realpath = os.path.realpath(e.filename)

        print(f"\n{get_snap_permission_error(realpath, symlink_path, is_current_directory=False)}\n")
        sys.exit(1)
      raise
    model_instance.release_indexing_resources()

  return rclip, model_instance, database


def print_results(result: List[RClip.SearchResult], args: helpers.argparse.Namespace):
  # if we are not outputting to console on windows, ensure unicode encoding is correct
  if not sys.stdout.isatty() and os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8-sig")

  if args.filepath_only:
    for search_result in result:
      print(search_result.filepath)
  else:
    print("score\tfilepath")
    for search_result in result:
      print(f'{search_result.score:.3f}\t"{search_result.filepath}"')
      if args.preview:
        preview(search_result.filepath, args.preview_height)


def main():
  arg_parser = helpers.init_arg_parser()
  args = arg_parser.parse_args()
  if args.force_interactive:
    args.interactive = True
  if args.force_preview:
    args.preview = True

  if args.force_interactive and args.preview and not args.force_preview:
    arg_parser.error("--force-interactive cannot be used with --preview")
  if args.force_preview and args.interactive and not args.force_interactive:
    arg_parser.error("--force-preview cannot be used with --interactive")
  if args.force_interactive and args.force_preview:
    arg_parser.error("--force-interactive and --force-preview cannot be used together")
  if args.force_interactive and args.filepath_only:
    arg_parser.error("--force-interactive cannot be used with --filepath-only")
  if args.force_preview and args.filepath_only:
    arg_parser.error("--force-preview cannot be used with --filepath-only")

  if args.query is None and not args.interactive:
    arg_parser.error("query is required unless --interactive is used")
  if args.interactive and args.query is not None:
    arg_parser.error("--interactive does not accept a query")
  if args.interactive and (args.add or args.subtract):
    arg_parser.error("--interactive does not support --add or --subtract")
  if args.interactive and args.top and args.top > 100:
    arg_parser.error("--top cannot exceed 100 with --interactive")
  top_k = args.top or (100 if args.interactive else 10)

  if args.interactive and not args.force_interactive:
    terminal.check_interactive_terminal_or_exit()
  if args.preview and not args.force_preview:
    terminal.check_preview_terminal_or_exit()

  current_directory = os.getcwd()
  if is_snap():
    check_snap_permissions(current_directory, is_current_directory=True)

  rclip, model_instance, db = init_rclip(
    working_directory=current_directory,
    indexing_batch_size=args.indexing_batch_size,
    exclude_dir=args.exclude_dir,
    no_indexing=args.no_indexing,
    enable_raw_support=args.experimental_raw_support,
    max_image_pixels=args.max_image_megapixels,
    include_hidden=args.include_hidden,
    allow_cross_thread_db=args.interactive,
  )

  try:
    if args.interactive:
      from rclip.tui import run_tui

      run_tui(rclip, current_directory, top_k)
    else:
      result = rclip.search(args.query, current_directory, top_k, args.add, args.subtract)
      print_results(result, args)
  finally:
    rclip.close()
    model_instance.close()
    db.close()


if __name__ == "__main__":
  main()
