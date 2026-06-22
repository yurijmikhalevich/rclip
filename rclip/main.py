import itertools
import os
import re
import sys
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Deque, Iterable, Iterator, List, NamedTuple, Optional, Tuple, TypedDict, cast

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import PIL
import PIL.Image

from rclip import db, fs, model
from rclip.const import IMAGE_EXT, IMAGE_RAW_EXT
from rclip.utils.preprocess import preprocess
from rclip.utils.preview import preview
from rclip.utils.snap import check_snap_permissions, is_snap, get_snap_permission_error
from rclip.utils import helpers


class ImageMeta(TypedDict):
  modified_at: float
  size: int


PathMetaVector = Tuple[str, ImageMeta, model.FeatureVector]


def get_image_meta(entry: os.DirEntry[str]) -> ImageMeta:
  stat = entry.stat()
  return ImageMeta(modified_at=stat.st_mtime, size=stat.st_size)


def is_image_meta_equal(image: db.Image, meta: ImageMeta) -> bool:
  return meta["modified_at"] == image["modified_at"] and meta["size"] == image["size"]


def _too_large_message(path: str, pixels: int, limit: int) -> str:
  size = f" ({pixels / 1_000_000:.0f} MP)" if pixels else ""
  return (
    f"skipping {path}: it is too large to process{size};"
    f" the limit is {limit / 1_000_000:.0f} MP."
    ' Raise or disable it with "--max-image-megapixels" if you want to index this image'
  )


def _read_and_preprocess(path: str) -> npt.NDArray[np.float32]:
  """Reads an image and runs the CLIP preprocessing on it. Runs on the loader
  threads so that the decoding and resizing happen in parallel and only the
  model forward pass is left for the consumer."""
  return preprocess(helpers.read_image(path))


class RClip:
  EXCLUDE_DIRS_DEFAULT = ["@eaDir", "node_modules", ".git"]
  DB_IMAGES_BEFORE_COMMIT = 50_000
  # how many indexing batches to keep loading ahead of the model; preprocessed
  # images are small (a few hundred KB each), so a few batches of look-ahead
  # cost little memory.
  LOOKAHEAD_BATCHES = 3
  MAX_IMAGE_LOADING_WORKERS = 16

  class SearchResult(NamedTuple):
    filepath: str
    score: float

  def __init__(
    self,
    model_instance: model.Model,
    database: db.DB,
    indexing_batch_size: int,
    exclude_dirs: Optional[List[str]],
    enable_raw_support: bool = False,
    max_image_pixels: helpers.MaxImagePixels = helpers.AUTO_MAX_IMAGE_PIXELS,
  ):
    self._model = model_instance
    self._db = database
    self._indexing_batch_size = indexing_batch_size
    self._enable_raw_support = enable_raw_support

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

  def _shutdown_image_loading_executor(self) -> None:
    if self._image_loading_executor is None:
      return
    self._image_loading_executor.shutdown(wait=True)
    self._image_loading_executor = None

  def close(self) -> None:
    self._shutdown_image_loading_executor()

  def _load_images(
    self, items: Iterable[Tuple[str, ImageMeta]]
  ) -> Iterator[Tuple[str, ImageMeta, npt.NDArray[np.float32]]]:
    helpers._ensure_image_loading_configured()
    executor = self._get_image_loading_executor()
    # keep a few full batches in flight so they preprocess while the model
    # processes the current one.
    max_in_flight = max(self.LOOKAHEAD_BATCHES * self._indexing_batch_size, self._image_loading_workers)
    items_iter = iter(items)
    in_flight: Deque[Tuple[str, ImageMeta, Future[npt.NDArray[np.float32]]]] = deque()

    def submit_next() -> None:
      item = next(items_iter, None)
      if item is not None:
        path, meta = item
        in_flight.append((path, meta, executor.submit(_read_and_preprocess, path)))

    for _ in range(max_in_flight):
      submit_next()

    while in_flight:
      path, meta, future = in_flight.popleft()
      submit_next()  # refill so the window stays full while the consumer is busy
      try:
        yield path, meta, future.result()
      except helpers.ImageTooLargeError as ex:
        print(_too_large_message(path, ex.pixels, ex.limit), file=sys.stderr)
      except (PIL.Image.DecompressionBombError, PIL.Image.DecompressionBombWarning) as ex:
        # backstop for formats whose true size only surfaces while decoding in the worker
        print(
          _too_large_message(path, helpers._parse_bomb_pixels(ex), helpers.get_max_image_pixels() or 0), file=sys.stderr
        )
      except MemoryError:
        print(f"skipping {path}: ran out of memory while processing it", file=sys.stderr)
      except PIL.UnidentifiedImageError:
        print(f"skipping {path}: it is not a readable image", file=sys.stderr)
      except Exception as ex:
        print(f"skipping {path}: {ex}", file=sys.stderr)

  def _index_images(self, items: Iterable[Tuple[str, ImageMeta]]) -> None:
    paths: List[str] = []
    metas: List[ImageMeta] = []
    images: List[npt.NDArray[np.float32]] = []

    def flush() -> None:
      if images:
        self._store_image_features(paths, metas, images)
        paths.clear()
        metas.clear()
        images.clear()

    for path, meta, image in self._load_images(items):
      paths.append(path)
      metas.append(meta)
      images.append(image)
      if len(images) >= self._indexing_batch_size:
        flush()
    flush()

  def _store_image_features(
    self, paths: List[str], metas: List[ImageMeta], images: List[npt.NDArray[np.float32]]
  ) -> None:
    try:
      features = self._model.compute_preprocessed_image_features(images, for_indexing=True)
    except Exception as ex:
      print("error computing features:", ex, file=sys.stderr)
      return
    for path, meta, vector in cast(Iterable[PathMetaVector], zip(paths, metas, features)):
      self._db.upsert_image(
        db.NewImage(filepath=path, modified_at=meta["modified_at"], size=meta["size"], vector=vector.tobytes()),
        commit=False,
      )

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

  def _iter_images_to_index(self, directory: str, pbar: tqdm) -> Iterator[Tuple[str, ImageMeta]]:
    """Walks the directory and yields (path, meta) for every image that needs (re)indexing, skipping
    the ones already up to date in the database. Advances the progress bar once per scanned file."""
    images_processed = 0
    for entry in fs.walk(directory, self._exclude_dir_regex, self._image_regex):
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

      image = self._db.get_image(filepath=filepath)
      if image and is_image_meta_equal(image, meta):
        self._db.remove_indexing_flag(filepath, commit=False)
        continue

      yield filepath, meta

  def ensure_index(self, directory: str):
    print(
      "checking images in the current directory for changes;"
      ' use "--no-indexing" to skip this if no images were added, changed, or removed',
      file=sys.stderr,
    )

    self._db.remove_indexing_flag_from_all_images(commit=False)
    self._db.flag_images_in_a_dir_as_indexing(directory, commit=True)

    with tqdm(total=None, unit="images") as pbar:

      def update_total_images(count: int):
        pbar.total = count
        pbar.refresh()

      counter_thread = threading.Thread(
        target=fs.count_files,
        args=(directory, self._exclude_dir_regex, self._image_regex, update_total_images),
      )
      counter_thread.start()

      self._index_images(self._iter_images_to_index(directory, pbar))

      self._db.commit()
      counter_thread.join()

    self._db.flag_indexing_images_in_a_dir_as_deleted(directory)
    print("", file=sys.stderr)

  def search(
    self,
    query: str,
    directory: str,
    top_k: int = 10,
    positive_queries: List[str] = [],
    negative_queries: List[str] = [],
  ) -> List[SearchResult]:
    filepaths, features = self._get_features(directory)

    positive_queries = [query] + positive_queries
    sorted_similarities = self._model.compute_similarities_to_text(features, positive_queries, negative_queries)

    # exclude images that were part of the query from the results
    exclude_files = [
      os.path.abspath(query) for query in positive_queries + negative_queries if helpers.is_file_path(query)
    ]

    filtered_similarities = filter(
      lambda similarity: (
        not self._exclude_dir_regex.match(filepaths[similarity[1]]) and filepaths[similarity[1]] not in exclude_files
      ),
      sorted_similarities,
    )
    top_k_similarities = itertools.islice(filtered_similarities, top_k)

    return [RClip.SearchResult(filepath=filepaths[th[1]], score=th[0]) for th in top_k_similarities]

  def _get_features(self, directory: str) -> Tuple[List[str], model.FeatureVector]:
    filepaths: List[str] = []
    features: List[model.FeatureVector] = []
    for image in self._db.get_image_vectors_by_dir_path(directory):
      filepaths.append(image["filepath"])
      features.append(np.frombuffer(image["vector"], np.float32))
    if not filepaths:
      return [], np.ndarray(shape=(0, model.Model.VECTOR_SIZE))
    return filepaths, np.stack(features)


def init_rclip(
  working_directory: str,
  indexing_batch_size: int,
  exclude_dir: Optional[List[str]] = None,
  no_indexing: bool = False,
  enable_raw_support: bool = False,
  max_image_pixels: helpers.MaxImagePixels = helpers.AUTO_MAX_IMAGE_PIXELS,
):
  datadir = helpers.get_app_datadir()
  db_path = datadir / "db.sqlite3"

  database = db.DB(db_path, allow_vector_cache_reset=not no_indexing)
  model_instance = model.Model()
  model_instance.ensure_downloaded()
  rclip = RClip(
    model_instance=model_instance,
    database=database,
    indexing_batch_size=indexing_batch_size,
    exclude_dirs=exclude_dir,
    enable_raw_support=enable_raw_support,
    max_image_pixels=max_image_pixels,
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

  current_directory = os.getcwd()
  if is_snap():
    check_snap_permissions(current_directory, is_current_directory=True)

  rclip, model_instance, db = init_rclip(
    current_directory,
    args.indexing_batch_size,
    args.exclude_dir,
    args.no_indexing,
    args.experimental_raw_support,
    args.max_image_megapixels,
  )

  try:
    result = rclip.search(args.query, current_directory, args.top, args.add, args.subtract)
    print_results(result, args)
  finally:
    rclip.close()
    model_instance.close()
    db.close()


if __name__ == "__main__":
  main()
