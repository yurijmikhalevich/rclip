import itertools
import os
from os import path
import re
import textwrap
from typing import Iterable, List, NamedTuple, Optional, Tuple, TypedDict, cast

import numpy as np
from tqdm import tqdm
import PIL
from PIL import Image, ImageFile

from rclip import db, model
from rclip.utils.preview import preview
from rclip.utils.snap import check_snap_permissions, is_snap
from rclip.utils import helpers


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageMeta(TypedDict):
  modified_at: float
  size: int


PathMetaVector = Tuple[str, ImageMeta, model.FeatureVector]


def get_image_meta(filepath: str) -> ImageMeta:
  return ImageMeta(
    modified_at=os.path.getmtime(filepath),
    size=os.path.getsize(filepath)
  )


def is_image_meta_equal(image: db.Image, meta: ImageMeta) -> bool:
  for key in meta:
    if meta[key] != image[key]:
      return False
  return True


class RClip:
  EXCLUDE_DIRS_DEFAULT = ['@eaDir', 'node_modules', '.git']
  IMAGE_REGEX = re.compile(r'^.+\.(jpe?g|png)$', re.I)
  BATCH_SIZE = 8
  DB_IMAGES_BEFORE_COMMIT = 50_000

  class SearchResult(NamedTuple):
    filepath: str
    score: float

  def __init__(self, model_instance: model.Model, database: db.DB, exclude_dirs: Optional[List[str]]):
    self._model = model_instance
    self._db = database

    excluded_dirs = '|'.join(re.escape(dir) for dir in exclude_dirs or self.EXCLUDE_DIRS_DEFAULT)
    self._exclude_dir_regex = re.compile(f'^.+\\{os.path.sep}({excluded_dirs})(\\{os.path.sep}.+)?$')

  def _index_files(self, filepaths: List[str], metas: List[ImageMeta]):
    images: List[Image.Image] = []
    filtered_paths: List[str] = []
    for path in filepaths:
      try:
        image = Image.open(path)
        images.append(image)
        filtered_paths.append(path)
      except PIL.UnidentifiedImageError as ex:
        pass
      except Exception as ex:
        print(f'error loading image {path}:', ex)

    try:
      features = self._model.compute_image_features(images)
    except Exception as ex:
      print('error computing features:', ex)
      return
    for path, meta, vector in cast(Iterable[PathMetaVector], zip(filtered_paths, metas, features)):
      self._db.upsert_image(db.NewImage(
        filepath=path,
        modified_at=meta['modified_at'],
        size=meta['size'],
        vector=vector.tobytes()
      ), commit=False)

  def ensure_index(self, directory: str):
    # We will mark existing images as existing later
    self._db.flag_images_in_a_dir_as_deleted(directory)

    images_processed = 0
    batch: List[str] = []
    metas: List[ImageMeta] = []
    for root, _, files in os.walk(directory):
      if self._exclude_dir_regex.match(root):
        continue
      filtered_files = list(f for f in files if self.IMAGE_REGEX.match(f))
      if not filtered_files:
        continue
      for file in cast(Iterable[str], tqdm(filtered_files, desc=root)):
        filepath = path.join(root, file)

        image = self._db.get_image(filepath=filepath)
        try:
          meta = get_image_meta(filepath)
        except Exception as ex:
          print(f'error getting fs metadata for {filepath}:', ex)
          continue

        if not images_processed % self.DB_IMAGES_BEFORE_COMMIT:
          self._db.commit()
        images_processed += 1

        if image and is_image_meta_equal(image, meta):
          self._db.remove_deleted_flag(filepath, commit=False)
          continue

        batch.append(filepath)
        metas.append(meta)

        if len(batch) >= self.BATCH_SIZE:
          self._index_files(batch, metas)
          batch = []
          metas = []

    if len(batch) != 0:
      self._index_files(batch, metas)

    self._db.commit()

  def search(
      self, query: str, directory: str, top_k: int = 10,
      positive_queries: List[str] = [], negative_queries: List[str] = []) -> List[SearchResult]:
    filepaths, features = self._get_features(directory)

    positive_queries = [query] + positive_queries
    sorted_similarities = self._model.compute_similarities_to_text(features, positive_queries, negative_queries)

    # exclude images that were part of the query from the results
    exclude_files = [
      os.path.abspath(query) for query in positive_queries + negative_queries if helpers.is_file_path(query)
    ]

    filtered_similarities = filter(
      lambda similarity: (
        not self._exclude_dir_regex.match(filepaths[similarity[1]]) and
        not filepaths[similarity[1]] in exclude_files
      ),
      sorted_similarities
    )
    top_k_similarities = itertools.islice(filtered_similarities, top_k)

    return [RClip.SearchResult(filepath=filepaths[th[1]], score=th[0]) for th in top_k_similarities]

  def _get_features(self, directory: str) -> Tuple[List[str], model.FeatureVector]:
    filepaths: List[str] = []
    features: List[model.FeatureVector] = []
    for image in self._db.get_image_vectors_by_dir_path(directory):
      filepaths.append(image['filepath'])
      features.append(np.frombuffer(image['vector'], np.float32))
    if not filepaths:
      return [], np.ndarray(shape=(0, model.Model.VECTOR_SIZE))
    return filepaths, np.stack(features)


def main():
  arg_parser = helpers.init_arg_parser()
  args = arg_parser.parse_args()

  current_directory = os.getcwd()
  if is_snap():
    check_snap_permissions(current_directory)

  model_instance = model.Model(device=vars(args).get("device", "cpu"))
  datadir = helpers.get_app_datadir()
  database = db.DB(datadir / 'db.sqlite3')
  rclip = RClip(model_instance, database, args.exclude_dir)

  has_any_images = database.has_any_images()
  if args.no_indexing and not has_any_images:
    print('you shouldn\'t use --no-indexing on the first run')
    return

  if not args.no_indexing:
    if not has_any_images:
      text_width = min(70, os.get_terminal_size().columns - 2)
      print(
        '\n' +
        textwrap.fill(
          'When you first run rclip in a new directory, it will index all'
          ' images to build its search database. This indexing process may'
          ' take some time depending on your hardware and number of images.',
          width=text_width,
        ) +
        '\n\n'
        'In the past, indexing took approximately:\n' +
        textwrap.fill(
          '- 1 day to index 73 thousand photos on an NAS with an Intel Celeron J3455 CPU',
          subsequent_indent='  ',
          width=text_width,
        ) +
        '\n' +
        textwrap.fill(
          '- 3 hours to index 1.28 million images on a MacBook with an M1 Max CPU',
          subsequent_indent='  ',
          width=text_width,
        ) +
        '\n\n' +
        textwrap.fill(
          'On subsequent runs in the same directory, rclip will only check for'
          ' and add any new images to the existing index. This is much faster than a full re-index.',
          break_on_hyphens=False,
          width=text_width,
        ) +
        '\n\n' +
        textwrap.fill(
          'You can skip re-indexing entirely on future runs by using'
          ' the "--no-indexing" or "-n" flag if you know no new images were added.',
          break_on_hyphens=False,
          width=text_width,
        ) +
        '\n\n' +
        textwrap.fill(
          'This message is showed only on the first run. Later uses will proceed'
          ' directly to indexing or searching. You can read this message'
          ' again (and find other helpful tips) by running "rclip --help".',
          width=text_width,
        ),
        '\n\n'
        'Proceed? [y/n] ',
        end='',
      )
      if input().lower() != 'y':
        return
    rclip.ensure_index(current_directory)

  result = rclip.search(args.query, current_directory, args.top, args.add, args.subtract)
  if args.filepath_only:
    for r in result:
      print(r.filepath)
  else:
    print('score\tfilepath')
    for r in result:
      print(f'{r.score:.3f}\t"{r.filepath}"')
      if args.preview:
        preview(r.filepath, args.preview_height)


if __name__ == '__main__':
  main()
