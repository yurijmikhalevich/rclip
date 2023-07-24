import itertools
import os
from os import path
import re
from typing import Iterable, List, NamedTuple, Optional, Tuple, TypedDict, cast

import numpy as np
from tqdm import tqdm
import PIL
from PIL import Image, ImageFile

from rclip import db, model, utils


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageMeta(TypedDict):
  modified_at: float
  size: int


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
    self._exclude_dir_regex = re.compile(f'^.+\\/({excluded_dirs})(\\/.+)?$')

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
    for path, meta, vector in cast(Iterable[Tuple[str, ImageMeta, np.ndarray]], zip(filtered_paths, metas, features)):
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
      os.path.abspath(query) for query in positive_queries + negative_queries if utils.is_file_path(query)
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

  def _get_features(self, directory: str) -> Tuple[List[str], np.ndarray]:
    filepaths: List[str] = []
    features: List[np.ndarray] = []
    for image in self._db.get_image_vectors_by_dir_path(directory):
      filepaths.append(image['filepath'])
      features.append(np.frombuffer(image['vector'], np.float32))
    if not filepaths:
      return [], np.ndarray(shape=(0, model.Model.VECTOR_SIZE))
    return filepaths, np.stack(features)


def main():
  arg_parser = utils.init_arg_parser()
  args = arg_parser.parse_args()

  current_directory = os.getcwd()

  model_instance = model.Model()
  datadir = utils.get_app_datadir()
  database = db.DB(datadir / 'db.sqlite3')
  rclip = RClip(model_instance, database, args.exclude_dir)

  if not args.skip_index:
    rclip.ensure_index(current_directory)

  result = rclip.search(args.query, current_directory, args.top, args.add, args.subtract)
  if args.filepath_only:
    for r in result:
      print(r.filepath)
  else:
    print('score\tfilepath')
    for r in result:
      print(f'{r.score:.3f}\t"{r.filepath}"')


if __name__ == '__main__':
  main()
