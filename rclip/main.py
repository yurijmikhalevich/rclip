import os
from os import path
import re
from typing import Any, Iterable, List, NamedTuple, Tuple, TypedDict, cast

import numpy as np
from tqdm import tqdm
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from rclip import db, model, utils


class ImageMeta(TypedDict):
  modified_at: float
  size: int


def get_image_meta(filepath: str) -> ImageMeta:
  return ImageMeta(
    modified_at=os.path.getmtime(filepath),
    size=os.path.getsize(filepath)
  )


def compare_image_meta(image: db.Image, meta: ImageMeta) -> bool:
  for key in meta:
    if meta[key] != image[key]: return False
  return True


class RClip:
  EXCLUDE_DIRS = ['@eaDir', 'node_modules', '.git']
  EXCLUDE_DIR_REGEX = re.compile(r'^.+\/(' + '|'.join(re.escape(dir) for dir in EXCLUDE_DIRS) + r')(\/.+)?$')
  IMAGE_REGEX = re.compile(r'^.+\.(jpe?g|png)$', re.I)
  BATCH_SIZE = 8

  class SearchResult(NamedTuple):
    filepath: str
    score: float

  def __init__(self, model_instance: model.Model, database: db.DB):
    self._model = model_instance
    self._db = database

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
      ))

  def ensure_index(self, directory: str):
    batch: List[str] = []
    metas: List[ImageMeta] = []
    for root, _, files in cast(Iterable[Tuple[str, Any, List[str]]], tqdm(os.walk(directory), desc=directory)):
      if self.EXCLUDE_DIR_REGEX.match(root): continue
      filtered_files = list(f for f in files if self.IMAGE_REGEX.match(f))
      if not filtered_files: continue
      for file in cast(Iterable[str], tqdm(filtered_files, desc=root)):
        filepath = path.join(root, file)

        image = self._db.get_image(filepath=filepath)
        try:
          meta = get_image_meta(filepath)
        except Exception as ex:
          print(f'error getting fs metadata for {filepath}:', ex)
          continue
        if image and compare_image_meta(image, meta):
          continue

        batch.append(filepath)
        metas.append(meta)

        if len(batch) >= self.BATCH_SIZE:
          self._index_files(batch, metas)
          batch = []
          metas = []

    if len(batch) != 0:
      self._index_files(batch, metas)

  def search(self, query: str, directory: str, top_k: int = 10) -> List[SearchResult]:
    filepaths, features = self._get_features(directory)

    sorted_similarities = self._model.compute_similarities_to_text(features, query)

    return [RClip.SearchResult(filepath=filepaths[th[1]], score=th[0]) for th in sorted_similarities[:top_k]]

  def _get_features(self, directory: str) -> Tuple[List[str], np.ndarray]:
    filepaths: List[str] = []
    features: List[np.ndarray] = []
    for image in self._db.get_images_by_dir_path(directory):
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
  rclip = RClip(model_instance, database)

  if not args.skip_index:
    rclip.ensure_index(current_directory)

  result = rclip.search(args.query, current_directory, args.top)
  if args.filepath_only:
    for r in result:
      print(r.filepath)
  else:
    print('score\tfilepath')
    for r in result:
      print(f'{r.score:.3f}\t"{r.filepath}"')


if __name__ == '__main__':
  main()
