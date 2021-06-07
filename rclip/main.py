import argparse
import os
from os import path
import re
from typing import List, NamedTuple, Tuple, TypedDict, cast

import clip
import numpy as np
import torch
from tqdm import tqdm
import PIL
from PIL import Image

from rclip import db, utils

DATADIR = utils.get_app_datadir()
DB = db.DB(DATADIR / 'db.sqlite3')

EXCLUDE_DIRS = ['@eaDir', 'node_modules', '.git']
EXCLUDE_DIR_REGEX = re.compile(r'^.+\/(' + '|'.join(re.escape(dir) for dir in EXCLUDE_DIRS) + r')(\/.+)?$')
IMAGE_REGEX = re.compile(r'^.+\.(jpg|png)$', re.I)


class ImageMeta(TypedDict):
  modified_at: float
  size: int


class SearchResult(NamedTuple):
  filepath: str
  score: float


def init_arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('query')
  return parser


model_name = 'ViT-B/32'
device = 'cpu'
model, preprocess = clip.load(model_name, device=device)
batch_size = 8


def compute_clip_features(images: List[Image.Image]) -> np.ndarray:
  images_preprocessed = torch.stack([cast(torch.Tensor, preprocess(thumb)) for thumb in images]).to(device)

  with torch.no_grad():
    images_features = model.encode_image(images_preprocessed)
    images_features /= images_features.norm(dim=-1, keepdim=True)

  images_features = images_features.cpu().numpy()

  return images_features


def get_image_meta(filepath: str) -> ImageMeta:
  return ImageMeta(
    modified_at=os.path.getmtime(filepath),
    size=os.path.getsize(filepath)
  )


def compare_image_meta(image: db.Image, meta: ImageMeta) -> bool:
  for key in meta:
    if meta[key] != image[key]: return False
  return True



def index_files(filepaths: List[str], metas: List[ImageMeta]):
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
    features = compute_clip_features(images)
  except Exception as ex:
    print('error computing features:', ex)
    return
  for path, meta, vector in zip(filtered_paths, metas, features):
    DB.upsert_image(db.NewImage(
      filepath=path,
      modified_at=meta['modified_at'],
      size=meta['size'],
      vector=vector.tobytes()
    ))


def ensure_index(directory: str):
  batch = []
  metas = []
  for root, _, files in tqdm(os.walk(directory), desc=directory):
    if EXCLUDE_DIR_REGEX.match(root): continue
    filtered_files = list(f for f in files if IMAGE_REGEX.match(f))
    if not filtered_files: continue
    for file in tqdm(filtered_files, desc=root):
      filepath = path.join(root, file)

      image = DB.get_image(filepath=filepath)
      try:
        meta = get_image_meta(filepath)
      except Exception as ex:
        print(f'error getting fs metadata for {filepath}:', ex)
        continue
      if image and compare_image_meta(image, meta):
        continue

      batch.append(filepath)
      metas.append(meta)

      if len(batch) >= batch_size:
        index_files(batch, metas)
        batch = []
        metas = []

  if len(batch) != 0:
    index_files(batch, metas)


def get_features(directory: str) -> Tuple[List[str], np.ndarray]:
  filepaths = []
  features = []
  for image in DB.get_images_by_path(directory):
    filepaths.append(image['filepath'])
    features.append(np.frombuffer(image['vector'], np.float32))
  return filepaths, np.stack(features)


def search(query: str, directory: str, top_k: int = 10) -> List[SearchResult]:
  ensure_index(directory)

  filepaths, features = get_features(directory)

  with torch.no_grad():
    text_encoded = model.encode_text(clip.tokenize(query).to(device))
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

  text_features = text_encoded.cpu().numpy()
  similarities = list((text_features @ features.T).squeeze(0))
  best_thumbs = sorted(zip(similarities, range(features.shape[0])), key=lambda x: x[0], reverse=True)

  return [SearchResult(filepath=filepaths[th[1]], score=th[0]) for th in best_thumbs[:top_k]]


def main():
  arg_parser = init_arg_parser()
  args = arg_parser.parse_args()
  result = search(args.query, os.getcwd())
  for r in result:
    print(r)


if __name__ == '__main__':
  main()
