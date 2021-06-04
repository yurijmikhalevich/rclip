import argparse
import os
from os import path
from typing import List, NamedTuple, cast

import clip
import numpy as np
import torch
from PIL import Image


class SearchResult(NamedTuple):
  filepath: str
  score: float


def init_arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('query')
  return parser


device = 'cpu'
model, preprocess = clip.load('ViT-B/32', device=device)
batch_size = 8


def compute_clip_features(thumbs_batch):
  thumbs = [Image.open(thumb_file) for thumb_file in thumbs_batch]

  thumbs_preprocessed = torch.stack([preprocess(thumb) for thumb in thumbs]).to(device)

  with torch.no_grad():
    thumbs_features = model.encode_image(thumbs_preprocessed)
    thumbs_features /= thumbs_features.norm(dim=-1, keepdim=True)

  return thumbs_features.cpu().numpy()


def search(query: str, directory: str, top_k: int = 10) -> List[SearchResult]:
  features = []
  paths = []
  batch = []
  for root, _, files in os.walk(directory):
    for file in (f for f in files if f.lower().endswith('.jpg')):
      filepath = path.join(root, file)
      paths.append(filepath)
      batch.append(filepath)
      if len(batch) >= batch_size:
        features.append(compute_clip_features(batch))
        batch = []
  if len(batch) != 0:
    features.append(compute_clip_features(batch))

  features = np.concatenate(features)

  with torch.no_grad():
    text_encoded = model.encode_text(clip.tokenize(query).to(device))
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

  text_features = text_encoded.cpu().numpy()
  similarities = list((text_features @ features.T).squeeze(0))
  best_thumbs = sorted(zip(similarities, range(features.shape[0])), key=lambda x: x[0], reverse=True)

  return [SearchResult(filepath=paths[th[1]], score=th[0]) for th in best_thumbs[:top_k]]


def main():
  arg_parser = init_arg_parser()
  args = arg_parser.parse_args()
  result = search(args.query, os.getcwd())
  for r in result:
    print(r)


if __name__ == '__main__':
  main()
