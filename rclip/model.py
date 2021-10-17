from typing import Callable, List, Tuple, cast

import clip
import clip.model
import itertools
import numpy as np
from PIL import Image
import re
import requests
import torch
import torch.nn


class Model:
  VECTOR_SIZE = 512
  _device = 'cpu'
  _model_name = 'ViT-B/32'

  def __init__(self):
    model, preprocess = cast(
      Tuple[clip.model.CLIP, Callable[[Image.Image], torch.Tensor]],
      clip.load(self._model_name, device=self._device)
    )
    self._model = model
    self._preprocess = preprocess

  def compute_image_features(self, images: List[Image.Image]) -> np.ndarray:
    images_preprocessed = torch.stack([self._preprocess(thumb) for thumb in images]).to(self._device)

    with torch.no_grad():
      image_features = self._model.encode_image(images_preprocessed)
      image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy()

  def compute_text_features(self, text: List[str]) -> np.ndarray:
    with torch.no_grad():
      text_encoded = self._model.encode_text(clip.tokenize(text).to(self._device))
      text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    return text_encoded.cpu().numpy()

  # See: https://meta.wikimedia.org/wiki/User-Agent_policy
  def download_image(self, url: str) -> Image.Image:
    headers = {'User-agent': 'rclip - (https://github.com/yurijmikhalevich/rclip)'}
    check_size = requests.request('HEAD', url, headers = headers, timeout = 60)
    if length := check_size.headers.get('Content-Length'):
        if int(length) > 50_000_000:
            raise(ValueError(f"Avoiding download of large ({length} byte) file."))
    img = Image.open(requests.get(url, headers = headers, stream = True, timeout = 60).raw)
    return img

  def image_from_file(self, query: str) -> Image.Image:
    path = query.removeprefix('file://')
    img = Image.open(path)
    return img

  def group_query_parameters_by_type(self, queries: List[str]) -> Tuple[List[str]]:
    phrase_queries = []
    local_file_queries = []
    url_queries = []
    for query in queries:
        if query.startswith('https://') or query.startswith('http://'):
          url_queries.append(query)
        elif (query.startswith('/') or
              query.startswith('file://') or
              query.startswith('./') or
              re.match(r'(?i)^[a-z]:\\', query)):
          local_file_queries.append(query)
        else:
          phrase_queries.append(query)
    return phrase_queries, local_file_queries, url_queries

  def compute_features_for_queries(self, queries: List[str]) -> Tuple[np.ndarray]:
    text_features = image_features = None
    phrases,files,urls = self.group_query_parameters_by_type(queries)
    if phrases:
      text_features = np.add.reduce(self.compute_text_features(phrases))
    if files or urls:
      images = ([self.download_image(q) for q in urls] +
                [self.image_from_file(q) for q in files])
      image_features = np.add.reduce(self.compute_image_features(images))
    return(text_features,image_features)
        

  def compute_similarities_to_text(
      self, item_features: np.ndarray,
      positive_queries: List[str], negative_queries: List[str]) -> List[Tuple[float, int]]:

    text_features, image_features = self.compute_features_for_queries(positive_queries)
    n_text_features, n_image_features = self.compute_features_for_queries(negative_queries)

    features = np.zeros(Model.VECTOR_SIZE)
    if text_features is not None:
        features += text_features
    if image_features is not None:
        features += image_features
    if n_text_features is not None:
        features -= n_text_features
    if n_image_features is not None:
        features -= n_image_features

    similarities = features @ item_features.T
    sorted_similarities = sorted(zip(similarities, range(item_features.shape[0])), key=lambda x: x[0], reverse=True)

    return sorted_similarities
