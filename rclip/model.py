import re
from typing import Callable, List, Tuple, Optional, cast
import sys

import clip
import clip.model
import numpy as np
from PIL import Image, UnidentifiedImageError
from rclip import utils
import torch
import torch.nn

QUERY_WITH_MULTIPLIER_RE = re.compile(r'^(?P<multiplier>(\d+(\.\d+)?|\.\d+|\d+\.)):(?P<query>.+)$')
QueryWithMultiplier = Tuple[float, str]


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

  @staticmethod
  def _extract_query_multiplier(query: str) -> QueryWithMultiplier:
    match = QUERY_WITH_MULTIPLIER_RE.match(query)
    if not match:
      return 1., query
    multiplier = float(match.group('multiplier'))
    query = match.group('query')
    return multiplier, query

  @staticmethod
  def _group_queries_by_type(queries: List[str]) -> Tuple[
    List[QueryWithMultiplier],
    List[QueryWithMultiplier],
    List[QueryWithMultiplier]
  ]:
    phrase_queries: List[Tuple[float, str]] = []
    local_file_queries: List[Tuple[float, str]] = []
    url_queries: List[Tuple[float, str]] = []
    for query in queries:
        multiplier, query = Model._extract_query_multiplier(query)
        if utils.is_http_url(query):
          url_queries.append((multiplier, query))
        elif utils.is_file_path(query):
          local_file_queries.append((multiplier, query))
        else:
          phrase_queries.append((multiplier, query))
    return phrase_queries, local_file_queries, url_queries

  def compute_features_for_queries(self, queries: List[str]) -> np.ndarray:
    text_features: Optional[np.ndarray] = None
    image_features: Optional[np.ndarray] = None
    phrases, files, urls = self._group_queries_by_type(queries)
    if phrases:
      phrase_multipliers, phrase_queries = cast(Tuple[Tuple[float], Tuple[str]], zip(*phrases))
      phrase_multipliers_np = np.array(phrase_multipliers).reshape(-1, 1)
      text_features = np.add.reduce(self.compute_text_features([*phrase_queries]) * phrase_multipliers_np)
    if files or urls:
      file_multipliers, file_paths = cast(Tuple[Tuple[float], Tuple[str]], zip(*(files))) if files else ((), ())
      url_multipliers, url_paths = cast(Tuple[Tuple[float], Tuple[str]], zip(*(urls))) if urls else ((), ())
      try:
        images = ([utils.download_image(q) for q in url_paths] +
                  [utils.read_image(q) for q in file_paths])
      except FileNotFoundError as e:
        print(f'File "{e.filename}" not found. Check if you have typos in the filename.')
        sys.exit(1)
      except UnidentifiedImageError as e:
        print(f'File "{e.filename}" is not an image. You can only use image files or text as queries.')
        sys.exit(1)
      image_multipliers = np.array(url_multipliers + file_multipliers)
      image_features = np.add.reduce(self.compute_image_features(images) * image_multipliers.reshape(-1, 1))

    if text_features is not None and image_features is not None:
        return text_features + image_features
    elif text_features is not None:
        return text_features
    elif image_features is not None:
        return image_features
    else:
        return np.zeros(Model.VECTOR_SIZE)

  def compute_similarities_to_text(
      self, item_features: np.ndarray,
      positive_queries: List[str], negative_queries: List[str]) -> List[Tuple[float, int]]:

    positive_features = self.compute_features_for_queries(positive_queries)
    negative_features = self.compute_features_for_queries(negative_queries)

    features = positive_features - negative_features

    similarities = features @ item_features.T
    sorted_similarities = sorted(zip(similarities, range(item_features.shape[0])), key=lambda x: x[0], reverse=True)

    return sorted_similarities
