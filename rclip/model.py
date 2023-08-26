import re
from typing import List, Tuple, Optional, cast
import sys

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError
from rclip.utils import helpers

QUERY_WITH_MULTIPLIER_RE = re.compile(r'^(?P<multiplier>(\d+(\.\d+)?|\.\d+|\d+\.)):(?P<query>.+)$')
QueryWithMultiplier = Tuple[float, str]
FeatureVector = npt.NDArray[np.float32]


class Model:
  VECTOR_SIZE = 512
  _model_name = 'ViT-B-32'
  _checkpoint_name = 'openai'

  def __init__(self, device: str = 'cpu'):
    self._device = device
    self.__model = None
    self.__preprocess = None
    self.__tokenizer = None

  @property
  def _tokenizer(self):
    import open_clip
    if not self.__tokenizer:
      self.__tokenizer = open_clip.get_tokenizer(self._model_name)
    return self.__tokenizer

  @property
  def _model(self):
    import open_clip
    if not self.__model:
      self.__model, _, self.__preprocess = open_clip.create_model_and_transforms(
        self._model_name,
        pretrained=self._checkpoint_name,
        device=self._device,
      )
    return self.__model

  @property
  def _preprocess(self):
    import open_clip
    if not self.__preprocess:
      self.__model, _, self.__preprocess = open_clip.create_model_and_transforms(
        self._model_name,
        pretrained=self._checkpoint_name,
        device=self._device,
      )
    return self.__preprocess

  def compute_image_features(self, images: List[Image.Image]) -> np.ndarray:
    import torch
    images_preprocessed = torch.stack([self._preprocess(thumb) for thumb in images]).to(self._device)
    with torch.no_grad():
      image_features = self._model.encode_image(images_preprocessed)
      image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

  def compute_text_features(self, text: List[str]) -> np.ndarray:
    import torch
    with torch.no_grad():
      text_features = self._model.encode_text(self._tokenizer(text).to(self._device))
      text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

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
        if helpers.is_http_url(query):
          url_queries.append((multiplier, query))
        elif helpers.is_file_path(query):
          local_file_queries.append((multiplier, query))
        else:
          phrase_queries.append((multiplier, query))
    return phrase_queries, local_file_queries, url_queries

  def compute_features_for_queries(self, queries: List[str]) -> FeatureVector:
    text_features: Optional[FeatureVector] = None
    image_features: Optional[FeatureVector] = None
    phrases, files, urls = self._group_queries_by_type(queries)
    if phrases:
      phrase_multipliers, phrase_queries = cast(Tuple[Tuple[float], Tuple[str]], zip(*phrases))
      phrase_multipliers_np = np.array(phrase_multipliers).reshape(-1, 1)
      text_features = np.add.reduce(self.compute_text_features([*phrase_queries]) * phrase_multipliers_np)
    if files or urls:
      file_multipliers, file_paths = cast(Tuple[Tuple[float], Tuple[str]], zip(*(files))) if files else ((), ())
      url_multipliers, url_paths = cast(Tuple[Tuple[float], Tuple[str]], zip(*(urls))) if urls else ((), ())
      try:
        images = ([helpers.download_image(q) for q in url_paths] +
                  [helpers.read_image(q) for q in file_paths])
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
        return np.zeros(Model.VECTOR_SIZE, dtype=np.float32)

  def compute_similarities_to_text(
      self, item_features: FeatureVector,
      positive_queries: List[str], negative_queries: List[str]) -> List[Tuple[float, int]]:

    positive_features = self.compute_features_for_queries(positive_queries)
    negative_features = self.compute_features_for_queries(negative_queries)

    features = positive_features - negative_features

    similarities = features @ item_features.T
    sorted_similarities = sorted(zip(similarities, range(item_features.shape[0])), key=lambda x: x[0], reverse=True)

    return sorted_similarities
