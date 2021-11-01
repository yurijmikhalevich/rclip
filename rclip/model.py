from typing import Callable, List, Tuple, Optional, cast

import clip
import clip.model
import numpy as np
from PIL import Image
from rclip import utils
import re
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

  def group_query_parameters_by_type(self, queries: List[str]) -> Tuple[List[str], List[str], List[str]]:
    phrase_queries: List[str] = []
    local_file_queries: List[str] = []
    url_queries: List[str] = []
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

  def compute_features_for_queries(self, queries: List[str]) -> np.ndarray:
    text_features: Optional[np.ndarray] = None
    image_features: Optional[np.ndarray] = None
    phrases, files, urls = self.group_query_parameters_by_type(queries)
    if phrases:
      text_features = np.add.reduce(self.compute_text_features(phrases))
    if files or urls:
      images = ([utils.download_image(q) for q in urls] +
                [utils.read_image(q) for q in files])
      image_features = np.add.reduce(self.compute_image_features(images))

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
