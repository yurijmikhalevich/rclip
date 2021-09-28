from typing import Callable, List, Tuple, cast

import clip
import clip.model
import numpy as np
from PIL import Image
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

  def compute_similarities_to_text(
      self, item_features: np.ndarray,
      positive_queries: List[str], negative_queries: List[str]) -> List[Tuple[float, int]]:

    positive_features = self.compute_text_features(positive_queries)
    text_features = np.add.reduce(positive_features)
    if (negative_queries):
        negative_features = self.compute_text_features(negative_queries)
        text_features -= np.add.reduce(negative_features)

    similarities = text_features @ item_features.T
    sorted_similarities = sorted(zip(similarities, range(item_features.shape[0])), key=lambda x: x[0], reverse=True)

    return sorted_similarities
