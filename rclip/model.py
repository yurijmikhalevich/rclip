from typing import Callable, List, Tuple, cast

import clip
import clip.model
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

  def compute_text_or_image_features(self, query: str) -> np.ndarray:
    if query.startswith('https://') or query.startswith('http://'):
      img = self.download_image(query)
      return self.compute_image_features([img])
    elif (query.startswith('/') or
          query.startswith('file://') or
          query.startswith('./') or
          re.match(r'(?i)^[a-z]:\\', query)):
      path = query.removeprefix('file://')
      img = Image.open(path)
      return self.compute_image_features([img])
    else:
      return self.compute_text_features([query])

  def compute_similarities_to_text(
      self, item_features: np.ndarray,
      positive_queries: List[str], negative_queries: List[str]) -> List[Tuple[float, int]]:

    positive_features = np.array([self.compute_text_or_image_features(q)[0] for q in positive_queries])
    negative_features = np.array([self.compute_text_or_image_features(q)[0] for q in negative_queries])
    text_features = np.add.reduce(positive_features)
    if negative_queries:
        text_features -= np.add.reduce(negative_features)

    similarities = text_features @ item_features.T
    sorted_similarities = sorted(zip(similarities, range(item_features.shape[0])), key=lambda x: x[0], reverse=True)

    return sorted_similarities
