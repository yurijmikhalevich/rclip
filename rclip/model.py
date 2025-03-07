import re
from typing import List, Tuple, Optional, cast
import sys

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError
from rclip.utils import helpers
from importlib.metadata import version
import open_clip

QUERY_WITH_MULTIPLIER_RE = re.compile(r"^(?P<multiplier>(\d+(\.\d+)?|\.\d+|\d+\.)):(?P<query>.+)$")
QueryWithMultiplier = Tuple[float, str]
FeatureVector = npt.NDArray[np.float32]

TEXT_ONLY_SUPPORTED_MODELS = [
  {
    "model_name": "ViT-B-32-quickgelu",
    "checkpoint_name": "openai",
  }
]


def get_open_clip_version():
  return version("open_clip_torch")


class Model:
  VECTOR_SIZE = 512
  _model_name = "ViT-B-32-quickgelu"
  _checkpoint_name = "openai"

  def __init__(self, device: str = "cpu"):
    self._device = device

    self._model_var = None
    self._model_text_var = None
    self._preprocess_var = None
    self._tokenizer_var = None

    self._text_model_path = helpers.get_app_datadir() / f"{self._model_name}_{self._checkpoint_name}_text.pth"
    self._text_model_version_path = (
      helpers.get_app_datadir() / f"{self._model_name}_{self._checkpoint_name}_text.version"
    )

  @property
  def _tokenizer(self):
    if not self._tokenizer_var:
      self._tokenizer_var = open_clip.get_tokenizer(self._model_name)
    return self._tokenizer_var

  def _load_model(self):
    self._model_var, _, self._preprocess_var = open_clip.create_model_and_transforms(
      self._model_name,
      pretrained=self._checkpoint_name,
      device=self._device,
    )
    self._model_text_var = None

    if {
      "model_name": self._model_name,
      "checkpoint_name": self._checkpoint_name,
    } in TEXT_ONLY_SUPPORTED_MODELS and self._should_update_text_model():
      import torch

      model_text = self._get_text_model(cast(open_clip.CLIP, self._model_var))
      torch.save(model_text, self._text_model_path)

      with self._text_model_version_path.open("w") as f:
        f.write(get_open_clip_version())

  @staticmethod
  def _get_text_model(model: open_clip.CLIP):
    import copy

    model_text = copy.deepcopy(model)
    model_text.visual = None  # type: ignore
    return model_text

  def _should_update_text_model(self):
    if not self._text_model_path.exists():
      return True

    if not self._text_model_version_path.exists():
      return True

    with self._text_model_version_path.open("r") as f:
      text_model_version = f.read().strip()

    # to be safe, update the text model on open_clip update (which could update the base model)
    return get_open_clip_version() != text_model_version

  @property
  def _model(self):
    if not self._model_var:
      self._load_model()
    return cast(open_clip.CLIP, self._model_var)

  @property
  def _model_text(self):
    if self._model_var:
      return self._model_var

    if self._model_text_var:
      return self._model_text_var

    if self._text_model_path.exists() and not self._should_update_text_model():
      import torch

      self._model_text_var = torch.load(self._text_model_path, weights_only=False)
      return self._model_text_var

    if not self._model_var:
      self._load_model()

    return cast(open_clip.CLIP, self._model_var)

  @property
  def _preprocess(self):
    from torchvision.transforms import Compose

    if not self._preprocess_var:
      self._load_model()
    return cast(Compose, self._preprocess_var)

  def compute_image_features(self, images: List[Image.Image]) -> npt.NDArray[np.float32]:
    import torch

    images_preprocessed = torch.stack(cast(list[torch.Tensor], [self._preprocess(thumb) for thumb in images])).to(
      self._device
    )
    with torch.no_grad():
      image_features = self._model.encode_image(images_preprocessed)
      image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

  def compute_text_features(self, text: List[str]) -> npt.NDArray[np.float32]:
    import torch

    with torch.no_grad():
      text_features = self._model_text.encode_text(self._tokenizer(text).to(self._device))
      text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

  @staticmethod
  def _extract_query_multiplier(query: str) -> QueryWithMultiplier:
    match = QUERY_WITH_MULTIPLIER_RE.match(query)
    if not match:
      return 1.0, query
    multiplier = float(match.group("multiplier"))
    query = match.group("query")
    return multiplier, query

  @staticmethod
  def _group_queries_by_type(
    queries: List[str],
  ) -> Tuple[List[QueryWithMultiplier], List[QueryWithMultiplier], List[QueryWithMultiplier]]:
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

    # process images first to avoid loading BOTH full and text-only models
    # if we need to process images, we will load the full model, and the text processing logic will use it, too
    # if we don't need to process images, we will skip loading the full model, and the text processing
    # logic will load the text-only model

    if files or urls:
      file_multipliers, file_paths = cast(Tuple[Tuple[float], Tuple[str]], zip(*(files))) if files else ((), ())
      url_multipliers, url_paths = cast(Tuple[Tuple[float], Tuple[str]], zip(*(urls))) if urls else ((), ())
      try:
        images = [helpers.download_image(q) for q in url_paths] + [helpers.read_image(q) for q in file_paths]
      except FileNotFoundError as e:
        print(f'File "{e.filename}" not found. Check if you have typos in the filename.')
        sys.exit(1)
      except UnidentifiedImageError as e:
        print(f'File "{e.filename}" is not an image. You can only use image files or text as queries.')
        sys.exit(1)
      image_multipliers = np.array(url_multipliers + file_multipliers)
      image_features = np.add.reduce(self.compute_image_features(images) * image_multipliers.reshape(-1, 1))

    if phrases:
      phrase_multipliers, phrase_queries = cast(Tuple[Tuple[float], Tuple[str]], zip(*phrases))
      phrase_multipliers_np = np.array(phrase_multipliers).reshape(-1, 1)
      text_features = np.add.reduce(self.compute_text_features([*phrase_queries]) * phrase_multipliers_np)

    if text_features is not None and image_features is not None:
      return text_features + image_features
    elif text_features is not None:
      return text_features
    elif image_features is not None:
      return image_features
    else:
      return np.zeros(Model.VECTOR_SIZE, dtype=np.float32)

  def compute_similarities_to_text(
    self, item_features: FeatureVector, positive_queries: List[str], negative_queries: List[str]
  ) -> List[Tuple[float, int]]:
    positive_features = self.compute_features_for_queries(positive_queries)
    negative_features = self.compute_features_for_queries(negative_queries)

    features = positive_features - negative_features

    similarities = features @ item_features.T
    sorted_similarities = sorted(zip(similarities, range(item_features.shape[0])), key=lambda x: x[0], reverse=True)

    return sorted_similarities
