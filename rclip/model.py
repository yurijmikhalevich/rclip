import re
from typing import List, Tuple, Optional, cast
import sys

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from rclip.utils import helpers
from rclip.utils.preprocess import preprocess
from rclip.utils.tokenizer import SimpleTokenizer

QUERY_WITH_MULTIPLIER_RE = re.compile(r"^(?P<multiplier>(\d+(\.\d+)?|\.\d+|\d+\.)):(?P<query>.+)$")
QueryWithMultiplier = Tuple[float, str]
FeatureVector = npt.NDArray[np.float32]

_HF_REPO_ID = "Marqo/onnx-open_clip-ViT-B-32-quickgelu"
_TEXTUAL_ONNX_FILE = "onnx32-open_clip-ViT-B-32-quickgelu-openai-textual.onnx"
_VISUAL_ONNX_FILE = "onnx32-open_clip-ViT-B-32-quickgelu-openai-visual.onnx"


def _get_providers(device: str) -> List[str]:
  if device == "coreml":
    return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
  return ["CPUExecutionProvider"]


def _download_model(filename: str) -> str:
  return hf_hub_download(repo_id=_HF_REPO_ID, filename=filename)


class Model:
  VECTOR_SIZE = 512

  def __init__(self, device: str = "cpu"):
    self._providers = _get_providers(device)

    self._session_text_var: Optional[ort.InferenceSession] = None
    self._session_visual_var: Optional[ort.InferenceSession] = None
    self._tokenizer_var: Optional[SimpleTokenizer] = None

  @property
  def _tokenizer(self) -> SimpleTokenizer:
    if not self._tokenizer_var:
      self._tokenizer_var = SimpleTokenizer()
    return self._tokenizer_var

  @property
  def _session_text(self) -> ort.InferenceSession:
    if not self._session_text_var:
      path = _download_model(_TEXTUAL_ONNX_FILE)
      self._session_text_var = ort.InferenceSession(path, providers=self._providers)
    return self._session_text_var

  @property
  def _session_visual(self) -> ort.InferenceSession:
    if not self._session_visual_var:
      path = _download_model(_VISUAL_ONNX_FILE)
      self._session_visual_var = ort.InferenceSession(path, providers=self._providers)
    return self._session_visual_var

  def compute_image_features(self, images: List[Image.Image]) -> npt.NDArray[np.float32]:
    batch = np.stack([preprocess(img) for img in images])
    (image_features,) = self._session_visual.run(None, {"input": batch})
    image_features = cast(npt.NDArray[np.float32], image_features)
    image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
    return image_features

  def compute_text_features(self, text: List[str]) -> npt.NDArray[np.float32]:
    tokens = self._tokenizer(text)
    (text_features,) = self._session_text.run(None, {"input": tokens})
    text_features = cast(npt.NDArray[np.float32], text_features)
    text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
    return text_features

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
