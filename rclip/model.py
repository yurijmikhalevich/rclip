from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError

from rclip import model_download
from rclip.utils import helpers
from rclip.utils.preprocess import preprocess
from rclip.utils.tokenizer import SimpleTokenizer

logging.getLogger("coremltools").setLevel(logging.ERROR)

QUERY_WITH_MULTIPLIER_RE = re.compile(r"^(?P<multiplier>(\d+(\.\d+)?|\.\d+|\d+\.)):(?P<query>.+)$")
QueryWithMultiplier = Tuple[float, str]
FeatureVector = npt.NDArray[np.float32]


class Model:
  VECTOR_SIZE = 512

  def __init__(self):
    self._session_text_var: Optional[Any] = None
    self._session_visual_var: Optional[Any] = None
    self._session_visual_index_var: Optional[Any] = None
    self._tokenizer_var: Optional[SimpleTokenizer] = None
    self._preprocess_executor_var: Optional[ThreadPoolExecutor] = None
    self._preprocess_workers = max(1, min(8, os.cpu_count() or 1))

  def ensure_downloaded(self) -> None:
    model_download.ensure_downloaded()

  def unload(self) -> None:
    for session_attr in ("_session_text_var", "_session_visual_var", "_session_visual_index_var"):
      session = getattr(self, session_attr)
      close = getattr(session, "close", None)
      if callable(close):
        close()
      setattr(self, session_attr, None)

    if self._preprocess_executor_var is not None:
      self._preprocess_executor_var.shutdown()
      self._preprocess_executor_var = None

    self._tokenizer_var = None

  @property
  def _tokenizer(self) -> SimpleTokenizer:
    if self._tokenizer_var is None:
      self._tokenizer_var = SimpleTokenizer(bpe_path=model_download.download_tokenizer_vocab())
    return self._tokenizer_var

  def _run_session(
    self,
    session: Any,
    batch: npt.NDArray[np.generic],
    *,
    coreml_batch_size: int = 1,
  ) -> npt.NDArray[np.float32]:
    if hasattr(session, "predict"):
      outputs: list[npt.NDArray[np.float32]] = []
      for start in range(0, batch.shape[0], coreml_batch_size):
        chunk = batch[start : start + coreml_batch_size]
        actual_size = chunk.shape[0]
        if actual_size < coreml_batch_size:
          chunk = np.concatenate([chunk, np.repeat(chunk[-1:], coreml_batch_size - actual_size, axis=0)], axis=0)
        result = session.predict({"input": chunk})
        outputs.append(np.array(result["output"], dtype=np.float32)[:actual_size])
      return np.concatenate(outputs, axis=0)

    input_type = session.get_inputs()[0].type
    if input_type == "tensor(float16)":
      batch = batch.astype(np.float16)
    (features,) = session.run(None, {"input": batch})
    return np.asarray(features, dtype=np.float32)

  @property
  def _session_text(self) -> Any:
    if self._session_text_var is None:
      self._session_text_var = model_download.load_text_session()
    return self._session_text_var

  @property
  def _session_visual(self) -> Any:
    if self._session_visual_var is None:
      self._session_visual_var = model_download.load_visual_query_session()
    return self._session_visual_var

  @property
  def _session_visual_index(self) -> Any:
    if not model_download.use_coreml_for_visual_index():
      return self._session_visual
    if self._session_visual_index_var is None:
      self._session_visual_index_var = model_download.load_visual_index_session()
    return self._session_visual_index_var

  def _run_visual(self, batch: npt.NDArray[np.float32], *, for_indexing: bool = False) -> npt.NDArray[np.float32]:
    session = self._session_visual_index if for_indexing else self._session_visual
    return self._run_session(session, batch, coreml_batch_size=model_download.COREML_VISUAL_BATCH_SIZE)

  def _run_textual(self, tokens: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    return self._run_session(self._session_text, tokens)

  def compute_image_features(self, images: List[Image.Image], *, for_indexing: bool = False) -> npt.NDArray[np.float32]:
    if len(images) < 2 or self._preprocess_workers == 1:
      batch = np.stack([preprocess(img) for img in images])
    else:
      if self._preprocess_executor_var is None:
        self._preprocess_executor_var = ThreadPoolExecutor(max_workers=self._preprocess_workers)
      batch = np.stack(list(self._preprocess_executor_var.map(preprocess, images)))
    image_features = self._run_visual(batch, for_indexing=for_indexing)
    image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
    return image_features

  def compute_text_features(self, text: List[str]) -> npt.NDArray[np.float32]:
    tokens = self._tokenizer(text)
    text_features = self._run_textual(tokens)
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

    if files or urls:
      file_multipliers, file_paths = zip(*files) if files else ((), ())
      url_multipliers, url_paths = zip(*urls) if urls else ((), ())
      try:
        images = [helpers.download_image(url_path) for url_path in url_paths] + [
          helpers.read_image(file_path) for file_path in file_paths
        ]
      except FileNotFoundError as error:
        print(f'File "{error.filename}" not found. Check if you have typos in the filename.')
        import sys

        sys.exit(1)
      except UnidentifiedImageError as error:
        print(f'File "{error.filename}" is not an image. You can only use image files or text as queries.')
        import sys

        sys.exit(1)
      image_multipliers = np.array(url_multipliers + file_multipliers)
      image_features = np.add.reduce(self.compute_image_features(images) * image_multipliers.reshape(-1, 1))

    if phrases:
      phrase_multipliers, phrase_queries = zip(*phrases)
      phrase_multipliers_np = np.array(phrase_multipliers).reshape(-1, 1)
      text_features = np.add.reduce(self.compute_text_features([*phrase_queries]) * phrase_multipliers_np)

    if text_features is None:
      if image_features is not None:
        return image_features
      return np.zeros(Model.VECTOR_SIZE, dtype=np.float32)
    if image_features is None:
      return text_features
    return text_features + image_features

  def compute_similarities_to_text(
    self, item_features: FeatureVector, positive_queries: List[str], negative_queries: List[str]
  ) -> List[Tuple[float, int]]:
    positive_features = self.compute_features_for_queries(positive_queries)
    negative_features = self.compute_features_for_queries(negative_queries)

    features = positive_features - negative_features

    similarities = features @ item_features.T
    sorted_similarities = sorted(
      zip(similarities, range(item_features.shape[0])),
      key=lambda similarity_with_index: similarity_with_index[0],
      reverse=True,
    )

    return sorted_similarities
