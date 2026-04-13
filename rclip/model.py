from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import shutil
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError

from rclip.const import IS_MACOS
from rclip.utils import helpers
from rclip.utils.preprocess import preprocess
from rclip.utils.tokenizer import SimpleTokenizer

logging.getLogger("coremltools").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

if TYPE_CHECKING:
  import coremltools as ct
  import onnxruntime as ort

  _SessionType = Union[ort.InferenceSession, ct.models.MLModel, ct.models.CompiledMLModel]

QUERY_WITH_MULTIPLIER_RE = re.compile(r"^(?P<multiplier>(\d+(\.\d+)?|\.\d+|\d+\.)):(?P<query>.+)$")
QueryWithMultiplier = Tuple[float, str]
FeatureVector = npt.NDArray[np.float32]

HF_REPO_ID = "yurijmikhalevich/rclip-models"
MODEL_SUBDIR = "ViT-B-32-256-datacomp_s34b_b86k"
VISUAL_ONNX = "visual.onnx"
TEXTUAL_ONNX = "textual.onnx"
VISUAL_COREML = "visual.mlpackage"
TOKENIZER_VOCAB = "tokenizer/bpe_simple_vocab_16e6.txt.gz"
USE_ONNX_RUNTIME_ON_MACOS_ENV_VAR = "RCLIP_USE_ONNX_ON_MACOS"
RUNTIME_ONNX = "onnx"
RUNTIME_COREML = "coreml"
COREML_VISUAL_BATCH_SIZE = 8


def get_model_dir() -> str:
  return _get_model_dir()


def ensure_compiled_coreml_model(package_path: str) -> str:
  return _ensure_compiled_coreml_model(package_path)


def _get_model_dir() -> str:
  return str(helpers.get_app_datadir())


def _get_model_cache_dir() -> Optional[str]:
  model_cache_dir = helpers.get_model_cache_dir()
  return str(model_cache_dir) if model_cache_dir else None


def _download_onnx_model(filename: str, tqdm_class: Optional[type] = None) -> str:
  model_dir = _get_model_dir()
  expected_path = os.path.join(model_dir, MODEL_SUBDIR, filename)
  if os.path.isfile(expected_path):
    return expected_path

  from huggingface_hub import snapshot_download

  os.makedirs(model_dir, exist_ok=True)
  kwargs = {}
  if tqdm_class is not None:
    kwargs["tqdm_class"] = tqdm_class

  path = snapshot_download(
    repo_id=HF_REPO_ID,
    allow_patterns=f"{MODEL_SUBDIR}/{filename}",
    cache_dir=_get_model_cache_dir(),
    local_dir=model_dir,
    **kwargs,
  )
  return os.path.join(path, MODEL_SUBDIR, filename)


def _download_tokenizer_vocab(tqdm_class: Optional[type] = None) -> str:
  model_dir = _get_model_dir()
  expected_path = os.path.join(model_dir, TOKENIZER_VOCAB)
  if os.path.isfile(expected_path):
    return expected_path

  from huggingface_hub import hf_hub_download

  kwargs = {}
  if tqdm_class is not None:
    kwargs["tqdm_class"] = tqdm_class

  return hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=TOKENIZER_VOCAB,
    cache_dir=_get_model_cache_dir(),
    local_dir=model_dir,
    **kwargs,
  )


def _download_coreml_model(dirname: str, tqdm_class: Optional[type] = None) -> str:
  model_dir = _get_model_dir()
  expected_path = os.path.join(model_dir, MODEL_SUBDIR, dirname)
  if os.path.isdir(expected_path):
    return expected_path

  from huggingface_hub import snapshot_download

  # CoreML compilation fails on the symlinked Hugging Face snapshot cache, so
  # materialize the package into the app data directory first.
  os.makedirs(model_dir, exist_ok=True)
  kwargs = {}
  if tqdm_class is not None:
    kwargs["tqdm_class"] = tqdm_class

  path = snapshot_download(
    repo_id=HF_REPO_ID,
    allow_patterns=f"{MODEL_SUBDIR}/{dirname}/**",
    cache_dir=_get_model_cache_dir(),
    local_dir=model_dir,
    **kwargs,
  )
  package_path = os.path.join(path, MODEL_SUBDIR, dirname)
  _ensure_compiled_coreml_model(package_path)
  return package_path


def _get_compiled_coreml_model_path(package_path: str) -> str:
  base_path = os.path.splitext(package_path)[0]
  return f"{base_path}.mlmodelc"


def _compile_coreml_model(package_path: str, *, force: bool = False) -> str:
  import coremltools as ct

  compiled_path = _get_compiled_coreml_model_path(package_path)
  if os.path.isdir(compiled_path):
    if not force:
      return compiled_path
    shutil.rmtree(compiled_path)

  return ct.models.utils.compile_model(package_path, compiled_path)


def _ensure_compiled_coreml_model(package_path: str) -> str:
  compiled_path = _get_compiled_coreml_model_path(package_path)
  if os.path.isdir(compiled_path):
    return compiled_path
  return _compile_coreml_model(package_path)


def _get_runtime(*, is_visual: bool, for_indexing: bool = False) -> str:
  if not IS_MACOS:
    return RUNTIME_ONNX
  if os.getenv(USE_ONNX_RUNTIME_ON_MACOS_ENV_VAR):
    return RUNTIME_ONNX
  if is_visual and for_indexing:
    return RUNTIME_COREML
  return RUNTIME_ONNX


class Model:
  VECTOR_SIZE = 512

  def __init__(self):
    self._session_text_var: Optional[_SessionType] = None
    self._session_visual_var: Optional[_SessionType] = None
    self._session_visual_index_var: Optional[_SessionType] = None
    self._tokenizer_var: Optional[SimpleTokenizer] = None
    self._preprocess_executor_var: Optional[ThreadPoolExecutor] = None
    self._preprocess_workers = max(1, min(8, os.cpu_count() or 1))

  def ensure_downloaded(self) -> None:
    model_dir = _get_model_dir()

    to_download: List[Tuple[str, Tuple[str, ...], Callable[[Optional[type]], str]]] = []
    model_files = [
      ("visual query model", VISUAL_ONNX, RUNTIME_ONNX),
      ("textual model", TEXTUAL_ONNX, RUNTIME_ONNX),
    ]
    if _get_runtime(is_visual=True, for_indexing=True) == RUNTIME_COREML:
      model_files.append(("visual indexing model", VISUAL_COREML, RUNTIME_COREML))

    for label, filename, runtime in model_files:
      use_coreml = runtime == RUNTIME_COREML
      path_prefix_suffix = "/" if use_coreml else ""
      path_exists = os.path.isdir if use_coreml else os.path.isfile
      download_model = _download_coreml_model if use_coreml else _download_onnx_model
      candidate_filenames = (filename,)
      existing_path = None
      candidate_path = os.path.join(model_dir, MODEL_SUBDIR, filename)
      if path_exists(candidate_path):
        existing_path = candidate_path
      if existing_path is not None:
        if use_coreml:
          _ensure_compiled_coreml_model(existing_path)
        continue
      to_download.append(
        (
          label,
          tuple(f"{MODEL_SUBDIR}/{candidate}{path_prefix_suffix}" for candidate in candidate_filenames),
          lambda tqdm_class, filename=filename, download_model=download_model: download_model(
            filename, tqdm_class=tqdm_class
          ),
        )
      )

    if not os.path.isfile(os.path.join(model_dir, TOKENIZER_VOCAB)):
      to_download.append(
        (
          "tokenizer",
          (TOKENIZER_VOCAB,),
          lambda tqdm_class: _download_tokenizer_vocab(tqdm_class=tqdm_class),
        )
      )

    if not to_download:
      return

    from huggingface_hub import HfApi
    from tqdm import tqdm as tqdm_cls

    from rclip.utils.download_progress import AggregatedProgressBar

    # Fetch file sizes so the progress bar total is known from the start.
    repo_info = HfApi().repo_info(HF_REPO_ID, files_metadata=True)
    size_by_file = {repo_file.rfilename: repo_file.size or 0 for repo_file in (repo_info.siblings or [])}

    selected_prefixes = []
    for _download_label, prefix_group, _download_function in to_download:
      selected_prefix = prefix_group[0]
      for prefix in prefix_group:
        if any(path.startswith(prefix) for path in size_by_file):
          selected_prefix = prefix
          break
      selected_prefixes.append(selected_prefix)

    total_bytes = sum(
      size for path, size in size_by_file.items() if any(path.startswith(prefix) for prefix in selected_prefixes)
    )

    shared_bar = tqdm_cls(total=total_bytes, desc="Downloading model", unit="B", unit_scale=True)
    AggregatedProgressBar.shared_bar = shared_bar
    shared_bar.set_description("Downloading the model")
    try:
      for _download_label, _prefix_group, download_function in to_download:
        download_function(AggregatedProgressBar)
    finally:
      AggregatedProgressBar.shared_bar = None
      shared_bar.close()

  @property
  def _tokenizer(self) -> SimpleTokenizer:
    if not self._tokenizer_var:
      self._tokenizer_var = SimpleTokenizer(bpe_path=_download_tokenizer_vocab())
    return self._tokenizer_var

  def _load_session(self, onnx_filename: str, *, runtime: str, coreml_dirname: Optional[str] = None) -> "_SessionType":
    if runtime == RUNTIME_COREML:
      import coremltools as ct

      assert coreml_dirname is not None
      package_path = _download_coreml_model(coreml_dirname)
      compiled_path = _ensure_compiled_coreml_model(package_path)
      try:
        return ct.models.CompiledMLModel(compiled_path, compute_units=ct.ComputeUnit.ALL)
      except Exception:
        compiled_path = _compile_coreml_model(package_path, force=True)
        return ct.models.CompiledMLModel(compiled_path, compute_units=ct.ComputeUnit.ALL)

    import onnxruntime as ort

    path = _download_onnx_model(onnx_filename)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

  def _run_session(
    self,
    session: "_SessionType",
    batch: npt.NDArray[np.generic],
    *,
    runtime: str,
    coreml_input_dtype: npt.DTypeLike | None = None,
    coreml_batch_size: int = 1,
  ) -> npt.NDArray[np.float32]:
    if runtime == RUNTIME_COREML:
      from coremltools.models import CompiledMLModel, MLModel

      assert isinstance(session, (MLModel, CompiledMLModel))
      if coreml_input_dtype is not None:
        batch = batch.astype(coreml_input_dtype)
      outputs: list[npt.NDArray[np.float32]] = []
      for start in range(0, batch.shape[0], coreml_batch_size):
        chunk = batch[start : start + coreml_batch_size]
        actual_size = chunk.shape[0]
        if actual_size < coreml_batch_size:
          chunk = np.concatenate([chunk, np.repeat(chunk[-1:], coreml_batch_size - actual_size, axis=0)], axis=0)
        result = session.predict({"input": chunk})
        outputs.append(np.array(result["output"], dtype=np.float32)[:actual_size])
      return np.concatenate(outputs, axis=0)

    from onnxruntime import InferenceSession

    assert isinstance(session, InferenceSession)
    input_type = session.get_inputs()[0].type
    if input_type == "tensor(float16)":
      batch = batch.astype(np.float16)
    (features,) = session.run(None, {"input": batch})
    return np.asarray(features, dtype=np.float32)

  @property
  def _session_text(self):
    if not self._session_text_var:
      self._session_text_var = self._load_session(TEXTUAL_ONNX, runtime=RUNTIME_ONNX)
    return self._session_text_var

  @property
  def _session_visual(self):
    if not self._session_visual_var:
      self._session_visual_var = self._load_session(VISUAL_ONNX, runtime=RUNTIME_ONNX)
    return self._session_visual_var

  @property
  def _session_visual_index(self):
    runtime = _get_runtime(is_visual=True, for_indexing=True)
    if runtime == RUNTIME_ONNX:
      return self._session_visual
    if not self._session_visual_index_var:
      self._session_visual_index_var = self._load_session(VISUAL_ONNX, runtime=runtime, coreml_dirname=VISUAL_COREML)
    return self._session_visual_index_var

  def _run_visual(self, batch: npt.NDArray[np.float32], *, for_indexing: bool = False) -> npt.NDArray[np.float32]:
    runtime = _get_runtime(is_visual=True, for_indexing=for_indexing)
    session = self._session_visual_index if for_indexing else self._session_visual
    return self._run_session(session, batch, runtime=runtime, coreml_batch_size=COREML_VISUAL_BATCH_SIZE)

  def _run_textual(self, tokens: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    return self._run_session(self._session_text, tokens, runtime=RUNTIME_ONNX)

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
      file_multipliers, file_paths = cast(Tuple[Tuple[float], Tuple[str]], zip(*(files))) if files else ((), ())
      url_multipliers, url_paths = cast(Tuple[Tuple[float], Tuple[str]], zip(*(urls))) if urls else ((), ())
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
    sorted_similarities = sorted(
      zip(similarities, range(item_features.shape[0])),
      key=lambda similarity_with_index: similarity_with_index[0],
      reverse=True,
    )

    return sorted_similarities
