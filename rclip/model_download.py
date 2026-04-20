import importlib
import io
import logging
import os
import shutil
import sys
import tempfile
from typing import Callable, Optional

from rclip.const import IS_MACOS
from rclip.utils import helpers

logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

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


def _filter_onnxruntime_stderr(stderr_output: str) -> str:
  filtered_lines = []
  for line in stderr_output.splitlines(True):
    if (
      "[W:onnxruntime" in line
      and "device_discovery.cc" in line
      and "DiscoverDevicesForPlatform" in line
      and "GPU device discovery failed" in line
    ):
      continue
    filtered_lines.append(line)
  return "".join(filtered_lines)


def _import_onnxruntime():
  if sys.platform != "linux" or "onnxruntime" in sys.modules:
    return importlib.import_module("onnxruntime")

  try:
    stderr_fd = sys.stderr.fileno()
  except (AttributeError, io.UnsupportedOperation):
    return importlib.import_module("onnxruntime")

  with tempfile.TemporaryFile(mode="w+b") as stderr_capture:
    saved_stderr_fd = os.dup(stderr_fd)
    try:
      os.dup2(stderr_capture.fileno(), stderr_fd)
      ort = importlib.import_module("onnxruntime")
    finally:
      os.dup2(saved_stderr_fd, stderr_fd)
      os.close(saved_stderr_fd)

    stderr_capture.seek(0)
    filtered_stderr = _filter_onnxruntime_stderr(stderr_capture.read().decode(errors="replace"))

  if filtered_stderr:
    sys.stderr.write(filtered_stderr)
    sys.stderr.flush()

  return ort


def get_model_dir() -> str:
  return str(helpers.get_app_datadir())


def _get_model_cache_dir() -> Optional[str]:
  model_cache_dir = helpers.get_model_cache_dir()
  return str(model_cache_dir) if model_cache_dir else None


def download_onnx_model(filename: str, tqdm_class: Optional[type] = None) -> str:
  model_dir = get_model_dir()
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


def download_visual_query_model(tqdm_class: Optional[type] = None) -> str:
  return download_onnx_model(VISUAL_ONNX, tqdm_class=tqdm_class)


def download_textual_model(tqdm_class: Optional[type] = None) -> str:
  return download_onnx_model(TEXTUAL_ONNX, tqdm_class=tqdm_class)


def download_tokenizer_vocab(tqdm_class: Optional[type] = None) -> str:
  model_dir = get_model_dir()
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


def download_coreml_model(dirname: str, tqdm_class: Optional[type] = None) -> str:
  model_dir = get_model_dir()
  expected_path = os.path.join(model_dir, MODEL_SUBDIR, dirname)
  if os.path.isdir(expected_path):
    return expected_path

  from huggingface_hub import snapshot_download

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
  ensure_compiled_coreml_model(package_path)
  return package_path


def download_visual_index_model_package(tqdm_class: Optional[type] = None) -> str:
  return download_coreml_model(VISUAL_COREML, tqdm_class=tqdm_class)


def _get_compiled_coreml_model_path(package_path: str) -> str:
  base_path = os.path.splitext(package_path)[0]
  return f"{base_path}.mlmodelc"


def compile_coreml_model(package_path: str, *, force: bool = False) -> str:
  import coremltools as ct

  compiled_path = _get_compiled_coreml_model_path(package_path)
  if os.path.isdir(compiled_path):
    if not force:
      return compiled_path
    shutil.rmtree(compiled_path)

  return ct.models.utils.compile_model(package_path, compiled_path)


def ensure_compiled_coreml_model(package_path: str) -> str:
  compiled_path = _get_compiled_coreml_model_path(package_path)
  if os.path.isdir(compiled_path):
    return compiled_path
  return compile_coreml_model(package_path)


def get_runtime(*, is_visual: bool, for_indexing: bool = False) -> str:
  if not IS_MACOS:
    return RUNTIME_ONNX
  if os.getenv(USE_ONNX_RUNTIME_ON_MACOS_ENV_VAR):
    return RUNTIME_ONNX
  if is_visual and for_indexing:
    return RUNTIME_COREML
  return RUNTIME_ONNX


def use_coreml_for_visual_index() -> bool:
  return get_runtime(is_visual=True, for_indexing=True) == RUNTIME_COREML


def ensure_downloaded() -> None:
  model_dir = get_model_dir()
  to_download: list[tuple[str, Callable[[type | None], str]]] = []

  visual_query_path = os.path.join(model_dir, MODEL_SUBDIR, VISUAL_ONNX)
  if not os.path.isfile(visual_query_path):
    to_download.append(
      (f"{MODEL_SUBDIR}/{VISUAL_ONNX}", lambda tqdm_class: download_visual_query_model(tqdm_class=tqdm_class))
    )

  textual_path = os.path.join(model_dir, MODEL_SUBDIR, TEXTUAL_ONNX)
  if not os.path.isfile(textual_path):
    to_download.append(
      (f"{MODEL_SUBDIR}/{TEXTUAL_ONNX}", lambda tqdm_class: download_textual_model(tqdm_class=tqdm_class))
    )

  if use_coreml_for_visual_index():
    visual_index_path = os.path.join(model_dir, MODEL_SUBDIR, VISUAL_COREML)
    if os.path.isdir(visual_index_path):
      ensure_compiled_coreml_model(visual_index_path)
    else:
      to_download.append(
        (
          f"{MODEL_SUBDIR}/{VISUAL_COREML}/",
          lambda tqdm_class: download_visual_index_model_package(tqdm_class=tqdm_class),
        )
      )

  tokenizer_path = os.path.join(model_dir, TOKENIZER_VOCAB)
  if not os.path.isfile(tokenizer_path):
    to_download.append((TOKENIZER_VOCAB, lambda tqdm_class: download_tokenizer_vocab(tqdm_class=tqdm_class)))

  if not to_download:
    _import_onnxruntime()
    return

  from huggingface_hub import HfApi
  from tqdm import tqdm as tqdm_cls

  from rclip.utils.download_progress import AggregatedProgressBar

  repo_info = HfApi().repo_info(HF_REPO_ID, files_metadata=True)
  size_by_file = {repo_file.rfilename: repo_file.size or 0 for repo_file in (repo_info.siblings or [])}
  selected_prefixes = [prefix for prefix, _download_function in to_download]
  total_bytes = sum(
    size for path, size in size_by_file.items() if any(path.startswith(prefix) for prefix in selected_prefixes)
  )

  shared_bar = tqdm_cls(total=total_bytes, desc="Downloading model", unit="B", unit_scale=True)
  AggregatedProgressBar.shared_bar = shared_bar
  shared_bar.set_description("Downloading the model")
  try:
    for _prefix, download_function in to_download:
      download_function(AggregatedProgressBar)
  finally:
    AggregatedProgressBar.shared_bar = None
    shared_bar.close()

  _import_onnxruntime()


def _load_onnx_session(model_path: str):
  ort = _import_onnxruntime()

  session_options = ort.SessionOptions()
  sched_getaffinity = getattr(os, "sched_getaffinity", None)
  if sched_getaffinity is not None:
    try:
      session_options.intra_op_num_threads = max(1, len(sched_getaffinity(0)))
    except OSError:
      session_options.intra_op_num_threads = max(1, os.cpu_count() or 1)
  else:
    session_options.intra_op_num_threads = max(1, os.cpu_count() or 1)

  return ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])


def load_text_session():
  return _load_onnx_session(download_textual_model())


def load_visual_query_session():
  return _load_onnx_session(download_visual_query_model())


def load_visual_index_session():
  if not use_coreml_for_visual_index():
    return load_visual_query_session()

  import coremltools as ct

  package_path = download_visual_index_model_package()
  compiled_path = ensure_compiled_coreml_model(package_path)
  try:
    return ct.models.CompiledMLModel(compiled_path, compute_units=ct.ComputeUnit.ALL)
  except Exception:
    compiled_path = compile_coreml_model(package_path, force=True)
    return ct.models.CompiledMLModel(compiled_path, compute_units=ct.ComputeUnit.ALL)
