from pathlib import Path
import sys
import tempfile
import types

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

import rclip.model as model_module
from rclip.model import Model

FeatureBatch = npt.NDArray[np.float32]
TokenBatch = npt.NDArray[np.int64]


class FakeTokenizer:
  def __init__(self, bpe_path: str):
    self.bpe_path = bpe_path

  def __call__(self, text: list[str]) -> TokenBatch:
    return np.arange(len(text) * 4, dtype=np.int64).reshape(len(text), 4)


class FakeInferenceSession:
  created: list["FakeInferenceSession"] = []

  def __init__(self, path: str, providers: list[str]):
    self.path = path
    self.providers = providers
    type(self).created.append(self)

  def run(self, _output_names: object, inputs: dict[str, npt.NDArray[np.generic]]) -> tuple[FeatureBatch]:
    batch = inputs["input"]
    offset = 0.0 if self.path.endswith("textual.onnx") else 1000.0
    base = np.arange(1, Model.VECTOR_SIZE + 1, dtype=np.float32)
    features = np.stack([base + offset + i for i in range(batch.shape[0])]).astype(np.float32)
    return (features,)

  def get_inputs(self) -> list[object]:
    return [types.SimpleNamespace(type="tensor(float)")]


def _fake_download_onnx_model(filename: str) -> str:
  return f"/models/{filename}"


def test_download_coreml_model_materializes_real_package(monkeypatch: pytest.MonkeyPatch):
  def fake_get_app_datadir() -> Path:
    return Path("/tmp/rclip-datadir")

  def fake_ensure_compiled_coreml_model(path: str) -> str:
    compiled_paths.append(path)
    return path

  monkeypatch.setattr(model_module.helpers, "get_app_datadir", fake_get_app_datadir)
  compiled_paths: list[str] = []
  monkeypatch.setattr(model_module, "_ensure_compiled_coreml_model", fake_ensure_compiled_coreml_model)

  calls: list[dict[str, str | None]] = []

  def fake_snapshot_download(
    *, repo_id: str, allow_patterns: str, cache_dir: str | None = None, local_dir: str, **_kwargs: object
  ) -> str:
    calls.append({"repo_id": repo_id, "allow_patterns": allow_patterns, "cache_dir": cache_dir, "local_dir": local_dir})
    return local_dir

  fake_huggingface_hub = types.ModuleType("huggingface_hub")
  setattr(fake_huggingface_hub, "snapshot_download", fake_snapshot_download)
  monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)

  download_coreml_model = getattr(model_module, "_download_coreml_model")
  path = download_coreml_model("visual.mlpackage")

  assert Path(path) == Path("/tmp/rclip-datadir/ViT-B-32-256-datacomp_s34b_b86k/visual.mlpackage")
  assert calls == [
    {
      "repo_id": "yurijmikhalevich/rclip-models",
      "allow_patterns": "ViT-B-32-256-datacomp_s34b_b86k/visual.mlpackage/**",
      "cache_dir": None,
      "local_dir": str(Path("/tmp/rclip-datadir")),
    }
  ]
  assert compiled_paths == [str(Path("/tmp/rclip-datadir/ViT-B-32-256-datacomp_s34b_b86k/visual.mlpackage"))]


def test_load_session_uses_compiled_coreml_model(monkeypatch: pytest.MonkeyPatch):
  compiled_model_calls: list[tuple[str, object]] = []

  def fake_download_coreml_model(dirname: str) -> str:
    return f"/models/{dirname}"

  def fake_ensure_compiled_coreml_model(path: str) -> str:
    return f"{Path(path).with_suffix('')}.mlmodelc"

  class FakeCompiledMLModel:
    def __init__(self, path: str, compute_units: object):
      compiled_model_calls.append((path, compute_units))
      self.path = path

  fake_coremltools = types.ModuleType("coremltools")
  setattr(fake_coremltools, "ComputeUnit", types.SimpleNamespace(ALL="all"))
  setattr(
    fake_coremltools,
    "models",
    types.SimpleNamespace(CompiledMLModel=FakeCompiledMLModel, MLModel=object),
  )

  monkeypatch.setitem(sys.modules, "coremltools", fake_coremltools)
  monkeypatch.setattr(model_module, "IS_MACOS", True)
  monkeypatch.delenv("RCLIP_USE_ONNX_ON_MACOS", raising=False)
  monkeypatch.setattr(model_module, "_download_coreml_model", fake_download_coreml_model)
  monkeypatch.setattr(model_module, "_ensure_compiled_coreml_model", fake_ensure_compiled_coreml_model)

  model = Model()
  session = getattr(model, "_load_session")("visual.onnx", runtime="coreml", coreml_dirname="visual.mlpackage")

  assert isinstance(session, FakeCompiledMLModel)
  assert compiled_model_calls == [(str(Path("/models/visual.mlmodelc")), "all")]


def test_ensure_downloaded_compiles_existing_coreml_packages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  data_dir = tmp_path / "rclip-datadir"
  model_dir = data_dir / "ViT-B-32-256-datacomp_s34b_b86k"

  def fake_get_app_datadir() -> Path:
    return data_dir

  existing_paths = {
    model_dir / "visual.onnx",
    model_dir / "textual.onnx",
    model_dir / "visual.mlpackage",
    data_dir / "tokenizer/bpe_simple_vocab_16e6.txt.gz",
  }
  compiled_paths: list[str] = []

  def fake_isdir(path: str) -> bool:
    return Path(path) in existing_paths

  def fake_isfile(path: str) -> bool:
    return Path(path) in existing_paths

  def fake_ensure_compiled_coreml_model(path: str) -> str:
    compiled_paths.append(path)
    return f"{Path(path).with_suffix('')}.mlmodelc"

  monkeypatch.setattr(model_module.helpers, "get_app_datadir", fake_get_app_datadir)
  monkeypatch.setattr(model_module, "IS_MACOS", True)
  monkeypatch.delenv("RCLIP_USE_ONNX_ON_MACOS", raising=False)
  monkeypatch.setattr(model_module.os.path, "isdir", fake_isdir)
  monkeypatch.setattr(model_module.os.path, "isfile", fake_isfile)
  monkeypatch.setattr(model_module, "_ensure_compiled_coreml_model", fake_ensure_compiled_coreml_model)

  Model().ensure_downloaded(for_indexing=True)

  assert compiled_paths == [str(model_dir / "visual.mlpackage")]


def test_ensure_downloaded_uses_matching_downloader_for_each_runtime(monkeypatch: pytest.MonkeyPatch):
  def fake_get_app_datadir() -> Path:
    return Path("/tmp/rclip-datadir")

  downloaded: list[tuple[str, str]] = []

  def fake_isdir(_path: str) -> bool:
    return False

  def fake_isfile(_path: str) -> bool:
    return False

  def fake_download_onnx_model(filename: str, tqdm_class: type | None = None) -> str:
    downloaded.append(("onnx", filename))
    return f"/models/{filename}"

  def fake_download_coreml_model(dirname: str, tqdm_class: type | None = None) -> str:
    downloaded.append(("coreml", dirname))
    return f"/models/{dirname}"

  class FakeRepoFile:
    def __init__(self, path: str):
      self.rfilename = path
      self.size = 1

  class FakeHfApi:
    def repo_info(self, _repo_id: str, files_metadata: bool = False):
      assert files_metadata is True
      return types.SimpleNamespace(
        siblings=[
          FakeRepoFile("ViT-B-32-256-datacomp_s34b_b86k/visual.onnx"),
          FakeRepoFile("ViT-B-32-256-datacomp_s34b_b86k/textual.onnx"),
          FakeRepoFile("ViT-B-32-256-datacomp_s34b_b86k/visual.mlpackage/Manifest.json"),
          FakeRepoFile("tokenizer/bpe_simple_vocab_16e6.txt.gz"),
        ]
      )

  class FakeTqdm:
    def __init__(self, total: int, desc: str, unit: str, unit_scale: bool):
      self.total = total
      self.desc = desc
      self.unit = unit
      self.unit_scale = unit_scale

    def set_description(self, desc: str) -> None:
      self.desc = desc

    def close(self) -> None:
      pass

  fake_huggingface_hub = types.ModuleType("huggingface_hub")
  setattr(fake_huggingface_hub, "HfApi", FakeHfApi)
  monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)

  fake_tqdm_module = types.ModuleType("tqdm")
  setattr(fake_tqdm_module, "tqdm", FakeTqdm)
  monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm_module)

  fake_download_progress_module = types.ModuleType("rclip.utils.download_progress")

  class FakeAggregatedProgressBar:
    shared_bar = None

  setattr(fake_download_progress_module, "AggregatedProgressBar", FakeAggregatedProgressBar)
  monkeypatch.setitem(sys.modules, "rclip.utils.download_progress", fake_download_progress_module)

  monkeypatch.setattr(model_module.helpers, "get_app_datadir", fake_get_app_datadir)
  monkeypatch.setattr(model_module, "IS_MACOS", True)
  monkeypatch.delenv("RCLIP_USE_ONNX_ON_MACOS", raising=False)
  monkeypatch.setattr(model_module.os.path, "isdir", fake_isdir)
  monkeypatch.setattr(model_module.os.path, "isfile", fake_isfile)
  monkeypatch.setattr(model_module, "_download_onnx_model", fake_download_onnx_model)
  monkeypatch.setattr(model_module, "_download_coreml_model", fake_download_coreml_model)

  def fake_download_tokenizer_vocab(tqdm_class: type | None = None) -> str:
    return "/models/tokenizer.gz"

  monkeypatch.setattr(model_module, "_download_tokenizer_vocab", fake_download_tokenizer_vocab)

  Model().ensure_downloaded(for_indexing=True)

  assert downloaded == [
    ("onnx", "visual.onnx"),
    ("onnx", "textual.onnx"),
    ("coreml", "visual.mlpackage"),
  ]


def _fake_preprocess(image: Image.Image) -> FeatureBatch:
  return np.full((3, 256, 256), fill_value=float(image.size[0]), dtype=np.float32)


def _assert_unit_norm(features: FeatureBatch) -> None:
  norms = np.linalg.norm(features, axis=1)
  assert np.allclose(norms, 1.0, atol=1e-5), f"Expected unit norm, got {norms}"


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> list[FakeInferenceSession]:
  FakeInferenceSession.created.clear()
  fake_onnxruntime = types.ModuleType("onnxruntime")
  setattr(fake_onnxruntime, "InferenceSession", FakeInferenceSession)

  monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
  monkeypatch.setattr(model_module, "IS_MACOS", False)
  monkeypatch.setattr(model_module, "SimpleTokenizer", FakeTokenizer)
  monkeypatch.setattr(model_module, "_download_tokenizer_vocab", lambda: "/models/tokenizer.gz")
  monkeypatch.setattr(model_module, "_download_onnx_model", _fake_download_onnx_model)
  monkeypatch.setattr(model_module, "preprocess", _fake_preprocess)

  return FakeInferenceSession.created


def test_extract_query_multiplier():
  assert Model._extract_query_multiplier("1.5:cat") == (1.5, "cat")  # type: ignore
  assert Model._extract_query_multiplier("cat") == (1.0, "cat")  # type: ignore
  assert Model._extract_query_multiplier("1:cat") == (1.0, "cat")  # type: ignore
  assert Model._extract_query_multiplier("0.5:cat") == (0.5, "cat")  # type: ignore
  assert Model._extract_query_multiplier(".5:cat") == (0.5, "cat")  # type: ignore
  assert Model._extract_query_multiplier("1.:cat") == (1.0, "cat")  # type: ignore
  assert Model._extract_query_multiplier("1..:cat") == (1.0, "1..:cat")  # type: ignore
  assert Model._extract_query_multiplier("..:cat") == (1.0, "..:cat")  # type: ignore
  assert Model._extract_query_multiplier("whatever:cat") == (1.0, "whatever:cat")  # type: ignore
  assert (
    Model._extract_query_multiplier("1.5:complex and long query")  # type: ignore
    == (1.5, "complex and long query")
  )


def test_uses_dedicated_model_cache_dir_when_configured(monkeypatch: pytest.MonkeyPatch):
  with tempfile.TemporaryDirectory() as tmp_datadir, tempfile.TemporaryDirectory() as tmp_model_cache_dir:
    monkeypatch.setenv("RCLIP_DATADIR", tmp_datadir)
    monkeypatch.setenv("RCLIP_MODEL_CACHE_DIR", tmp_model_cache_dir)

    cache_dir = Path(tmp_model_cache_dir)
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(
      *, allow_patterns: str | list[str], cache_dir: str | None = None, local_dir: str, **_kwargs: str
    ) -> str:
      calls.append({"allow_patterns": allow_patterns, "cache_dir": cache_dir, "local_dir": local_dir})
      assert cache_dir == str(Path(tmp_model_cache_dir))
      if allow_patterns == "ViT-B-32-256-datacomp_s34b_b86k/visual.onnx":
        downloaded_path = Path(local_dir) / "ViT-B-32-256-datacomp_s34b_b86k/visual.onnx"
        downloaded_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path.touch()
      return tmp_datadir

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    download_onnx_model = getattr(model_module, "_download_onnx_model")
    assert download_onnx_model("visual.onnx") == str(Path(tmp_datadir) / "ViT-B-32-256-datacomp_s34b_b86k/visual.onnx")
    assert cache_dir.exists()
    assert calls == [
      {
        "allow_patterns": "ViT-B-32-256-datacomp_s34b_b86k/visual.onnx",
        "cache_dir": str(Path(tmp_model_cache_dir)),
        "local_dir": tmp_datadir,
      }
    ]


def test_compute_text_features_only_loads_text_session(fake_runtime: list[FakeInferenceSession]):
  model = Model()

  assert getattr(model, "_session_text_var") is None
  assert getattr(model, "_session_visual_var") is None

  features = model.compute_text_features(["cat", "dog", "bird"])

  assert features.shape == (3, Model.VECTOR_SIZE)
  _assert_unit_norm(features)
  assert [session.path for session in fake_runtime] == ["/models/textual.onnx"]
  assert getattr(model, "_session_text_var") is fake_runtime[0]
  assert getattr(model, "_session_visual_var") is None


def test_compute_image_features_only_loads_visual_session(fake_runtime: list[FakeInferenceSession]):
  model = Model()

  assert getattr(model, "_session_text_var") is None
  assert getattr(model, "_session_visual_var") is None

  features = model.compute_image_features([Image.new("RGB", (100, 100), color="red")])

  assert features.shape == (1, Model.VECTOR_SIZE)
  _assert_unit_norm(features)
  assert [session.path for session in fake_runtime] == ["/models/visual.onnx"]
  assert getattr(model, "_session_text_var") is None
  assert getattr(model, "_session_visual_var") is fake_runtime[0]


def test_run_session_casts_float16_inputs_for_onnx(monkeypatch: pytest.MonkeyPatch):
  captured_dtypes: list[np.dtype[np.generic]] = []

  class FakeFp16InferenceSession(FakeInferenceSession):
    def get_inputs(self) -> list[object]:
      return [types.SimpleNamespace(type="tensor(float16)")]

    def run(self, output_names: object, inputs: dict[str, npt.NDArray[np.generic]]) -> tuple[FeatureBatch]:
      captured_dtypes.append(inputs["input"].dtype)
      return super().run(output_names, inputs)

  fake_onnxruntime = types.ModuleType("onnxruntime")
  setattr(fake_onnxruntime, "InferenceSession", FakeFp16InferenceSession)
  monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

  model = Model()
  session = FakeFp16InferenceSession("/models/visual.onnx", providers=["CPUExecutionProvider"])
  batch = np.ones((1, 3, 256, 256), dtype=np.float32)

  features = getattr(model, "_run_session")(session, batch, runtime="onnx")

  assert features.shape == (1, Model.VECTOR_SIZE)
  assert captured_dtypes == [np.dtype(np.float16)]


def test_compute_image_features_uses_separate_visual_session_for_indexing_on_macos(monkeypatch: pytest.MonkeyPatch):
  created_sessions: list[tuple[str, object]] = []

  class FakeCompiledMLModel:
    def __init__(self, path: str, compute_units: object):
      created_sessions.append((path, compute_units))
      self.path = path

    def predict(self, inputs: dict[str, npt.NDArray[np.generic]]) -> dict[str, FeatureBatch]:
      batch = inputs["input"]
      base = np.arange(1, Model.VECTOR_SIZE + 1, dtype=np.float32)
      features = np.stack([base + 1000.0 + i for i in range(batch.shape[0])]).astype(np.float32)
      return {"output": features}

  fake_coremltools = types.ModuleType("coremltools")
  fake_coremltools_models = types.ModuleType("coremltools.models")
  setattr(fake_coremltools, "ComputeUnit", types.SimpleNamespace(ALL="all"))
  setattr(fake_coremltools_models, "CompiledMLModel", FakeCompiledMLModel)
  setattr(fake_coremltools_models, "MLModel", object)
  setattr(fake_coremltools, "models", fake_coremltools_models)

  FakeInferenceSession.created.clear()
  fake_onnxruntime = types.ModuleType("onnxruntime")
  setattr(fake_onnxruntime, "InferenceSession", FakeInferenceSession)

  monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
  monkeypatch.setitem(sys.modules, "coremltools", fake_coremltools)
  monkeypatch.setitem(sys.modules, "coremltools.models", fake_coremltools_models)
  monkeypatch.setattr(model_module, "IS_MACOS", True)
  monkeypatch.delenv("RCLIP_USE_ONNX_ON_MACOS", raising=False)
  monkeypatch.setattr(model_module, "_download_onnx_model", _fake_download_onnx_model)

  def fake_download_coreml_model(dirname: str) -> str:
    return f"/models/{dirname}"

  def fake_ensure_compiled_coreml_model(path: str) -> str:
    return f"{Path(path).with_suffix('')}.mlmodelc"

  monkeypatch.setattr(model_module, "_download_coreml_model", fake_download_coreml_model)
  monkeypatch.setattr(model_module, "_ensure_compiled_coreml_model", fake_ensure_compiled_coreml_model)
  monkeypatch.setattr(model_module, "preprocess", _fake_preprocess)

  model = Model()

  query_features = model.compute_image_features([Image.new("RGB", (64, 64), color="red")])
  indexing_features = model.compute_image_features([Image.new("RGB", (64, 64), color="blue")], for_indexing=True)

  assert query_features.shape == (1, Model.VECTOR_SIZE)
  assert indexing_features.shape == (1, Model.VECTOR_SIZE)
  assert [session.path for session in FakeInferenceSession.created] == ["/models/visual.onnx"]
  assert created_sessions == [(str(Path("/models/visual.mlmodelc")), "all")]


def test_ensure_downloaded_skips_coreml_without_indexing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  data_dir = tmp_path / "rclip-datadir"
  model_dir = data_dir / "ViT-B-32-256-datacomp_s34b_b86k"

  def fake_get_app_datadir() -> Path:
    return data_dir

  existing_paths = {
    model_dir / "visual.onnx",
    model_dir / "textual.onnx",
    data_dir / "tokenizer/bpe_simple_vocab_16e6.txt.gz",
  }
  compiled_paths: list[str] = []

  monkeypatch.setattr(model_module.helpers, "get_app_datadir", fake_get_app_datadir)
  monkeypatch.setattr(model_module, "IS_MACOS", True)
  monkeypatch.delenv("RCLIP_USE_ONNX_ON_MACOS", raising=False)

  def fake_isdir(path: str) -> bool:
    return Path(path) in existing_paths

  def fake_isfile(path: str) -> bool:
    return Path(path) in existing_paths

  def fake_ensure_compiled_coreml_model(path: str) -> str:
    compiled_paths.append(path)
    return path

  monkeypatch.setattr(model_module.os.path, "isdir", fake_isdir)
  monkeypatch.setattr(model_module.os.path, "isfile", fake_isfile)
  monkeypatch.setattr(model_module, "_ensure_compiled_coreml_model", fake_ensure_compiled_coreml_model)

  Model().ensure_downloaded(for_indexing=False)

  assert compiled_paths == []


def test_text_then_image_loads_sessions_lazily(fake_runtime: list[FakeInferenceSession]):
  model = Model()

  model.compute_text_features(["cat"])
  text_session = getattr(model, "_session_text_var")

  model.compute_image_features([Image.new("RGB", (64, 64), color="red")])

  assert [session.path for session in fake_runtime] == ["/models/textual.onnx", "/models/visual.onnx"]
  assert getattr(model, "_session_text_var") is text_session
  assert getattr(model, "_session_visual_var") is fake_runtime[1]

  model.compute_text_features(["dog"])

  assert len(fake_runtime) == 2
  assert getattr(model, "_session_text_var") is text_session


def test_image_then_text_loads_sessions_lazily(fake_runtime: list[FakeInferenceSession]):
  model = Model()

  model.compute_image_features([Image.new("RGB", (64, 64), color="red")])
  visual_session = getattr(model, "_session_visual_var")

  model.compute_text_features(["cat"])

  assert [session.path for session in fake_runtime] == ["/models/visual.onnx", "/models/textual.onnx"]
  assert getattr(model, "_session_visual_var") is visual_session
  assert getattr(model, "_session_text_var") is fake_runtime[1]

  model.compute_image_features([Image.new("RGB", (32, 32), color="blue")])

  assert len(fake_runtime) == 2
  assert getattr(model, "_session_visual_var") is visual_session
