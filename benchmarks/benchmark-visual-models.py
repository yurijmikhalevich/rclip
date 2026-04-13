#!/usr/bin/env python3

import argparse
import importlib.util
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

from rclip import db, model
from rclip.utils.preprocess import preprocess

FeatureBatch = npt.NDArray[np.float32]
ClassIdBatch = npt.NDArray[np.str_]

DEFAULT_DATASET_DIR = "/Users/yurij/datasets/imagenet_1k/sample5k"
COREML_BATCH_SIZE = model.COREML_VISUAL_BATCH_SIZE


def _normalize(features: FeatureBatch) -> FeatureBatch:
  return features / np.linalg.norm(features, axis=-1, keepdims=True)


def _load_imagenet_classes(dataset_root: Path) -> dict[str, str]:
  classes_path = dataset_root / "classes.py"
  spec = importlib.util.spec_from_file_location("imagenet_classes", classes_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load ImageNet classes from {classes_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return dict(module.IMAGENET2012_CLASSES)


def _iter_image_files(dataset_dir: Path) -> list[Path]:
  return sorted(path for path in dataset_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg"})


def _run_onnx_visual(session: ort.InferenceSession, batch: FeatureBatch) -> FeatureBatch:
  (features,) = session.run(None, {"input": batch})
  return np.asarray(features, dtype=np.float32)


def _run_coreml_visual(session: ct.models.CompiledMLModel, batch: FeatureBatch) -> FeatureBatch:
  actual_size = batch.shape[0]
  if actual_size < COREML_BATCH_SIZE:
    batch = np.concatenate([batch, np.repeat(batch[-1:], COREML_BATCH_SIZE - actual_size, axis=0)], axis=0)
  result = session.predict({"input": batch})
  return np.asarray(result["output"], dtype=np.float32)[:actual_size]


def _compute_text_features(class_map: dict[str, str]) -> tuple[ClassIdBatch, FeatureBatch]:
  model_instance = model.Model()
  ids, descriptions = zip(*class_map.items())
  text_features = model_instance.compute_text_features([f"photo of {description}" for description in descriptions])
  return np.asarray(ids, dtype=np.str_), text_features


def _compute_accuracy(
  image_features: FeatureBatch, image_files: list[Path], class_ids: ClassIdBatch, text_features: FeatureBatch
) -> tuple[float, float]:
  similarities = image_features @ text_features.T
  ordered_predicted_classes = np.argsort(similarities, axis=1)
  targets = np.array([path.name.split("_")[0] for path in image_files])
  top1 = np.mean(targets == class_ids[ordered_predicted_classes[:, -1]])
  top5 = np.mean(np.any(targets.reshape(-1, 1) == class_ids[ordered_predicted_classes[:, -5:]], axis=1))
  return float(top1), float(top5)


def _compute_visual_features(image_files: list[Path]) -> tuple[dict[str, FeatureBatch], dict[str, float]]:
  model_dir = Path(model.get_model_dir()) / "ViT-B-32-256-datacomp_s34b_b86k"
  onnx_path = model_dir / "visual.onnx"
  coreml_compiled_path = model.ensure_compiled_coreml_model(str(model_dir / "visual.mlpackage"))

  onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  coreml_session = ct.models.CompiledMLModel(coreml_compiled_path, compute_units=ct.ComputeUnit.ALL)

  onnx_parts: list[FeatureBatch] = []
  coreml_parts: list[FeatureBatch] = []
  onnx_time = 0.0
  coreml_time = 0.0

  for start in tqdm(range(0, len(image_files), COREML_BATCH_SIZE), desc="Visual batches"):
    chunk = image_files[start : start + COREML_BATCH_SIZE]
    batch = np.stack([preprocess(Image.open(path).convert("RGB")) for path in chunk]).astype(np.float32)

    t0 = time.perf_counter()
    onnx_parts.append(_run_onnx_visual(onnx_session, batch))
    onnx_time += time.perf_counter() - t0

    t0 = time.perf_counter()
    coreml_parts.append(_run_coreml_visual(coreml_session, batch))
    coreml_time += time.perf_counter() - t0

  return {
    "onnx": _normalize(np.concatenate(onnx_parts, axis=0)),
    "coreml": _normalize(np.concatenate(coreml_parts, axis=0)),
  }, {
    "onnx": onnx_time,
    "coreml": coreml_time,
  }


def _write_temp_index(image_files: list[Path], vectors: FeatureBatch) -> None:
  with tempfile.TemporaryDirectory() as tmpdir:
    database = db.DB(Path(tmpdir) / "db.sqlite3")
    try:
      for path, vector in zip(image_files, vectors):
        stat = path.stat()
        database.upsert_image(
          db.NewImage(
            filepath=str(path), modified_at=stat.st_mtime, size=stat.st_size, vector=vector.astype(np.float32).tobytes()
          ),
          commit=False,
        )
      database.commit()
    finally:
      database.close()


def main() -> None:
  parser = argparse.ArgumentParser(description="Benchmark deployed visual ONNX vs CoreML models")
  parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Flat ImageNet subset directory")
  args = parser.parse_args()

  dataset_dir = Path(args.dataset_dir)
  dataset_root = dataset_dir.parent
  image_files = _iter_image_files(dataset_dir)
  class_map = _load_imagenet_classes(dataset_root)
  class_ids, text_features = _compute_text_features(class_map)

  visual_features, timings = _compute_visual_features(image_files)
  cosine = np.sum(visual_features["onnx"] * visual_features["coreml"], axis=1)
  max_abs = np.max(np.abs(visual_features["onnx"] - visual_features["coreml"]), axis=1)

  onnx_top1, onnx_top5 = _compute_accuracy(visual_features["onnx"], image_files, class_ids, text_features)
  coreml_top1, coreml_top5 = _compute_accuracy(visual_features["coreml"], image_files, class_ids, text_features)

  _write_temp_index(image_files, visual_features["coreml"])

  print(f"Dataset: {dataset_dir}")
  print(f"Images: {len(image_files)}")
  print(f"ONNX img/s: {len(image_files) / timings['onnx']:.1f}")
  print(f"CoreML img/s: {len(image_files) / timings['coreml']:.1f}")
  print(
    f"CoreML speedup over ONNX: {(len(image_files) / timings['coreml']) / (len(image_files) / timings['onnx']):.2f}x"
  )
  print(f"Cosine mean: {float(cosine.mean()):.9f}")
  print(f"Cosine min: {float(cosine.min()):.9f}")
  print(f"Cosine p01: {float(np.quantile(cosine, 0.01)):.9f}")
  print(f"Normalized max abs diff mean: {float(max_abs.mean()):.9e}")
  print(f"Normalized max abs diff max: {float(max_abs.max()):.9e}")
  print(f"ONNX Top-1: {onnx_top1:.6f}")
  print(f"ONNX Top-5: {onnx_top5:.6f}")
  print(f"CoreML Top-1: {coreml_top1:.6f}")
  print(f"CoreML Top-5: {coreml_top5:.6f}")
  print(f"Top-1 delta: {coreml_top1 - onnx_top1:+.6f}")
  print(f"Top-5 delta: {coreml_top5 - onnx_top5:+.6f}")


if __name__ == "__main__":
  main()
