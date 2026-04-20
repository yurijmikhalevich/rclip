#!/usr/bin/env python3
"""
Convert ViT-B-32-256 DataComp-1B from open_clip to ONNX and CoreML formats.

Requirements (not rclip runtime deps):
  pip install torch open_clip_torch coremltools huggingface_hub

Usage:
  python scripts/convert_model.py --output-dir ./models
  python scripts/convert_model.py --output-dir ./models --upload --hf-repo yurijmikhalevich/rclip-models
"""

import argparse
import logging
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import open_clip
import torch

if TYPE_CHECKING:
  pass

MODEL_NAME = "ViT-B-32-256"
PRETRAINED = "datacomp_s34b_b86k"
IMAGE_SIZE = 256
CONTEXT_LENGTH = 77
COREML_VISUAL_BATCH_SIZE = 8
FeatureBatch = npt.NDArray[np.float32]


@contextmanager
def _quiet_onnx_export():
  schema_logger = logging.getLogger("torch.onnx._internal.exporter._schemas")
  original_level = schema_logger.level
  schema_logger.setLevel(logging.ERROR)
  try:
    with warnings.catch_warnings():
      warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated, use `isinstance\(treespec, TreeSpec\) and treespec\.is_leaf\(\)` instead\.",
        category=FutureWarning,
      )
      yield
  finally:
    schema_logger.setLevel(original_level)


@contextmanager
def _disable_mha_fastpath():
  original_state = torch.backends.mha.get_fastpath_enabled()
  torch.backends.mha.set_fastpath_enabled(False)
  try:
    yield
  finally:
    torch.backends.mha.set_fastpath_enabled(original_state)


class _VisualWrapper(torch.nn.Module):
  def __init__(self, visual_model: torch.nn.Module):
    super().__init__()
    self.visual = visual_model

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    return self.visual(inputs)


class _TextualWrapper(torch.nn.Module):
  def __init__(self, clip_model: open_clip.CLIP):
    super().__init__()
    self.clip_model = clip_model

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    return self.clip_model.encode_text(inputs)


def _normalize_rows(features: FeatureBatch) -> FeatureBatch:
  return features / np.linalg.norm(features, axis=-1, keepdims=True)


def _assert_normalized_features_close(
  reference: FeatureBatch,
  candidate: FeatureBatch,
  *,
  label: str,
  cosine_threshold: float,
  max_abs_threshold: float,
) -> None:
  reference = _normalize_rows(np.asarray(reference, dtype=np.float32))
  candidate = _normalize_rows(np.asarray(candidate, dtype=np.float32))

  cosine = np.sum(reference * candidate, axis=1)
  max_abs_diff = np.max(np.abs(reference - candidate))
  print(f"  {label} cosine min: {float(cosine.min()):.8f}")
  print(f"  {label} normalized max abs diff: {max_abs_diff:.2e}")
  if float(cosine.min()) <= cosine_threshold:
    raise RuntimeError(f"{label} cosine drift is too large")
  if max_abs_diff >= max_abs_threshold:
    raise RuntimeError(f"{label} normalized feature drift is too large")


def export_visual_onnx(model: open_clip.CLIP, output_path: Path) -> None:
  wrapper = _VisualWrapper(model.visual)
  wrapper.eval()

  dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
  with _quiet_onnx_export(), _disable_mha_fastpath():
    torch.onnx.export(
      wrapper,
      (dummy_input,),
      str(output_path),
      input_names=["input"],
      output_names=["output"],
      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
      opset_version=18,
      dynamo=False,
    )
  print(f"  Exported visual ONNX: {output_path}")


def export_textual_onnx(model: open_clip.CLIP, output_path: Path) -> None:
  wrapper = _TextualWrapper(model)
  wrapper.eval()

  dummy_input = torch.zeros(1, CONTEXT_LENGTH, dtype=torch.long)
  with _quiet_onnx_export(), _disable_mha_fastpath():
    torch.onnx.export(
      wrapper,
      (dummy_input,),
      str(output_path),
      input_names=["input"],
      output_names=["output"],
      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
      opset_version=18,
      dynamo=False,
    )
  print(f"  Exported textual ONNX: {output_path}")


def convert_visual_to_coreml(model: open_clip.CLIP, output_path: Path) -> None:
  import coremltools as ct

  wrapper = _VisualWrapper(model.visual)
  wrapper.eval()

  dummy_input = torch.randn(COREML_VISUAL_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
  traced = torch.jit.trace(wrapper, dummy_input, check_trace=False)

  ml_model = ct.convert(
    traced,
    source="pytorch",
    inputs=[ct.TensorType(name="input", shape=(COREML_VISUAL_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))],
    outputs=[ct.TensorType(name="output")],
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS13,
  )
  if not isinstance(ml_model, ct.models.MLModel):
    raise RuntimeError("CoreML conversion did not return an MLModel")
  ml_model.save(str(output_path))
  print(f"  Converted visual to CoreML FP32: {output_path}")


def verify_onnx(model: open_clip.CLIP, visual_onnx_path: Path, textual_onnx_path: Path) -> None:
  import onnxruntime as ort

  model.eval()

  dummy_image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
  with torch.no_grad():
    pt_visual = np.asarray(model.visual(dummy_image).numpy(), dtype=np.float32)

  visual_session = ort.InferenceSession(str(visual_onnx_path), providers=["CPUExecutionProvider"])
  (onnx_visual_raw,) = visual_session.run(None, {"input": dummy_image.numpy()})
  onnx_visual = np.asarray(onnx_visual_raw, dtype=np.float32)
  visual_diff = np.max(np.abs(pt_visual - onnx_visual))
  print(f"  Visual max abs diff: {visual_diff:.2e}")
  if visual_diff >= 1e-4:
    raise RuntimeError(f"Visual ONNX diverges too much: {visual_diff}")

  dummy_text = torch.zeros(2, CONTEXT_LENGTH, dtype=torch.long)
  dummy_text[0, 0] = 49406
  dummy_text[0, 1] = 320
  dummy_text[0, 2] = 49407
  dummy_text[1, 0] = 49406
  dummy_text[1, 1] = 539
  dummy_text[1, 2] = 49407
  with torch.no_grad():
    pt_text = np.asarray(model.encode_text(dummy_text).numpy(), dtype=np.float32)

  textual_session = ort.InferenceSession(str(textual_onnx_path), providers=["CPUExecutionProvider"])
  (onnx_text_raw,) = textual_session.run(None, {"input": dummy_text.numpy()})
  onnx_text = np.asarray(onnx_text_raw, dtype=np.float32)
  text_diff = np.max(np.abs(pt_text - onnx_text))
  print(f"  Textual max abs diff: {text_diff:.2e}")
  if text_diff >= 1e-4:
    raise RuntimeError(f"Textual ONNX diverges too much: {text_diff}")

  print("  ONNX verification passed!")


def verify_coreml(
  model: open_clip.CLIP,
  visual_coreml_path: Path,
  *,
  cosine_threshold: float = 0.99999,
  max_abs_threshold: float = 1e-4,
) -> None:
  import coremltools as ct

  model.eval()

  rng = np.random.default_rng(0)
  dummy_image = rng.standard_normal((COREML_VISUAL_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
  with torch.no_grad():
    pt_visual = np.asarray(model.visual(torch.from_numpy(dummy_image)).numpy(), dtype=np.float32)

  ml_model = ct.models.MLModel(str(visual_coreml_path), compute_units=ct.ComputeUnit.ALL)
  result = ml_model.predict({"input": dummy_image})
  coreml_visual = np.asarray(result["output"], dtype=np.float32)

  _assert_normalized_features_close(
    pt_visual,
    coreml_visual,
    label="Visual CoreML/PyTorch",
    cosine_threshold=cosine_threshold,
    max_abs_threshold=max_abs_threshold,
  )
  print("  CoreML verification passed!")


def verify_coreml_matches_visual_onnx(visual_onnx_path: Path, visual_coreml_path: Path) -> None:
  import coremltools as ct
  import onnxruntime as ort

  rng = np.random.default_rng(0)
  dummy_image = rng.standard_normal((COREML_VISUAL_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

  onnx_session = ort.InferenceSession(str(visual_onnx_path), providers=["CPUExecutionProvider"])
  (onnx_visual_raw,) = onnx_session.run(None, {"input": dummy_image})
  onnx_visual = np.asarray(onnx_visual_raw, dtype=np.float32)

  ml_model = ct.models.MLModel(str(visual_coreml_path), compute_units=ct.ComputeUnit.ALL)
  result = ml_model.predict({"input": dummy_image})
  coreml_visual = np.asarray(result["output"], dtype=np.float32)

  _assert_normalized_features_close(
    onnx_visual,
    coreml_visual,
    label="Visual ONNX/CoreML",
    cosine_threshold=0.99999,
    max_abs_threshold=1e-4,
  )
  print("  CoreML/ONNX parity verification passed!")


def upload_to_hf(output_dir: Path, hf_repo: str) -> None:
  from huggingface_hub import HfApi

  api = HfApi()
  api.create_repo(hf_repo, exist_ok=True)

  commit_message = f"Upload converted {MODEL_NAME}/{PRETRAINED} artifacts"
  api.upload_folder(
    repo_id=hf_repo,
    folder_path=output_dir,
    commit_message=commit_message,
    commit_description="Replace the existing exported model artifacts with the latest snapshot.",
  )
  print(f"  Uploaded snapshot from: {output_dir}")

  api.super_squash_history(hf_repo, commit_message=commit_message)
  print("  Squashed Hugging Face repo history")


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert CLIP model to ONNX and CoreML")
  parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save converted models")
  parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
  parser.add_argument("--hf-repo", default="yurijmikhalevich/rclip-models", help="HuggingFace repo ID")
  parser.add_argument("--skip-coreml", action="store_true", help="Skip CoreML conversion (for non-macOS)")
  args = parser.parse_args()

  model_dir = args.output_dir / f"{MODEL_NAME}-{PRETRAINED}"
  model_dir.mkdir(parents=True, exist_ok=True)

  print(f"Loading {MODEL_NAME}/{PRETRAINED}...")
  model, _preprocess_train, _preprocess_eval = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
  if not isinstance(model, open_clip.CLIP):
    raise RuntimeError(f"Expected open_clip.CLIP, got {type(model).__name__}")
  model.eval()

  visual_onnx = model_dir / "visual.onnx"
  textual_onnx = model_dir / "textual.onnx"

  print("Exporting to ONNX...")
  export_visual_onnx(model, visual_onnx)
  export_textual_onnx(model, textual_onnx)

  print("Verifying ONNX...")
  verify_onnx(model, visual_onnx, textual_onnx)

  if not args.skip_coreml:
    visual_coreml = model_dir / "visual.mlpackage"
    if visual_coreml.exists():
      shutil.rmtree(visual_coreml)

    print("Converting to CoreML FP32...")
    convert_visual_to_coreml(model, visual_coreml)

    print("Verifying CoreML...")
    verify_coreml(model, visual_coreml)
    verify_coreml_matches_visual_onnx(visual_onnx, visual_coreml)

  import open_clip as open_clip_module

  vocab_src = Path(open_clip_module.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
  vocab_dst = args.output_dir / "tokenizer" / "bpe_simple_vocab_16e6.txt.gz"
  vocab_dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(vocab_src, vocab_dst)
  print(f"Copied vocab: {vocab_dst}")

  if args.upload:
    print(f"Uploading to {args.hf_repo}...")
    upload_to_hf(args.output_dir, args.hf_repo)
    print("Upload complete!")

  print(f"\nAll models saved to: {model_dir}")


if __name__ == "__main__":
  main()
