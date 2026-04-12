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
from typing import cast

import numpy as np
import open_clip
import torch

MODEL_NAME = "ViT-B-32-256"
PRETRAINED = "datacomp_s34b_b86k"
IMAGE_SIZE = 256
CONTEXT_LENGTH = 77
EMBED_DIM = 512
COREML_VISUAL_BATCH_SIZE = 8
_MATMUL_OPS = ["MatMul", "Gemm"]


@contextmanager
def _quiet_onnx_export():
  # Suppress known exporter noise from dependencies while keeping real errors visible.
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

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.visual(x)


class _TextualWrapper(torch.nn.Module):
  def __init__(self, clip_model: open_clip.CLIP):
    super().__init__()
    self.clip_model = clip_model

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.clip_model.encode_text(x)


def _clone_model_for_onnx_export(model: open_clip.CLIP, *, dtype: torch.dtype) -> open_clip.CLIP:
  cloned = cast(open_clip.CLIP, open_clip.create_model(MODEL_NAME, pretrained=PRETRAINED))
  cloned.load_state_dict(model.state_dict())
  cloned = cloned.to(dtype=dtype)
  cloned.eval()
  return cloned


def export_visual_onnx(model: open_clip.CLIP, output_path: Path):
  """Export the visual encoder to ONNX."""
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


def export_visual_onnx_fp16(model: open_clip.CLIP, output_path: Path):
  """Export the visual encoder to ONNX with FP16 weights and activations."""
  export_model = _clone_model_for_onnx_export(model, dtype=torch.float16)
  wrapper = _VisualWrapper(export_model.visual)
  wrapper.eval()

  dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float16)
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
  print(f"  Exported visual ONNX FP16: {output_path}")


def export_textual_onnx(model: open_clip.CLIP, output_path: Path):
  """Export the textual encoder to ONNX."""
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


def export_textual_onnx_fp16(model: open_clip.CLIP, output_path: Path):
  """Export the textual encoder to ONNX with FP16 weights and activations."""
  export_model = _clone_model_for_onnx_export(model, dtype=torch.float16)
  wrapper = _TextualWrapper(export_model)
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
  print(f"  Exported textual ONNX FP16: {output_path}")


def quantize_visual_onnx_fp32_to_int8(input_path: Path, output_path: Path):
  """Quantize the visual encoder for CPU inference with dynamic INT8 weights."""
  from onnxruntime.quantization import quantize_dynamic, QuantType

  quantize_dynamic(
    str(input_path),
    str(output_path),
    weight_type=QuantType.QInt8,
    op_types_to_quantize=_MATMUL_OPS,
  )
  orig_mb = input_path.stat().st_size / 1024 / 1024
  quant_mb = output_path.stat().st_size / 1024 / 1024
  print(f"  Quantized visual ONNX to INT8: {output_path} ({orig_mb:.1f} MB -> {quant_mb:.1f} MB)")


def convert_visual_to_coreml(model: open_clip.CLIP, output_path: Path):
  """Convert the visual encoder to a fixed batch-8 CoreML model."""
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
  assert isinstance(ml_model, ct.models.MLModel)
  ml_model.save(str(output_path))
  print(f"  Converted visual to CoreML: {output_path}")


def verify_onnx(model: open_clip.CLIP, visual_onnx_path: Path, textual_onnx_path: Path):
  """Verify ONNX models produce the same output as PyTorch."""
  import onnxruntime as ort

  model.eval()

  # Verify visual with batch > 1 so the exported batch dimension is exercised.
  dummy_image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
  with torch.no_grad():
    pt_visual = model.visual(dummy_image).numpy()

  session = ort.InferenceSession(str(visual_onnx_path), providers=["CPUExecutionProvider"])
  (onnx_visual,) = session.run(None, {"input": dummy_image.numpy()})

  visual_diff = np.max(np.abs(pt_visual - onnx_visual))
  print(f"  Visual max abs diff: {visual_diff:.2e}")
  assert visual_diff < 1e-4, f"Visual ONNX diverges too much: {visual_diff}"

  # Verify textual with batch > 1 for the same reason.
  dummy_text = torch.zeros(2, CONTEXT_LENGTH, dtype=torch.long)
  dummy_text[0, 0] = 49406  # <start_of_text>
  dummy_text[0, 1] = 320  # "a"
  dummy_text[0, 2] = 49407  # <end_of_text>
  dummy_text[1, 0] = 49406
  dummy_text[1, 1] = 539
  dummy_text[1, 2] = 49407
  with torch.no_grad():
    pt_text = model.encode_text(dummy_text).numpy()

  session = ort.InferenceSession(str(textual_onnx_path), providers=["CPUExecutionProvider"])
  (onnx_text,) = session.run(None, {"input": dummy_text.numpy()})

  text_diff = np.max(np.abs(pt_text - onnx_text))
  print(f"  Textual max abs diff: {text_diff:.2e}")
  assert text_diff < 1e-4, f"Textual ONNX diverges too much: {text_diff}"

  print("  ONNX verification passed!")


def verify_onnx_fp16(model: open_clip.CLIP, visual_onnx_path: Path, textual_onnx_path: Path):
  """Verify FP16 ONNX models produce close outputs to FP32 PyTorch."""
  import onnxruntime as ort

  model.eval()

  dummy_image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
  with torch.no_grad():
    pt_visual = model.visual(dummy_image).numpy()

  session = ort.InferenceSession(str(visual_onnx_path), providers=["CPUExecutionProvider"])
  (onnx_visual,) = session.run(None, {"input": dummy_image.numpy().astype(np.float16)})

  visual_diff = np.max(np.abs(pt_visual - np.asarray(onnx_visual, dtype=np.float32)))
  print(f"  Visual FP16 max abs diff: {visual_diff:.2e}")
  assert visual_diff < 5e-2, f"Visual ONNX FP16 diverges too much: {visual_diff}"

  dummy_text = torch.zeros(2, CONTEXT_LENGTH, dtype=torch.long)
  dummy_text[0, 0] = 49406
  dummy_text[0, 1] = 320
  dummy_text[0, 2] = 49407
  dummy_text[1, 0] = 49406
  dummy_text[1, 1] = 539
  dummy_text[1, 2] = 49407
  with torch.no_grad():
    pt_text = model.encode_text(dummy_text).numpy()

  session = ort.InferenceSession(str(textual_onnx_path), providers=["CPUExecutionProvider"])
  (onnx_text,) = session.run(None, {"input": dummy_text.numpy()})

  text_diff = np.max(np.abs(pt_text - np.asarray(onnx_text, dtype=np.float32)))
  print(f"  Textual FP16 max abs diff: {text_diff:.2e}")
  assert text_diff < 5e-2, f"Textual ONNX FP16 diverges too much: {text_diff}"

  print("  ONNX FP16 verification passed!")


def verify_coreml(model: open_clip.CLIP, visual_coreml_path: Path):
  """Verify the visual CoreML model produces the same output as PyTorch."""
  import coremltools as ct

  model.eval()

  # Verify visual with the exported batch size.
  dummy_image = torch.randn(COREML_VISUAL_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
  with torch.no_grad():
    pt_visual = model.visual(dummy_image).numpy()

  ml_model = ct.models.MLModel(str(visual_coreml_path), compute_units=ct.ComputeUnit.ALL)
  result = ml_model.predict({"input": dummy_image.numpy()})
  coreml_visual = result["output"]

  visual_diff = np.max(np.abs(pt_visual - coreml_visual))
  print(f"  Visual max abs diff: {visual_diff:.2e}")
  assert visual_diff < 1e-3, f"Visual CoreML diverges too much: {visual_diff}"

  print("  CoreML verification passed!")


def upload_to_hf(output_dir: Path, hf_repo: str):
  """Upload model files to HuggingFace Hub without accumulating history."""
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

  # Keep a single snapshot in repo history so repeated uploads do not retain older file revisions.
  api.super_squash_history(hf_repo, commit_message=commit_message)
  print("  Squashed Hugging Face repo history")


def main():
  parser = argparse.ArgumentParser(description="Convert CLIP model to ONNX and CoreML")
  parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save converted models")
  parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
  parser.add_argument("--hf-repo", default="yurijmikhalevich/rclip-models", help="HuggingFace repo ID")
  parser.add_argument("--skip-coreml", action="store_true", help="Skip CoreML conversion (for non-macOS)")
  args = parser.parse_args()

  model_dir = args.output_dir / f"{MODEL_NAME}-{PRETRAINED}"
  model_dir.mkdir(parents=True, exist_ok=True)

  print(f"Loading {MODEL_NAME}/{PRETRAINED}...")
  clip_model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
  model = cast(open_clip.CLIP, clip_model)
  model.eval()

  visual_onnx_fp32 = model_dir / "visual.fp32.onnx"
  textual_onnx_fp32 = model_dir / "textual.fp32.onnx"
  visual_onnx = model_dir / "visual.onnx"
  textual_onnx = model_dir / "textual.onnx"

  print("Exporting to ONNX...")
  export_visual_onnx(model, visual_onnx_fp32)
  export_textual_onnx(model, textual_onnx_fp32)
  quantize_visual_onnx_fp32_to_int8(visual_onnx_fp32, visual_onnx)
  export_textual_onnx_fp16(model, textual_onnx)

  print("Verifying ONNX...")
  verify_onnx(model, visual_onnx_fp32, textual_onnx_fp32)
  verify_onnx_fp16(model, visual_onnx, textual_onnx)

  if not args.skip_coreml:
    visual_coreml = model_dir / "visual.mlpackage"

    # Remove existing mlpackage dirs if they exist (they're directories)
    if visual_coreml.exists():
      shutil.rmtree(visual_coreml)

    print("Converting to CoreML FP32...")
    convert_visual_to_coreml(model, visual_coreml)

    print("Verifying CoreML...")
    verify_coreml(model, visual_coreml)

  # Copy tokenizer vocab from open_clip (dev dependency, identical to the vendored file)
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
