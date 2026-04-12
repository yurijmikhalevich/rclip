#!/usr/bin/env python3
"""
Benchmark INT8 quantized ONNX models vs FP32 ONNX and PyTorch+MPS on ImageNet-1k.

Usage:
  poetry run python benchmarks/benchmark-int8-quantization.py
  poetry run python benchmarks/benchmark-int8-quantization.py --images-per-class 50
  poetry run python benchmarks/benchmark-int8-quantization.py --all-images
"""

import argparse
import os
import re
import tempfile
import time
from collections.abc import Callable
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

from rclip.utils.preprocess import preprocess
from rclip.utils.tokenizer import SimpleTokenizer

DATASET_DIR = "/Users/yurij/datasets/imagenet_1k/data"
BATCH_SIZE = 64
NUM_CALIBRATION_IMAGES = 200
DEFAULT_IMAGES_PER_CLASS = 50
MLP_NODE_RE = re.compile(r"/mlp/(c_fc|c_proj)/")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def get_image_files(dataset_dir: str, max_per_class: int | None = None) -> list[str]:
  if max_per_class is None:
    files = []
    for subdir in sorted(os.listdir(dataset_dir)):
      p = os.path.join(dataset_dir, subdir)
      if not os.path.isdir(p):
        continue
      for f in sorted(os.listdir(p)):
        if f.endswith(".JPEG"):
          files.append(os.path.join(p, f))
    return files

  by_class: dict[str, list[str]] = defaultdict(list)
  for subdir in sorted(os.listdir(dataset_dir)):
    p = os.path.join(dataset_dir, subdir)
    if not os.path.isdir(p):
      continue
    for f in sorted(os.listdir(p)):
      if f.endswith(".JPEG"):
        by_class[f.split("_")[0]].append(os.path.join(p, f))
  files = []
  for synset in sorted(by_class.keys()):
    files.extend(by_class[synset][:max_per_class])
  return files


def get_synset_to_idx(dataset_dir: str) -> dict[str, int]:
  synset_ids: set[str] = set()
  for subdir in os.listdir(dataset_dir):
    p = os.path.join(dataset_dir, subdir)
    if not os.path.isdir(p):
      continue
    for f in os.listdir(p):
      if f.endswith(".JPEG"):
        synset_ids.add(f.split("_")[0])
  return {s: i for i, s in enumerate(sorted(synset_ids))}


def get_image_class(filepath: str) -> str:
  return os.path.basename(filepath).split("_")[0]


def get_model_dir() -> str:
  from rclip.utils.helpers import get_app_datadir

  return os.path.join(str(get_app_datadir()), "ViT-B-32-256-datacomp_s34b_b86k")


def get_tokenizer() -> SimpleTokenizer:
  from rclip.model import Model
  from rclip.utils.helpers import get_app_datadir

  Model().ensure_downloaded()
  vocab_path = os.path.join(str(get_app_datadir()), "tokenizer", "bpe_simple_vocab_16e6.txt.gz")
  return SimpleTokenizer(bpe_path=vocab_path)


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def run_onnx(session: ort.InferenceSession, batch: npt.NDArray[np.generic]) -> npt.NDArray[np.float32]:
  input_meta = session.get_inputs()[0]
  if input_meta.type == "tensor(float16)":
    batch = batch.astype(np.float16)
  (features,) = session.run(None, {"input": batch})
  features = np.asarray(features, dtype=np.float32)
  return features / np.linalg.norm(features, axis=-1, keepdims=True)


def can_load_onnx(path: str) -> bool:
  try:
    ort.InferenceSession(path, providers=["CPUExecutionProvider"])
  except Exception as exc:
    print(f"    Skipping {os.path.basename(path)}: {str(exc).splitlines()[0]}")
    return False
  return True


def compute_text_features(
  session: ort.InferenceSession,
  class_names: tuple[str, ...],
  tokenizer: SimpleTokenizer,
) -> npt.NDArray[np.float32]:
  prompts = [f"a photo of a {name}" for name in class_names]
  tokens = tokenizer(prompts)
  parts = []
  for start in range(0, len(prompts), 128):
    parts.append(run_onnx(session, tokens[start : start + 128]))
  return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# INT8 quantization
# ---------------------------------------------------------------------------


def _print_size_delta(input_path: str, output_path: str) -> None:
  orig_mb = os.path.getsize(input_path) / 1024 / 1024
  quant_mb = os.path.getsize(output_path) / 1024 / 1024
  print(f"    {orig_mb:.1f} MB -> {quant_mb:.1f} MB ({quant_mb / orig_mb * 100:.0f}%)")


def _get_quantizable_nodes(input_path: str, *, pattern: re.Pattern[str] | None = None) -> list[str]:
  model = onnx.load(input_path)
  node_names: list[str] = []
  for node in model.graph.node:
    if node.op_type not in {"MatMul", "Gemm"} or not node.name:
      continue
    if pattern is not None and not pattern.search(node.name):
      continue
    node_names.append(node.name)
  return node_names


def export_fp16_onnx(output_dir: str) -> tuple[str, str]:
  import open_clip

  from scripts.convert_model import export_textual_onnx_fp16, export_visual_onnx_fp16

  model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-256", pretrained="datacomp_s34b_b86k")
  model = cast(open_clip.CLIP, model)
  model.eval()

  visual_output = os.path.join(output_dir, "visual_fp16.onnx")
  textual_output = os.path.join(output_dir, "textual_fp16.onnx")
  export_visual_onnx_fp16(model, Path(visual_output))
  export_textual_onnx_fp16(model, Path(textual_output))

  return visual_output, textual_output


def quantize_dynamic(
  input_path: str,
  output_path: str,
  *,
  nodes_to_quantize: list[str] | None = None,
  per_channel: bool = False,
) -> None:
  from onnxruntime.quantization import quantize_dynamic as _qd, QuantType

  _qd(
    input_path,
    output_path,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
    nodes_to_quantize=nodes_to_quantize,
    per_channel=per_channel,
  )
  _print_size_delta(input_path, output_path)


def quantize_static_visual(
  input_path: str,
  output_path: str,
  calib_files: list[str],
  *,
  nodes_to_quantize: list[str] | None = None,
  per_channel: bool = False,
) -> None:
  from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static as _qs,
    QuantType,
    QuantFormat,
    CalibrationMethod,
  )

  class _Reader(CalibrationDataReader):
    def __init__(self) -> None:
      self.index = 0

    def get_next(self) -> dict[str, npt.NDArray[np.float32]]:
      if self.index >= len(calib_files):
        return {}
      img = Image.open(calib_files[self.index]).convert("RGB")
      self.index += 1
      return {"input": preprocess(img)[np.newaxis]}

    def rewind(self) -> None:
      self.index = 0

  _qs(
    input_path,
    output_path,
    cast(Any, _Reader()),
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    calibrate_method=CalibrationMethod.MinMax,
    op_types_to_quantize=["MatMul", "Gemm"],
    nodes_to_quantize=nodes_to_quantize,
    per_channel=per_channel,
  )
  _print_size_delta(input_path, output_path)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def compute_accuracy(
  image_features: npt.NDArray[np.float32],
  text_features: npt.NDArray[np.float32],
  image_files: list[str],
  synset_to_idx: dict[str, int],
) -> tuple[float, float]:
  n = len(image_files)
  top1 = top5 = 0
  for start in range(0, n, 2048):
    end = min(start + 2048, n)
    sims = image_features[start:end] @ text_features.T
    top5_preds = np.argsort(sims, axis=1)[:, -5:]
    for i, idx in enumerate(range(start, end)):
      gt_idx = synset_to_idx[get_image_class(image_files[idx])]
      if top5_preds[i, -1] == gt_idx:
        top1 += 1
      if gt_idx in top5_preds[i]:
        top5 += 1
  return top1 / n, top5 / n


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def benchmark_onnx(
  image_files: list[str],
  class_names: tuple[str, ...],
  synset_to_idx: dict[str, int],
  tokenizer: SimpleTokenizer,
  variants: list[tuple[str, str, str]],  # (label, visual_path, textual_path)
) -> dict[str, dict[str, float]]:
  """Benchmark ONNX variants while reusing repeated visual/textual sessions."""
  results: dict[str, dict[str, float]] = {}

  visual_paths = list(dict.fromkeys(vp for _, vp, _ in variants))
  textual_paths = list(dict.fromkeys(tp for _, _, tp in variants))
  visual_sessions = {vp: ort.InferenceSession(vp, providers=["CPUExecutionProvider"]) for vp in visual_paths}
  textual_sessions = {tp: ort.InferenceSession(tp, providers=["CPUExecutionProvider"]) for tp in textual_paths}

  # Text features
  text_map: dict[str, npt.NDArray[np.float32]] = {}
  for tp in textual_paths:
    t0 = time.perf_counter()
    text_map[tp] = compute_text_features(textual_sessions[tp], class_names, tokenizer)
    labels = [label for label, _, textual_path in variants if textual_path == tp]
    print(f"  [{', '.join(labels)}] Text features: {time.perf_counter() - t0:.2f}s")

  # Image features — preprocess once, run through all visual sessions
  n_batches = (len(image_files) + BATCH_SIZE - 1) // BATCH_SIZE
  feat_lists: dict[str, list[npt.NDArray[np.float32]]] = {vp: [] for vp in visual_paths}
  inf_times: dict[str, float] = {vp: 0.0 for vp in visual_paths}

  for batch_idx in tqdm(range(n_batches), desc="  Image batches"):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(image_files))
    batch = np.stack([preprocess(Image.open(f).convert("RGB")) for f in image_files[start:end]])

    for vp in visual_paths:
      t0 = time.perf_counter()
      feat_lists[vp].append(run_onnx(visual_sessions[vp], batch))
      inf_times[vp] += time.perf_counter() - t0

  image_feature_map = {vp: np.concatenate(parts, axis=0) for vp, parts in feat_lists.items()}

  for label, vp, tp in variants:
    top1, top5 = compute_accuracy(image_feature_map[vp], text_map[tp], image_files, synset_to_idx)
    t = inf_times[vp]
    ips = len(image_files) / t
    results[label] = {"top1": top1, "top5": top5, "img_per_sec": ips, "inference_time": t}
    print(f"  [{label}] Top-1: {top1 * 100:.2f}%  Top-5: {top5 * 100:.2f}%  Inference: {t:.1f}s ({ips:.0f} img/s)")

  return results


def benchmark_pytorch_mps(
  image_files: list[str],
  class_names: tuple[str, ...],
  synset_to_idx: dict[str, int],
) -> dict[str, float]:
  import torch
  import open_clip

  device = torch.device("mps")
  model, _, torch_preprocess_raw = open_clip.create_model_and_transforms(
    "ViT-B-32-256",
    pretrained="datacomp_s34b_b86k",
  )
  torch_preprocess = cast(Callable[[Image.Image], torch.Tensor], torch_preprocess_raw)
  model = model.to(device)  # type: ignore[assignment]
  model.eval()

  oc_tokenizer = open_clip.get_tokenizer("ViT-B-32-256")

  print("  [PyTorch+MPS] Computing text features...")
  prompts = [f"a photo of a {name}" for name in class_names]
  with torch.no_grad():
    text_tokens = oc_tokenizer(prompts).to(device)
    text_features = model.encode_text(text_tokens)  # type: ignore[union-attr]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features_np: npt.NDArray[np.float32] = text_features.cpu().numpy()

  all_features: list[npt.NDArray[np.float32]] = []
  total_time = 0.0
  n_batches = (len(image_files) + BATCH_SIZE - 1) // BATCH_SIZE

  for batch_idx in tqdm(range(n_batches), desc="  [PyTorch+MPS] Image features"):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(image_files))
    images = [torch_preprocess(Image.open(f).convert("RGB")) for f in image_files[start:end]]
    batch = torch.stack(images).to(device)  # type: ignore[arg-type]

    t0 = time.perf_counter()
    with torch.no_grad():
      features = model.encode_image(batch)  # type: ignore[union-attr]
      features = features / features.norm(dim=-1, keepdim=True)
      torch.mps.synchronize()
    total_time += time.perf_counter() - t0
    all_features.append(features.cpu().numpy())

  img_feat = np.concatenate(all_features, axis=0)
  ips = len(image_files) / total_time
  print(f"  [PyTorch+MPS] Visual inference: {total_time:.1f}s ({ips:.0f} img/s)")

  top1, top5 = compute_accuracy(img_feat, text_features_np, image_files, synset_to_idx)
  print(f"  [PyTorch+MPS] Top-1: {top1 * 100:.2f}%  Top-5: {top5 * 100:.2f}%")
  return {"top1": top1, "top5": top5, "img_per_sec": ips, "inference_time": total_time}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(description="INT8 quantization benchmark")
  parser.add_argument("--images-per-class", type=int, default=DEFAULT_IMAGES_PER_CLASS)
  parser.add_argument("--all-images", action="store_true")
  parser.add_argument("--skip-mps", action="store_true")
  parser.add_argument("--skip-static", action="store_true")
  args = parser.parse_args()

  max_per_class = None if args.all_images else args.images_per_class

  print("=" * 75)
  print("INT8 Quantization Benchmark on ImageNet-1k")
  print("=" * 75)

  print("\nCollecting image files...")
  image_files = get_image_files(DATASET_DIR, max_per_class=max_per_class)
  synset_to_idx = get_synset_to_idx(DATASET_DIR)
  print(f"  Images: {len(image_files)} ({f'{max_per_class}/class' if max_per_class else 'all'})")
  print(f"  Classes: {len(synset_to_idx)}")

  from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES

  class_names = IMAGENET_CLASSNAMES

  model_dir = get_model_dir()
  visual_fp32 = os.path.join(model_dir, "visual.onnx")
  textual_fp32 = os.path.join(model_dir, "textual.onnx")
  tokenizer = get_tokenizer()

  with tempfile.TemporaryDirectory() as tmpdir:
    # ---- Quantize ----
    print("\n" + "=" * 75)
    print("Quantizing models")
    print("=" * 75)

    visual_mlp_nodes = _get_quantizable_nodes(visual_fp32, pattern=MLP_NODE_RE)
    textual_mlp_nodes = _get_quantizable_nodes(textual_fp32, pattern=MLP_NODE_RE)

    print("\n  FP16 - direct export:")
    visual_fp16, textual_fp16 = export_fp16_onnx(tmpdir)
    _print_size_delta(visual_fp32, visual_fp16)
    _print_size_delta(textual_fp32, textual_fp16)

    print("\n  Dynamic INT8 — visual:")
    visual_dyn = os.path.join(tmpdir, "visual_dyn.onnx")
    quantize_dynamic(visual_fp32, visual_dyn)

    print("  Dynamic INT8 - visual MLP only:")
    visual_dyn_mlp = os.path.join(tmpdir, "visual_dyn_mlp.onnx")
    quantize_dynamic(visual_fp32, visual_dyn_mlp, nodes_to_quantize=visual_mlp_nodes)

    print("  Dynamic INT8 - visual MLP only (per-channel):")
    visual_dyn_mlp_pc = os.path.join(tmpdir, "visual_dyn_mlp_pc.onnx")
    quantize_dynamic(visual_fp32, visual_dyn_mlp_pc, nodes_to_quantize=visual_mlp_nodes, per_channel=True)

    print("  Dynamic INT8 — textual:")
    textual_dyn = os.path.join(tmpdir, "textual_dyn.onnx")
    quantize_dynamic(textual_fp32, textual_dyn)

    print("  Dynamic INT8 - textual MLP only:")
    textual_dyn_mlp = os.path.join(tmpdir, "textual_dyn_mlp.onnx")
    quantize_dynamic(textual_fp32, textual_dyn_mlp, nodes_to_quantize=textual_mlp_nodes)

    candidate_variants: list[tuple[str, str, str]] = [
      ("ONNX FP32", visual_fp32, textual_fp32),
      ("ONNX FP16", visual_fp16, textual_fp16),
      ("ONNX FP32 + Text FP16", visual_fp32, textual_fp16),
      ("INT8-Dyn Visual Only", visual_dyn, textual_fp32),
      ("INT8-Dyn Visual MLP", visual_dyn_mlp, textual_fp32),
      ("INT8-Dyn Visual MLP PC", visual_dyn_mlp_pc, textual_fp32),
      ("INT8-Dyn Both", visual_dyn, textual_dyn),
      ("INT8-Dyn Text MLP", visual_fp32, textual_dyn_mlp),
      ("INT8-Dyn Visual + Text FP16", visual_dyn, textual_fp16),
      ("INT8-Dyn Visual MLP + Text FP16", visual_dyn_mlp, textual_fp16),
    ]

    if not args.skip_static:
      calib_files = image_files[:NUM_CALIBRATION_IMAGES]
      print(f"\n  Static INT8 — visual (calibrating on {len(calib_files)} images):")
      visual_static = os.path.join(tmpdir, "visual_static.onnx")
      quantize_static_visual(visual_fp32, visual_static, calib_files)
      candidate_variants.append(("INT8-Static Visual Only", visual_static, textual_fp32))

      print(f"  Static INT8 - visual MLP only (calibrating on {len(calib_files)} images):")
      visual_static_mlp = os.path.join(tmpdir, "visual_static_mlp.onnx")
      quantize_static_visual(visual_fp32, visual_static_mlp, calib_files, nodes_to_quantize=visual_mlp_nodes)
      candidate_variants.append(("INT8-Static Visual MLP", visual_static_mlp, textual_fp32))

    # Only benchmark artifacts that ORT can actually load on this machine.
    variants = [(label, vp, tp) for label, vp, tp in candidate_variants if can_load_onnx(vp) and can_load_onnx(tp)]

    # ---- Benchmark ONNX ----
    print("\n" + "=" * 75)
    print("Benchmarking on ImageNet-1k")
    print("=" * 75)

    results = benchmark_onnx(image_files, class_names, synset_to_idx, tokenizer, variants)

    # ---- PyTorch+MPS ----
    if not args.skip_mps:
      print()
      results["PyTorch+MPS"] = benchmark_pytorch_mps(image_files, class_names, synset_to_idx)

    # ---- Summary ----
    # Build model size map
    size_map: dict[str, float] = {}
    for label, vp, tp in variants:
      size_map[label] = (os.path.getsize(vp) + os.path.getsize(tp)) / 1024 / 1024

    print("\n" + "=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    print(f"\nDataset: {len(image_files)} images, {len(synset_to_idx)} classes, batch size: {BATCH_SIZE}\n")
    header = f"{'Variant':<25} {'Top-1':>8} {'Top-5':>8} {'Inf.Time':>9} {'Img/s':>7} {'Size':>9}"
    print(header)
    print("-" * len(header))
    for label, m in results.items():
      size_str = f"{size_map[label]:.0f} MB" if label in size_map else "N/A"
      print(
        f"{label:<25} {m['top1'] * 100:>7.2f}% {m['top5'] * 100:>7.2f}% "
        f"{m['inference_time']:>8.1f}s {m['img_per_sec']:>6.0f} {size_str:>9}"
      )

    # Accuracy delta table
    baseline = results["ONNX FP32"]
    print(f"\nAccuracy delta vs ONNX FP32 baseline:")
    print(f"{'Variant':<25} {'Δ Top-1':>9} {'Δ Top-5':>9}")
    print("-" * 45)
    for label, m in results.items():
      if label == "ONNX FP32":
        continue
      d1 = (m["top1"] - baseline["top1"]) * 100
      d5 = (m["top5"] - baseline["top5"]) * 100
      print(f"{label:<25} {d1:>+8.2f}% {d5:>+8.2f}%")


if __name__ == "__main__":
  main()
