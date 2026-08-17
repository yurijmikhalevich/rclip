---
license: mit
base_model: laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K
datasets:
  - mlfoundations/datacomp_pools
pipeline_tag: zero-shot-image-classification
tags:
  - open_clip
  - onnx
  - coreml
---

# rclip model artifacts

This repository contains the inference artifacts downloaded by
[`rclip`](https://github.com/yurijmikhalevich/rclip). They are ONNX and Core ML
format conversions of OpenCLIP's `ViT-B-32-256` checkpoint
`datacomp_s34b_b86k`; the learned model is not fine-tuned or otherwise changed.

## Provenance and attribution

The source checkpoint is
[`laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K`](https://huggingface.co/laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K),
trained by Mehdi Cherti on the DataComp-1B dataset using
[`mlfoundations/open_clip`](https://github.com/mlfoundations/open_clip). The
tokenizer vocabulary originates from OpenAI CLIP through OpenCLIP.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the complete
attribution and [`LICENSE`](LICENSE) and
[`OPENAI-CLIP-MIT.txt`](OPENAI-CLIP-MIT.txt) for the applicable MIT notices.

## Files

- `ViT-B-32-256-datacomp_s34b_b86k/visual.onnx`: visual encoder for ONNX Runtime
- `ViT-B-32-256-datacomp_s34b_b86k/textual.onnx`: text encoder for ONNX Runtime
- `ViT-B-32-256-datacomp_s34b_b86k/visual.mlpackage`: visual encoder for Core ML
- `tokenizer/bpe_simple_vocab_16e6.txt.gz`: CLIP BPE vocabulary

## Uses and limitations

The model supports image and text retrieval and zero-shot image
classification. The upstream model card recommends task-specific testing and
warns against untested deployment, surveillance, and facial-recognition uses.
Review its full intended-use, limitation, bias, and training-data disclosures
before using these artifacts outside rclip's local image-search workflow.

## References

- [DataComp paper](https://arxiv.org/abs/2304.14108)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [OpenAI CLIP](https://github.com/openai/CLIP)
