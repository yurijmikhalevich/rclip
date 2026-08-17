# Third-party notices

## DataComp OpenCLIP model

The ONNX and Core ML files in this repository are format conversions of
[`laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K`](https://huggingface.co/laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K),
a ViT-B/32 model trained by Mehdi Cherti on DataComp-1B using OpenCLIP. The
upstream model card declares the model under the MIT license. The OpenCLIP MIT
copyright and permission notice is reproduced in [`LICENSE`](LICENSE).

## OpenAI CLIP tokenizer

The BPE vocabulary in `tokenizer/bpe_simple_vocab_16e6.txt.gz` comes from
OpenCLIP, where the tokenizer is identified as copied from OpenAI CLIP under
the MIT license. The OpenAI copyright and permission notice is reproduced in
[`OPENAI-CLIP-MIT.txt`](OPENAI-CLIP-MIT.txt).
