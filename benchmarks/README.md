# Benchmarks

Accuracy benchmarks for the model and the search pipeline. They are the check to run after changing
the model, its conversion, or the image preprocessing.

```bash
cargo run --release --example benchmark -- cifar100
cargo run --release --example benchmark -- imagenet
cargo run --release --example benchmark -- similarity-search
```

- `cifar100` classifies the 10,000 CIFAR-100 test images by their class names.
- `imagenet` indexes the ImageNet-1k training images and classifies them by `photo of <class description>`.
- `similarity-search` indexes the same images, searches by the first ten images of every class, and
  reports the share of the 100 best matches that belong to the class of the query image.

Several benchmarks can run at once: `benchmark imagenet similarity-search` indexes ImageNet only once.

The benchmarks use the model from rclip's data directory and read the datasets from
`benchmarks/datasets`; set `BENCHMARK_DATASET_DIR` to use another directory. `BENCHMARK_BATCH_SIZE`
changes the batch size from the default of 256.

## Datasets

- **CIFAR-100**: download the binary version from <https://www.cs.toronto.edu/~kriz/cifar.html> and
  extract it, so that the test images are at `benchmarks/datasets/cifar-100-binary/test.bin`.
- **ImageNet-1k**: clone <https://huggingface.co/datasets/imagenet-1k> into
  `benchmarks/datasets/imagenet_1k` and extract the `train_images_*.tar.gz` archives under
  `benchmarks/datasets/imagenet_1k/data`.

## Results

| Benchmark | Metric | Result |
| --- | --- | --- |
| `cifar100` | top-1 accuracy | 0.8072 |
| `cifar100` | top-5 accuracy | 0.9609 |
| `imagenet` | top-1 accuracy | 0.6874 |
| `imagenet` | top-5 accuracy | 0.8958 |
| `similarity-search` | accuracy | 0.5980 |

Measured with the ViT-B-32-256 DataComp-1B model on an M1 Max. The Python version of rclip, which
these benchmarks were ported from, scored 0.8100 and 0.9608 on CIFAR-100 and 0.6874 and 0.8959 on
ImageNet-1k. The small differences come from resizing images with `fast_image_resize` instead of
Pillow; CIFAR-100's 32x32 images, which are enlarged eightfold, show them the most.
