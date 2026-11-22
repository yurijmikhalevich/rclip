"""Builds the benchmark libraries: camera-sized JPEGs and small JPEGs, each with distinct content."""

import glob
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

from PIL import Image, ImageEnhance

out, photos, small_images = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sources = sorted(glob.glob("tests/e2e/images/*.jp*g") + glob.glob("tests/e2e/images/*.JPG"))


def variant(index: int, size: tuple[int, int]) -> Image.Image:
  random.seed(index)
  image = Image.open(sources[index % len(sources)]).convert("RGB")
  width, height = image.size
  # a random crop, flip, and tone make every file a different image with a different hash
  left, top = random.randint(0, width // 5), random.randint(0, height // 5)
  image = image.crop((left, top, width - random.randint(0, width // 5), height - random.randint(0, height // 5)))
  if random.random() < 0.5:
    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
  image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
  image = ImageEnhance.Color(image).enhance(random.uniform(0.6, 1.4))
  return image.resize(size, Image.Resampling.BICUBIC)


def photo(index: int) -> None:
  variant(index, (4032, 3024)).save(f"{out}/photos/IMG_{index:04}.jpg", quality=88, subsampling="4:2:0")


def small(index: int) -> None:
  variant(index, (640, 480)).save(f"{out}/small/{index % 20:02}/img_{index:05}.jpg", quality=80)


if __name__ == "__main__":
  os.makedirs(f"{out}/photos")
  for directory in range(20):
    os.makedirs(f"{out}/small/{directory:02}")
  with ProcessPoolExecutor() as pool:
    list(pool.map(photo, range(photos), chunksize=8))
    list(pool.map(small, range(small_images), chunksize=64))
