import os
import tempfile
import numpy as np
from tqdm import tqdm
from benchmarks.config import BATCH_SIZE, DATASET_DIR

from benchmarks.datasets.imagenet_1k.classes import IMAGENET2012_CLASSES  # type: ignore

from rclip import model, db
from rclip.main import RClip


# To run this benchmark, clone imagenet-1k dataset from hf to `DATASET_DIR/imagenet_1k`
# https://huggingface.co/datasets/imagenet-1k/tree/main
# Then, untar train_images_X.tar.gz archives under `imagenet_1k/data/`
# TODO(yurij): make this script do that
# You may also need to increase the ulimit to avoid "Too many open files" error:
# `ulimit -n 1024`


def main(tmp_datadir: str):
  TEST_IMAGE_PREFIX = os.path.join(DATASET_DIR, "imagenet_1k", "data")

  model_instance = model.Model("mps")
  database = db.DB(os.path.join(tmp_datadir, "db.sqlite3"))
  rclip = RClip(model_instance, database, BATCH_SIZE, None)

  rclip.ensure_index(TEST_IMAGE_PREFIX)

  def get_images_for_class(class_id: str, limit: int = 750):
    return database._con.execute(  # type: ignore
      """
                SELECT filepath, vector FROM images WHERE filepath LIKE ? AND deleted IS NULL ORDER BY RANDOM() LIMIT ?
            """,
      (TEST_IMAGE_PREFIX + f"{os.path.sep}%{os.path.sep}{class_id}_%", limit),
    )

  def get_image_class(filepath: str):
    return filepath.split("/")[-1].split("_")[0]

  accuracies = []
  for class_id in tqdm(IMAGENET2012_CLASSES.keys()):
    images = get_images_for_class(class_id, limit=10)
    for image in images:
      results = rclip.search(image["filepath"], TEST_IMAGE_PREFIX, top_k=100)
      top100_classes = [get_image_class(result.filepath) for result in results]
      accuracies.append(np.mean(np.array(top100_classes) == class_id))

  print(f"Accuracy: {np.mean(accuracies)}")  # type: ignore


if __name__ == "__main__":
  with tempfile.TemporaryDirectory() as tmp_dir:
    print(f"Using temporary directory: {tmp_dir}")
    main(tmp_dir)
