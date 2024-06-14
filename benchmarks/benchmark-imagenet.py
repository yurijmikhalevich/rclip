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
    TEST_IMAGE_PREFIX = os.path.join(DATASET_DIR, 'imagenet_1k', 'data')

    model_instance = model.Model()
    database = db.DB(os.path.join(tmp_datadir, 'db.sqlite3'))
    rclip = RClip(model_instance, database, BATCH_SIZE, None)

    rclip.ensure_index(TEST_IMAGE_PREFIX)

    ids_list, descriptions = zip(*IMAGENET2012_CLASSES.items())
    class_description_vectors = model_instance.compute_text_features(
        [f'photo of {description}' for description in descriptions]
    )
    ids = np.array(ids_list)

    def get_image_class(filepath: str):
        return filepath.split('/')[-1].split('_')[0]

    processed = 0
    top1_match = 0
    top5_match = 0
    batch = []

    def process_batch():
        nonlocal processed, top1_match, top5_match, batch

        image_features = np.stack([np.frombuffer(image['vector'], np.float32) for image in batch])

        similarities = image_features @ class_description_vectors.T
        ordered_predicted_classes = np.argsort(similarities, axis=1)

        target_classes = np.array([get_image_class(image['filepath']) for image in batch])
        top1_match += np.sum(target_classes == ids[ordered_predicted_classes[:, -1]])
        top5_match += np.sum(np.any(target_classes.reshape(-1, 1) == ids[ordered_predicted_classes[:, -5:]], axis=1))

        processed += len(batch)

        batch = []

    for image in tqdm(database.get_image_vectors_by_dir_path(TEST_IMAGE_PREFIX)):
        batch.append(image)
        if len(batch) < BATCH_SIZE:
            continue
        process_batch()

    if len(batch) > 0:
        process_batch()

    print(f'Processed: {processed}')
    print(f'Top-1 accuracy: {top1_match / processed}')
    print(f'Top-5 accuracy: {top5_match / processed}')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f'Using temporary directory: {tmp_dir}')
        main(tmp_dir)
