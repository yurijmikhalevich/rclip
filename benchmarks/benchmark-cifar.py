import os
from typing import Iterable, List, Tuple, cast
from PIL import Image
import numpy as np
from tqdm import tqdm
from benchmarks.config import DATASET_DIR, BATCH_SIZE

from rclip import model

from torchvision.datasets import CIFAR100


def main():
    cifar100 = CIFAR100(root=os.path.join(DATASET_DIR, 'cifar100'), download=True, train=False)
    model_instance = model.Model()
    class_description_vectors = model_instance.compute_text_features(cifar100.classes)

    processed = 0
    top1_match = 0
    top5_match = 0
    batch = []

    def process_batch():
        nonlocal processed, top1_match, top5_match, batch

        images, target_classes = zip(*batch)
        batch = []

        image_features = model_instance.compute_image_features(cast(List[Image.Image], images))

        similarities = image_features @ class_description_vectors.T
        ordered_predicted_classes = np.argsort(similarities, axis=1)

        target_classes_np = np.array(target_classes)
        top1_match += np.sum(target_classes_np == ordered_predicted_classes[:, -1])
        top5_match += np.sum(np.any(target_classes_np.reshape(-1, 1) == ordered_predicted_classes[:, -5:], axis=1))

        processed += len(images)

    for item in tqdm(cast(Iterable[Tuple[Image.Image, int]], cifar100)):
        batch.append(item)
        if len(batch) < BATCH_SIZE:
            continue
        process_batch()

    if len(batch) > 0:
        process_batch()

    print(f'Processed: {processed}')
    print(f'Top-1 accuracy: {top1_match / processed}')
    print(f'Top-5 accuracy: {top5_match / processed}')


if __name__ == '__main__':
    main()
