import numpy as np
from tqdm import tqdm

from rclip import model, utils, db

from torchvision.datasets import CIFAR100

cifar100 = CIFAR100(root='/Users/yurij/datasets/cifar100', download=True, train=False)

model_instance = model.Model()
datadir = utils.get_app_datadir()
database = db.DB(datadir / 'db.sqlite3')

text_vectors = []
for text in cifar100.classes:
    text_vectors.append(model_instance.compute_text_features([text])[0])

text_vectors_np = np.stack(text_vectors)


processed = 0
top1_match = 0
top5_match = 0


for image in tqdm(cifar100):
    image_features = model_instance.compute_image_features([image[0]])[0]

    similarity = text_vectors_np @ image_features.T
    sorted_similarities = sorted(zip(similarity, range(image_features.shape[0])), key=lambda x: x[0], reverse=True)

    image_class = image[1]

    if image_class == sorted_similarities[0][1]:
        top1_match += 1
        top5_match += 1
    elif image_class in [sorted_similarities[1][1], sorted_similarities[2][1], sorted_similarities[3][1], sorted_similarities[4][1]]:
        top5_match += 1

    processed += 1

    # if processed == 10:
    #     break


print(f'Processed: {processed}')
print(f'Top-1 accuracy: {top1_match / processed}')
print(f'Top-5 accuracy: {top5_match / processed}')
