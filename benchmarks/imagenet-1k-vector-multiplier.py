from tqdm import tqdm
from rclip import db, utils

file_prefix = '/Users/yurij/datasets/imagenet-1k/data'

datadir = utils.get_app_datadir()
database = db.DB(datadir / 'db.sqlite3')

for image in tqdm(database.get_image_vectors_by_dir_path(file_prefix)):
    for i in range(1, 10):
      new_image: db.NewImage = {
        'filepath': file_prefix + str(i) + image['filepath'][len(file_prefix):],
        'modified_at': 0,
        'size': 0,
        'vector': image['vector'],
      }
      database.upsert_image(new_image)
