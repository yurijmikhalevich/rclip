"""Builds a synthetic index of random unit vectors for images under a directory, in rclip's schema."""

import sys

import numpy as np

from rclip.db import DB, NewImage

path, directory, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
database = DB(path)
rng = np.random.default_rng(0)
for start in range(0, count, 10_000):
  vectors = rng.standard_normal((min(10_000, count - start), 512), dtype=np.float32)
  vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
  for offset, vector in enumerate(vectors):
    index = start + offset
    database.upsert_image(
      NewImage(
        filepath=f"{directory}/{index % 1000:03}/IMG_{index:07}.jpg",
        modified_at=1_700_000_000.0 + index,
        size=3_000_000 + index,
        vector=vector.tobytes(),
        hash=f"{index:064x}",
      ),
      commit=False,
    )
  database.commit()
database.close()
