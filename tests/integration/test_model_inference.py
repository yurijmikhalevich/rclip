import numpy as np
import numpy.typing as npt
from PIL import Image

from rclip.model import Model


FeatureBatch = npt.NDArray[np.float32]


def _assert_unit_norm(features: FeatureBatch) -> None:
  norms = np.linalg.norm(features, axis=1)
  assert np.allclose(norms, 1.0, atol=1e-5), f"Expected unit norm, got {norms}"


def test_real_image_features_shape_and_normalization() -> None:
  model = Model()
  img = Image.new("RGB", (100, 100), color="red")

  features = model.compute_image_features([img])

  assert features.shape == (1, Model.VECTOR_SIZE)
  _assert_unit_norm(features)


def test_real_text_features_shape_and_normalization() -> None:
  model = Model()

  features = model.compute_text_features(["a photo of a cat"])

  assert features.shape == (1, Model.VECTOR_SIZE)
  _assert_unit_norm(features)


def test_real_batch_text_features() -> None:
  model = Model()

  features = model.compute_text_features(["cat", "dog", "bird"])

  assert features.shape == (3, Model.VECTOR_SIZE)
  _assert_unit_norm(features)
