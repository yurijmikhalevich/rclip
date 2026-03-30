import numpy as np
from rclip.model import Model


def test_extract_query_multiplier():
  assert Model._extract_query_multiplier("1.5:cat") == (1.5, "cat")  # type: ignore
  assert Model._extract_query_multiplier("cat") == (1.0, "cat")  # type: ignore
  assert Model._extract_query_multiplier("1:cat") == (1.0, "cat")  # type: ignore
  assert Model._extract_query_multiplier("0.5:cat") == (0.5, "cat")  # type: ignore
  assert Model._extract_query_multiplier(".5:cat") == (0.5, "cat")  # type: ignore
  assert Model._extract_query_multiplier("1.:cat") == (1.0, "cat")  # type: ignore
  assert Model._extract_query_multiplier("1..:cat") == (1.0, "1..:cat")  # type: ignore
  assert Model._extract_query_multiplier("..:cat") == (1.0, "..:cat")  # type: ignore
  assert Model._extract_query_multiplier("whatever:cat") == (1.0, "whatever:cat")  # type: ignore
  assert (
    Model._extract_query_multiplier("1.5:complex and long query")  # type: ignore
    == (1.5, "complex and long query")
  )


def test_sessions_are_lazy():
  model = Model()
  assert model._session_text_var is None  # pyright: ignore[reportPrivateUsage]
  assert model._session_visual_var is None  # pyright: ignore[reportPrivateUsage]


def test_text_features_are_normalized():
  model = Model()
  features = model.compute_text_features(["a photo of a cat"])
  norms = np.linalg.norm(features, axis=-1)
  np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_text_features_shape():
  model = Model()
  features = model.compute_text_features(["hello", "world"])
  assert features.shape == (2, Model.VECTOR_SIZE)
  assert features.dtype == np.float32
