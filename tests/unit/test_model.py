from rclip.model import Model


def test_extract_query_multiplier():
  assert Model._extract_query_multiplier('1.5:cat') == (1.5, 'cat')  # type: ignore
  assert Model._extract_query_multiplier('cat') == (1., 'cat')  # type: ignore
  assert Model._extract_query_multiplier('1:cat') == (1., 'cat')  # type: ignore
  assert Model._extract_query_multiplier('0.5:cat') == (0.5, 'cat')  # type: ignore
  assert Model._extract_query_multiplier('.5:cat') == (0.5, 'cat')  # type: ignore
  assert Model._extract_query_multiplier('1.:cat') == (1., 'cat')  # type: ignore
  assert Model._extract_query_multiplier('1..:cat') == (1., '1..:cat')  # type: ignore
  assert Model._extract_query_multiplier('..:cat') == (1., '..:cat')  # type: ignore
  assert Model._extract_query_multiplier('whatever:cat') == (1., 'whatever:cat')  # type: ignore
  assert (Model._extract_query_multiplier('1.5:complex and long query') ==  # type: ignore
          (1.5, 'complex and long query'))
