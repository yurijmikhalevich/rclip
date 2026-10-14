import threading

import pytest

from rclip.utils import helpers


def test_iter_in_background_yields_every_item_in_order():
  assert list(helpers.iter_in_background(range(1000), max_buffered=64, chunk_size=7)) == list(range(1000))


def test_iter_in_background_raises_errors_after_the_items_before_them():
  def items():
    yield 1
    yield 2
    raise PermissionError("denied")

  received = []
  with pytest.raises(PermissionError, match="denied"):
    for item in helpers.iter_in_background(items(), max_buffered=8, chunk_size=4):
      received.append(item)
  assert received == [1, 2]


def test_iter_in_background_stops_producing_when_the_consumer_stops():
  produced = 0
  finished = threading.Event()

  def items():
    nonlocal produced
    try:
      while True:
        produced += 1
        yield produced
    finally:
      finished.set()

  background = helpers.iter_in_background(items(), max_buffered=4, chunk_size=2)
  assert next(background) == 1
  background.close()

  assert finished.wait(timeout=5)
  assert produced < 100
