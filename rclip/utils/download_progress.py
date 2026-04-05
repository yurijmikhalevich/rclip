from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from tqdm import tqdm as tqdm_base

if TYPE_CHECKING:
  _Base = tqdm_base[NoReturn]
else:
  _Base = tqdm_base


class AggregatedProgressBar(_Base):
  """Pass as ``tqdm_class`` to HF download functions.

  Byte-level ``update()`` calls are forwarded to the class-level
  ``shared_bar``; everything else (file-count bars from ``thread_map``,
  ``total`` adjustments) is silently consumed.

  Set ``shared_bar`` to a visible ``tqdm`` instance before use.
  """

  shared_bar: tqdm_base[NoReturn] | None = None

  def __init__(self, *args: object, **kwargs: object):
    self._is_bytes = kwargs.get("unit") == "B"
    kwargs["disable"] = True
    super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # proxying tqdm's complex constructor

  def update(self, n: float | None = 1) -> bool | None:
    ret = super().update(n)
    if self._is_bytes and self.shared_bar is not None:
      self.shared_bar.update(n)
    return ret
