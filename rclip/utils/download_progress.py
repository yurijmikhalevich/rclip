from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tqdm import tqdm

if TYPE_CHECKING:
  _TqdmBase = tqdm[Any]
else:
  _TqdmBase = tqdm


class AggregatedProgressBar(_TqdmBase):
  """Pass as ``tqdm_class`` to HF download functions.

  Byte-level ``update()`` calls are forwarded to the class-level
  ``shared_bar``; everything else (file-count bars from ``thread_map``,
  ``total`` adjustments) is silently consumed.

  Set ``shared_bar`` to a visible ``tqdm`` instance before use.
  """

  shared_bar: tqdm[Any] | None = None

  def __init__(self, *args: object, **kwargs: object):
    self._is_bytes = kwargs.get("unit") == "B"
    kwargs["disable"] = True
    super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # proxying tqdm's complex constructor

  def update(self, n: float | None = 1) -> bool | None:
    updated = super().update(n)
    if self._is_bytes and self.shared_bar is not None:
      self.shared_bar.update(n)
    return updated
