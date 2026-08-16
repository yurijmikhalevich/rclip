from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


PYTHON_NAME_PATTERN = re.compile(r"[-_.]+")


class ComplianceError(RuntimeError):
  """Raised when a bundle does not satisfy the distribution policy."""


def normalize_python_name(name: str) -> str:
  return PYTHON_NAME_PATTERN.sub("-", name).lower()


def _json_dump(path: Path, value: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(4 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _is_inside(path: Path, parent: Path) -> bool:
  try:
    path.resolve().relative_to(parent.resolve())
    return True
  except ValueError:
    return False


def _confined_file(path: Path, parent: Path) -> bool:
  return path.is_file() and _is_inside(path, parent)


def _required_string(values: dict[str, Any], key: str, description: str) -> str:
  value = values.get(key)
  if not isinstance(value, str) or not value:
    raise ComplianceError(f"{description} is missing {key}")
  return value
