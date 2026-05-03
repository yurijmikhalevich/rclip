import importlib.util
from pathlib import Path

import pytest


def _load_generate_formula_module():
  module_path = Path(__file__).resolve().parents[2] / "release-utils/homebrew/generate_formula.py"
  spec = importlib.util.spec_from_file_location("generate_formula", module_path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


generate_formula = _load_generate_formula_module()


def test_get_marker_environment_uses_homebrew_target_python(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(
    generate_formula,
    "default_environment",
    lambda: {
      "python_version": "3.11",
      "python_full_version": "3.11.12",
      "platform_python_implementation": "PyPy",
      "implementation_name": "pypy",
      "implementation_version": "3.11.12",
      "sys_platform": "darwin",
      "platform_system": "Darwin",
      "platform_machine": "arm64",
    },
  )

  marker_env = generate_formula.get_marker_environment({"sys_platform": "linux", "platform_system": "Linux"})

  assert marker_env["python_version"] == generate_formula.TARGET_PYTHON_VERSION
  assert marker_env["python_full_version"] == generate_formula.TARGET_PYTHON_FULL_VERSION
  assert marker_env["platform_python_implementation"] == "CPython"
  assert marker_env["implementation_name"] == "cpython"
  assert marker_env["implementation_version"] == generate_formula.TARGET_PYTHON_FULL_VERSION
  assert marker_env["sys_platform"] == "linux"
  assert marker_env["platform_system"] == "Linux"
  assert marker_env["platform_machine"] == "arm64"
