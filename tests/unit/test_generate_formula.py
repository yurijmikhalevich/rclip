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


def test_formula_uses_explicit_compliance_inputs() -> None:
  formula = generate_formula.TEMPLATE.render(
    package={"url": "https://example.test/rclip.tar.gz", "checksum": "sha256"},
    resources="",
    target_python_version=generate_formula.TARGET_PYTHON_VERSION,
    wheel_resources="",
    wheel_names="",
    wheel_packages=[],
    include_macos_wheel_resource=False,
    macos_wheel_resource="",
  )

  assert '"--policy", buildpath/"compliance/policy.toml"' in formula
  assert '"--common-notices", buildpath/"compliance/notices"' in formula


@pytest.mark.parametrize("macos", [False, True])
def test_dependency_graph_preserves_extras(monkeypatch: pytest.MonkeyPatch, macos: bool):
  from types import SimpleNamespace

  dependencies = {
    "root": ["markdown-it-py", "textual"],
    "textual": ["markdown-it-py[linkify]"],
    "markdown-it-py": [
      "mdurl",
      'linkify-it-py[unicode]; extra == "linkify"',
      'unrequested; extra == "plugins"',
      'wrong-platform; extra == "linkify" and sys_platform == "win32"',
      'base-only; extra != "linkify"',
    ],
    "mdurl": [],
    "base-only": [],
    "linkify-it-py": ['unicode-helper; extra == "unicode"', "markdown-it-py[linkify]"],
    "unicode-helper": [],
  }
  fetched = []

  def distribution(name):
    return SimpleNamespace(metadata={"Name": name, "Version": "1.0"}, requires=dependencies[name])

  def get(url, timeout):
    name = url.split("/")[-3]
    fetched.append(name)
    return SimpleNamespace(
      raise_for_status=lambda: None,
      json=lambda: {
        "urls": [{"packagetype": "sdist", "url": f"https://example.test/{name}.tar.gz", "digests": {"sha256": "hash"}}],
        "info": {"home_page": ""},
      },
    )

  monkeypatch.setattr(generate_formula.importlib.metadata, "distribution", distribution)
  monkeypatch.setattr(generate_formula.requests, "get", get)
  if macos:
    monkeypatch.setattr(generate_formula, "EXTRA_MACOS_RESOURCES", ["root"])
    graph = generate_formula.get_macos_only_resources()
  else:
    graph = generate_formula.make_graph("root", set())

  assert set(graph) == set(dependencies)
  assert sorted(fetched) == sorted(dependencies)


def test_dependency_graph_fails_for_missing_distribution(monkeypatch: pytest.MonkeyPatch):
  def distribution(name):
    raise generate_formula.importlib.metadata.PackageNotFoundError(name)

  monkeypatch.setattr(generate_formula.importlib.metadata, "distribution", distribution)

  with pytest.raises(generate_formula.importlib.metadata.PackageNotFoundError):
    generate_formula.make_graph("missing", set())


def test_formula_checks_dependencies_after_wheel_installation():
  formula = generate_formula.TEMPLATE.render(
    package={"url": "https://example.test/rclip.tar.gz", "checksum": "sha256"},
    resources="",
    target_python_version=generate_formula.TARGET_PYTHON_VERSION,
    wheel_resources="",
    wheel_names="rawpy",
    wheel_packages=[{"name": "rawpy"}],
    include_macos_wheel_resource=True,
    macos_wheel_resource="coremltools",
  )

  check = 'system "python3.13", "-m", "pip", "--python=#{libexec}/bin/python", "check"'
  assert formula.rindex('"install", "--no-deps", valid_wheel') < formula.index(check)
  assert formula.index(check) < formula.index('"rclip._compliance", "collect"')
