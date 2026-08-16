from __future__ import annotations

from pathlib import Path
import re
import tomllib
from typing import Any

from rclip.compliance.common import ComplianceError
from rclip.compliance.common import _required_string
from rclip.compliance.common import normalize_python_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _find_policy() -> Path:
  source_policy = PROJECT_ROOT / "compliance" / "policy.toml"
  candidates = [Path.cwd() / "compliance" / "policy.toml", source_policy]
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  raise ComplianceError("could not find compliance/policy.toml; pass --policy explicitly")


def _find_lock(policy_path: Path) -> Path:
  source_lock = PROJECT_ROOT / "uv.lock"
  candidates = [policy_path.resolve().parent.parent / "uv.lock", Path.cwd() / "uv.lock", source_lock]
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  raise ComplianceError("could not find uv.lock next to the distribution policy")


def _reject_unknown_keys(values: dict[str, Any], allowed: set[str], description: str) -> None:
  unknown = sorted(set(values) - allowed)
  if unknown:
    raise ComplianceError(f"{description} has unknown fields: {', '.join(unknown)}")


def _string_list(values: dict[str, Any], key: str, description: str, *, required: bool = False) -> list[str]:
  value = values.get(key)
  if value is None and not required:
    return []
  if (
    not isinstance(value, list)
    or (required and not value)
    or any(not isinstance(item, str) or not item for item in value)
  ):
    raise ComplianceError(f"{description} has invalid {key}")
  return value


def load_policy(path: Path | None) -> dict[str, Any]:
  policy_path = path or _find_policy()
  with policy_path.open("rb") as stream:
    policy = tomllib.load(stream)
  _reject_unknown_keys(
    policy,
    {
      "schema_version",
      "prohibited_python_packages",
      "unversioned_python_packages",
      "prohibited_runtime_files",
      "approved_python_licenses",
      "codecs",
      "copyleft",
    },
    f"policy in {policy_path}",
  )
  schema_version = policy.get("schema_version")
  if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
    raise ComplianceError(f"unsupported policy schema in {policy_path}")
  for key in ("prohibited_python_packages", "unversioned_python_packages", "prohibited_runtime_files"):
    _string_list(policy, key, f"policy in {policy_path}", required=True)
  approved_licenses = policy.get("approved_python_licenses")
  if not isinstance(approved_licenses, dict) or not approved_licenses:
    raise ComplianceError(f"policy in {policy_path} has invalid approved_python_licenses")
  normalized_names: set[str] = set()
  for package_name, expression in approved_licenses.items():
    if not isinstance(package_name, str) or not package_name:
      raise ComplianceError(f"policy in {policy_path} has an invalid Python package name")
    normalized_name = normalize_python_name(package_name)
    if normalized_name in normalized_names:
      raise ComplianceError(f"policy in {policy_path} has duplicate normalized Python package names")
    normalized_names.add(normalized_name)
    if not isinstance(expression, str) or not expression:
      raise ComplianceError(f"{package_name} Python package has invalid approved licence expression")
  unversioned = {normalize_python_name(name) for name in policy["unversioned_python_packages"]}
  missing_unversioned = sorted(unversioned - normalized_names)
  if missing_unversioned:
    raise ComplianceError(
      "unversioned Python distributions have no approved licence: " + ", ".join(missing_unversioned)
    )
  codecs = policy.get("codecs")
  if not isinstance(codecs, dict) or not codecs:
    raise ComplianceError(f"policy in {policy_path} is missing codecs")
  for codec_name, codec_policy in codecs.items():
    if not isinstance(codec_name, str) or not codec_name or not isinstance(codec_policy, dict):
      raise ComplianceError(f"{codec_name} codec policy must be a table")
    _reject_unknown_keys(
      codec_policy,
      {
        "allowed",
        "required_notice",
        "required_text",
        "required_sha256",
        "native_components",
        "forbidden_filename_patterns",
        "forbidden_package_patterns",
        "forbidden_binary_markers",
      },
      f"{codec_name} codec policy",
    )
    if not isinstance(codec_policy.get("allowed"), bool):
      raise ComplianceError(f"{codec_name} codec policy has invalid allowed")
    for key in ("required_notice", "required_text", "required_sha256"):
      value = codec_policy.get(key)
      if value is not None and (not isinstance(value, str) or not value):
        raise ComplianceError(f"{codec_name} codec policy has invalid {key}")
    required_sha256 = codec_policy.get("required_sha256")
    if required_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", required_sha256):
      raise ComplianceError(f"{codec_name} codec policy has invalid required_sha256")
    for key in ("forbidden_filename_patterns", "forbidden_package_patterns", "forbidden_binary_markers"):
      _string_list(codec_policy, key, f"{codec_name} codec policy")
    native_components = codec_policy.get("native_components", [])
    if not isinstance(native_components, list):
      raise ComplianceError(f"{codec_name} native_components must be an array")
    for component in native_components:
      if not isinstance(component, dict):
        raise ComplianceError(f"{codec_name} native component must be a table")
      _reject_unknown_keys(
        component,
        {"name", "version", "path_markers", "binary_markers"},
        f"{codec_name} native component",
      )
      _required_string(component, "name", f"{codec_name} native component")
      _required_string(component, "version", f"{codec_name} native component")
      path_markers = _string_list(component, "path_markers", f"{codec_name} native component")
      binary_markers = _string_list(component, "binary_markers", f"{codec_name} native component")
      if not path_markers and not binary_markers:
        raise ComplianceError(f"{codec_name} native component has no evidence markers")
  copyleft = policy.get("copyleft")
  if not isinstance(copyleft, dict) or not copyleft:
    raise ComplianceError(f"policy in {policy_path} has invalid copyleft")
  for component_name, component in copyleft.items():
    if not isinstance(component_name, str) or not component_name or not isinstance(component, dict):
      raise ComplianceError(f"{component_name} copyleft policy must be a table")
    _reject_unknown_keys(
      component, {"license", "must_be_shared", "source_component"}, f"{component_name} copyleft policy"
    )
    _required_string(component, "license", f"{component_name} copyleft policy")
    _required_string(component, "source_component", f"{component_name} copyleft policy")
    if not isinstance(component.get("must_be_shared"), bool):
      raise ComplianceError(f"{component_name} copyleft policy has invalid must_be_shared")
  return policy


def _locked_runtime_versions(path: Path) -> dict[str, set[str]]:
  with path.open("rb") as stream:
    data = tomllib.load(stream)
  packages = data.get("package")
  if not isinstance(packages, list):
    raise ComplianceError(f"lock file {path} has no package list")
  packages_by_name: dict[str, list[dict[str, Any]]] = {}
  for package in packages:
    if not isinstance(package, dict):
      raise ComplianceError(f"lock file {path} has an invalid package")
    name = package.get("name")
    version = package.get("version")
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
      raise ComplianceError(f"lock file {path} has a package without a name or version")
    packages_by_name.setdefault(normalize_python_name(name), []).append(package)
  if "rclip" not in packages_by_name:
    raise ComplianceError(f"lock file {path} has no rclip package")

  versions: dict[str, set[str]] = {}
  pending = ["rclip"]
  while pending:
    name = pending.pop()
    if name in versions:
      continue
    locked = packages_by_name.get(name)
    if locked is None:
      raise ComplianceError(f"lock file {path} is missing runtime dependency {name}")
    versions[name] = {str(package["version"]) for package in locked}
    for package in locked:
      dependencies = package.get("dependencies", [])
      if not isinstance(dependencies, list):
        raise ComplianceError(f"lock file {path} has invalid dependencies for {name}")
      for dependency in dependencies:
        dependency_name = dependency.get("name") if isinstance(dependency, dict) else None
        if not isinstance(dependency_name, str) or not dependency_name:
          raise ComplianceError(f"lock file {path} has an invalid dependency for {name}")
        pending.append(normalize_python_name(dependency_name))
  return versions


def _required_codec_text(policy: dict[str, Any], codec_name: str) -> str:
  value = policy.get("codecs", {}).get(codec_name, {}).get("required_text")
  if not isinstance(value, str) or not value:
    raise ComplianceError(f"{codec_name} policy is missing required_text")
  return value
