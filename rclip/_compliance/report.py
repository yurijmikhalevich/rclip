from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rclip._compliance.common import _confined_file
from rclip._compliance.common import _sha256
from rclip._compliance.common import normalize_python_name
from rclip._compliance.licenses import _approved_python_licenses
from rclip._compliance.policy import load_policy


def _reported_python_packages(legal_dir: Path) -> list[dict[str, str]]:
  report_path = legal_dir / "compliance-report.json"
  if not report_path.is_file():
    return []
  data = json.loads(report_path.read_text(encoding="utf-8"))
  packages: dict[tuple[str, str], dict[str, str]] = {}
  for component in data.get("components", []):
    if component.get("type", "python") != "python":
      continue
    name = normalize_python_name(str(component.get("name", "")))
    if not name or name == "cpython":
      continue
    version = str(component.get("version", ""))
    packages[(name, version)] = {"name": name, "version": version}
  return [packages[key] for key in sorted(packages)]


def _reported_native_versions(report: dict[str, Any]) -> dict[str, dict[str, str]]:
  versions = {}
  for component in report.get("native_component_versions", []):
    if not isinstance(component, dict):
      continue
    name = component.get("name")
    version = component.get("version")
    source = component.get("source")
    if all(isinstance(value, str) and value for value in (name, version, source)):
      versions[name] = {"version": version, "source": source}
  return versions


def _validate_legal_pack(legal_dir: Path, policy_path: Path) -> tuple[dict[str, Any], list[str]]:
  errors: list[str] = []
  report_path = legal_dir / "compliance-report.json"
  if not report_path.is_file():
    return {}, ["legal pack is missing compliance-report.json"]
  try:
    report = json.loads(report_path.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError) as error:
    return {}, [f"legal pack has an invalid compliance-report.json: {error}"]
  components = report.get("components")
  if report.get("schema_version") != 1 or not isinstance(components, list):
    errors.append("legal pack has an unsupported compliance report schema")
    components = []
  native_versions = report.get("native_component_versions")
  if not isinstance(native_versions, list):
    errors.append("legal pack has no native component version records")
  elif len(_reported_native_versions(report)) != len(native_versions):
    errors.append("legal pack contains an invalid native component version record")
  approved_licenses = _approved_python_licenses(load_policy(policy_path))
  for component in components:
    if not isinstance(component, dict):
      errors.append("legal pack contains an invalid component record")
      continue
    description = f"{component.get('name', '<unnamed>')} {component.get('version', '')}".rstrip()
    if component.get("type", "python") == "python":
      name = normalize_python_name(str(component.get("name", "")))
      expected_expression = approved_licenses.get(name)
      actual_expression = component.get("license_expression")
      if expected_expression is None:
        errors.append(f"legal component {description} has no approved licence")
      elif actual_expression != expected_expression:
        errors.append(
          f"legal component {description} has unapproved licence {actual_expression or '<missing>'} "
          f"(approved: {expected_expression})"
        )
    license_files = component.get("license_files")
    hashes = component.get("license_file_hashes")
    if not isinstance(license_files, list) or not license_files:
      errors.append(f"legal component {description} has no licence files")
      continue
    if not isinstance(hashes, dict):
      errors.append(f"legal component {description} has no licence hashes")
      continue
    for value in license_files:
      relative = Path(str(value))
      if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"legal component {description} has unsafe licence path {value!r}")
        continue
      target = legal_dir / relative
      if not _confined_file(target, legal_dir):
        errors.append(f"legal component {description} is missing {relative.as_posix()}")
        continue
      expected_hash = hashes.get(relative.as_posix())
      if not expected_hash or _sha256(target) != expected_hash:
        errors.append(f"legal component {description} has an altered licence file {relative.as_posix()}")
  bundled_policy = legal_dir / "policy.toml"
  if bundled_policy.is_file() and bundled_policy.read_bytes() != policy_path.read_bytes():
    errors.append("legal pack contains a policy that differs from the distribution policy")
  return report, errors
