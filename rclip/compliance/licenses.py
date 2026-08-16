from __future__ import annotations

from collections.abc import Iterable
import email.parser
from pathlib import Path
from typing import Any

from rclip.compliance.common import ComplianceError
from rclip.compliance.common import _confined_file
from rclip.compliance.common import _is_inside
from rclip.compliance.common import normalize_python_name


LEGAL_PREFIXES = (
  "authors",
  "copying",
  "copyright",
  "licence",
  "license",
  "notice",
  "patents",
  "thirdpartynotice",
)
LEGACY_LICENSE_ALIASES = {
  "3-Clause BSD License": "BSD-3-Clause",
  "Apache 2.0": "Apache-2.0",
  "BSD": "BSD-3-Clause",
  "MIT License": "MIT",
}
LEGACY_SPDX_EXPRESSIONS = {
  "Apache-2.0",
  "BSD-3-Clause",
  "MIT",
  "MPL-2.0",
  "MPL-2.0 AND MIT",
  "WTFPL",
}
LICENSE_CLASSIFIER_EXPRESSIONS = {
  "License :: OSI Approved :: Apache Software License": "Apache-2.0",
  "License :: OSI Approved :: BSD License": "BSD-3-Clause",
  "License :: OSI Approved :: MIT License": "MIT",
  "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
}


def _metadata_records(root: Path, excluded_root: Path | None = None) -> list[dict[str, Any]]:
  records: dict[tuple[str, str], dict[str, Any]] = {}
  parser = email.parser.Parser()
  for metadata_path in sorted(root.rglob("*.dist-info/METADATA")):
    if excluded_root is not None and _is_inside(metadata_path, excluded_root):
      continue
    metadata = parser.parsestr(metadata_path.read_text(encoding="utf-8", errors="replace"))
    display_name = metadata.get("Name")
    version = metadata.get("Version")
    if not display_name or not version:
      raise ComplianceError(f"missing Name or Version in {metadata_path}")
    name = normalize_python_name(display_name)
    records[(name, version)] = {
      "name": name,
      "display_name": display_name,
      "version": version,
      "dist_info": metadata_path.parent,
      "declared_license_files": metadata.get_all("License-File", []),
      "declared_license_expression": metadata.get("License-Expression", ""),
      "legacy_license": metadata.get("License", ""),
      "license_classifiers": [
        value for value in metadata.get_all("Classifier", []) if value.startswith("License ::")
      ],
    }
  return [records[key] for key in sorted(records)]


def _approved_python_licenses(policy: dict[str, Any]) -> dict[str, str]:
  return {
    normalize_python_name(name): str(expression)
    for name, expression in policy.get("approved_python_licenses", {}).items()
  }


def _validate_python_packages(
  records: Iterable[dict[str, Any]], policy: dict[str, Any], locked_versions: dict[str, set[str]]
) -> None:
  records = list(records)
  prohibited = {normalize_python_name(name) for name in policy.get("prohibited_python_packages", [])}
  approved = _approved_python_licenses(policy)
  unversioned = {normalize_python_name(name) for name in policy.get("unversioned_python_packages", [])}
  installed = {record["name"] for record in records}
  rejected = sorted(installed & prohibited)
  unknown = sorted(installed - approved.keys())
  errors = []
  if rejected:
    errors.append(f"prohibited Python distributions: {', '.join(rejected)}")
  if unknown:
    errors.append(f"disallowed Python distributions: {', '.join(unknown)}")
  missing_approvals = sorted(locked_versions.keys() - approved.keys())
  stale_approvals = sorted(approved.keys() - locked_versions.keys() - unversioned)
  if missing_approvals:
    errors.append(f"locked Python distributions have no approved licence: {', '.join(missing_approvals)}")
  if stale_approvals:
    errors.append(f"approved Python distributions are absent from the runtime lock: {', '.join(stale_approvals)}")
  version_drift = []
  for record in records:
    expected = None if record["name"] in unversioned else locked_versions.get(record["name"])
    actual = record.get("version")
    if expected is not None and str(actual) not in expected:
      version_drift.append(f"{record['name']} {actual or '<missing>'} (locked: {', '.join(sorted(expected))})")
  if version_drift:
    errors.append(f"disallowed Python versions: {', '.join(sorted(version_drift))}")
  if errors:
    raise ComplianceError("; ".join(errors))


def _declared_license_expression(record: dict[str, Any]) -> str:
  expression = " ".join(str(record.get("declared_license_expression", "")).split())
  if expression:
    return expression
  legacy = str(record.get("legacy_license", "")).strip()
  if legacy in LEGACY_SPDX_EXPRESSIONS:
    return legacy
  expression = LEGACY_LICENSE_ALIASES.get(legacy, "")
  if expression:
    return expression
  expressions = {
    LICENSE_CLASSIFIER_EXPRESSIONS[classifier]
    for classifier in record.get("license_classifiers", [])
    if classifier in LICENSE_CLASSIFIER_EXPRESSIONS
  }
  return expressions.pop() if len(expressions) == 1 else ""


def _validate_python_licenses(records: Iterable[dict[str, Any]], policy: dict[str, Any]) -> None:
  approved = _approved_python_licenses(policy)
  errors = []
  for record in records:
    expected = approved.get(record["name"])
    if expected is None:
      continue
    actual = _declared_license_expression(record)
    if not actual:
      errors.append(f"unknown Python licence declaration: {record['name']} {record['version']}")
    elif actual != expected:
      errors.append(
        f"unapproved Python licence: {record['name']} {record['version']} declares {actual} (approved: {expected})"
      )
  if errors:
    raise ComplianceError("; ".join(errors))


def _legal_files(record: dict[str, Any], policy_dir: Path) -> list[Path]:
  dist_info: Path = record["dist_info"]
  result: set[Path] = set()
  for path in dist_info.rglob("*"):
    if not _confined_file(path, dist_info):
      continue
    relative = path.relative_to(dist_info)
    if "licenses" in {part.lower() for part in relative.parts[:-1]} or path.name.lower().startswith(LEGAL_PREFIXES):
      result.add(path)
  for declared in record["declared_license_files"]:
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
      raise ComplianceError(f"{record['display_name']} declares unsafe licence file {declared!r}")
    candidates = [(dist_info / "licenses" / relative, dist_info), (dist_info / relative, dist_info)]
    if record["name"] == "rclip":
      candidates.append((policy_dir.parent / relative, policy_dir.parent))
    match = next((candidate for candidate, parent in candidates if _confined_file(candidate, parent)), None)
    if match is None:
      raise ComplianceError(f"{record['display_name']} declares missing licence file {declared!r}")
    result.add(match)
  package_names = {record["name"].replace("-", "_"), record["name"].replace("-", "")}
  for package_name in package_names:
    package_dir = dist_info.parent / package_name
    if not package_dir.is_dir():
      continue
    for path in package_dir.iterdir():
      if _confined_file(path, package_dir) and path.name.lower().startswith(LEGAL_PREFIXES):
        result.add(path)
  override_dir = policy_dir / "license-overrides" / record["name"] / record["version"]
  if override_dir.is_dir():
    result.update(path for path in override_dir.rglob("*") if _confined_file(path, override_dir))
  return sorted(result)
