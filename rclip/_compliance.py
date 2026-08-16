"""Build-time licence collection and codec compliance checks.

This module intentionally uses only the Python standard library so release
builders can run it inside partially assembled application bundles.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import email.parser
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from typing import Any


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
NATIVE_SUFFIXES = {".dll", ".dylib", ".exe", ".pyd", ".so"}
NATIVE_MAGICS = (
  b"\x7fELF",
  b"MZ",
  b"\xca\xfe\xba\xbe",
  b"\xca\xfe\xba\xbf",
  b"\xbe\xba\xfe\xca",
  b"\xbf\xba\xfe\xca",
  b"\xce\xfa\xed\xfe",
  b"\xcf\xfa\xed\xfe",
  b"\xfe\xed\xfa\xce",
  b"\xfe\xed\xfa\xcf",
)
PYTHON_NAME_PATTERN = re.compile(r"[-_.]+")
FORBIDDEN_RAWPY_FEATURES = ("DEMOSAIC_PACK_GPL2", "DEMOSAIC_PACK_GPL3")
LEGACY_LICENSE_EXPRESSIONS = {
  "3-Clause BSD License": "BSD-3-Clause",
  "Apache 2.0": "Apache-2.0",
  "Apache-2.0": "Apache-2.0",
  "BSD": "BSD-3-Clause",
  "BSD-3-Clause": "BSD-3-Clause",
  "MIT": "MIT",
  "MIT License": "MIT",
  "MPL-2.0": "MPL-2.0",
  "MPL-2.0 AND MIT": "MPL-2.0 AND MIT",
  "WTFPL": "WTFPL",
}
LICENSE_CLASSIFIER_EXPRESSIONS = {
  "License :: OSI Approved :: Apache Software License": "Apache-2.0",
  "License :: OSI Approved :: BSD License": "BSD-3-Clause",
  "License :: OSI Approved :: MIT License": "MIT",
  "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
}


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


def _find_policy() -> Path:
  source_policy = Path(__file__).resolve().parent.parent / "compliance" / "policy.toml"
  candidates = [Path.cwd() / "compliance" / "policy.toml", source_policy]
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  raise ComplianceError("could not find compliance/policy.toml; pass --policy explicitly")


def _find_lock(policy_path: Path) -> Path:
  source_lock = Path(__file__).resolve().parent.parent / "uv.lock"
  candidates = [policy_path.resolve().parent.parent / "uv.lock", Path.cwd() / "uv.lock", source_lock]
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  raise ComplianceError("could not find uv.lock next to the distribution policy")


def _required_string(values: dict[str, Any], key: str, description: str) -> str:
  value = values.get(key)
  if not isinstance(value, str) or not value:
    raise ComplianceError(f"{description} is missing {key}")
  return value


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
    record = {
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
    records[(name, version)] = record
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
  expression = LEGACY_LICENSE_EXPRESSIONS.get(legacy, "")
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


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(source, destination)
  return {"path": destination.as_posix(), "sha256": _sha256(destination)}


def _system_legal_materials(root: Path, output: Path, excluded_source: Path | None) -> list[dict[str, Any]]:
  """Collect Debian-style copyright files included in self-contained Linux bundles."""
  components = []
  seen: set[str] = set()
  excluded_path = excluded_source.resolve() if excluded_source is not None else None
  for source in sorted(root.rglob("share/doc/*/copyright")):
    doc_dir = source.parent
    if (
      (excluded_path is not None and source.resolve() == excluded_path)
      or not _confined_file(source, root)
      or _is_inside(source, output)
    ):
      continue
    name = doc_dir.name
    if name in seen:
      continue
    seen.add(name)
    target = output / "licenses" / "system" / name / "copyright"
    copied = _copy_file(source, target)
    relative = target.relative_to(output).as_posix()
    components.append(
      {
        "type": "system",
        "name": name,
        "version": "",
        "license_files": [relative],
        "license_file_hashes": {relative: copied["sha256"]},
      }
    )
  return components


def _python_runtime_license(root: Path) -> tuple[str, Path]:
  version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
  relative_candidates = (
    Path("LICENSE.txt"),
    Path("LICENSE"),
    Path("lib") / f"python{sys.version_info.major}.{sys.version_info.minor}" / "LICENSE.txt",
    Path("share") / "doc" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "copyright",
    Path("usr") / "share" / "doc" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "copyright",
  )
  prefixes = (Path(sys.base_prefix), root)
  for prefix in prefixes:
    for relative in relative_candidates:
      candidate = prefix / relative
      if candidate.is_file():
        return version, candidate
  raise ComplianceError(f"could not find the CPython {version} runtime licence")


def _native_component_versions(installed_packages: set[str]) -> list[dict[str, str]]:
  versions: dict[str, dict[str, str]] = {}
  if "pillow" in installed_packages:
    try:
      avif = importlib.import_module("PIL._avif")
    except ImportError:
      pass
    else:
      versions["libavif"] = {
        "name": "libavif",
        "version": str(avif.libavif_version),
        "source": "PIL._avif.libavif_version",
      }
      for match in re.finditer(r"(?:^|, )([A-Za-z0-9_.+-]+) \[[^]]+\]:([^, ]+)", str(avif.codec_versions())):
        name, version = match.groups()
        name = {"aom": "libaom"}.get(name, name)
        versions[name] = {
          "name": name,
          "version": version,
          "source": "PIL._avif.codec_versions()",
        }
    try:
      webp = importlib.import_module("PIL._webp")
    except ImportError:
      pass
    else:
      versions["libwebp"] = {
        "name": "libwebp",
        "version": str(webp.webpdecoder_version),
        "source": "PIL._webp.webpdecoder_version",
      }
  if "rawpy" in installed_packages:
    try:
      rawpy = importlib.import_module("rawpy")
    except ImportError:
      pass
    else:
      flags = getattr(rawpy, "flags", None)
      if not isinstance(flags, dict):
        raise ComplianceError("rawpy does not report optional feature flags")
      for feature in FORBIDDEN_RAWPY_FEATURES:
        if feature not in flags:
          raise ComplianceError(f"rawpy does not report required feature flag {feature}")
        if flags[feature] is not False:
          raise ComplianceError(f"rawpy forbidden feature is not disabled: {feature}={flags[feature]!r}")
      versions["libraw"] = {
        "name": "libraw",
        "version": ".".join(str(part) for part in getattr(rawpy, "libraw_version")),
        "source": "rawpy.libraw_version",
      }
  return [versions[name] for name in sorted(versions)]


def collect_legal_materials(
  root: Path,
  output: Path,
  policy_path: Path | None,
  common_notices: Path | None,
  include_python_runtime: bool = False,
) -> dict[str, Any]:
  root = root.resolve()
  output = output.resolve()
  if root == output:
    raise ComplianceError("legal output directory cannot be the scanned root")
  records = _metadata_records(root, output)
  if not records:
    raise ComplianceError(f"no Python distribution metadata found under {root}")
  resolved_policy_path = policy_path or _find_policy()
  policy = load_policy(resolved_policy_path)
  locked_versions = _locked_runtime_versions(_find_lock(resolved_policy_path))
  _validate_python_packages(records, policy, locked_versions)
  _validate_python_licenses(records, policy)

  if output.exists():
    shutil.rmtree(output)
  output.mkdir(parents=True)
  components: list[dict[str, Any]] = []
  for record in records:
    component_dir = output / "licenses" / f"{record['name']}-{record['version']}"
    copied = []
    for source in _legal_files(record, resolved_policy_path.parent):
      try:
        relative = source.relative_to(record["dist_info"])
      except ValueError:
        relative = Path(source.name)
      copied.append(_copy_file(source, component_dir / relative))
    if not copied:
      raise ComplianceError(f"{record['display_name']} {record['version']} does not provide a licence file")
    components.append(
      {
        "type": "python",
        "name": record["name"],
        "version": record["version"],
        "license_expression": _declared_license_expression(record),
        "license_files": sorted(item["path"].replace(output.as_posix() + "/", "") for item in copied),
        "license_file_hashes": {
          item["path"].replace(output.as_posix() + "/", ""): item["sha256"]
          for item in sorted(copied, key=lambda x: x["path"])
        },
      }
    )

  python_license = None
  if include_python_runtime:
    python_version, python_license = _python_runtime_license(root)
    python_target = output / "licenses" / f"cpython-{python_version}" / python_license.name
    python_copied = _copy_file(python_license, python_target)
    python_relative = python_target.relative_to(output).as_posix()
    components.append(
      {
        "type": "runtime",
        "name": "cpython",
        "version": python_version,
        "license_files": [python_relative],
        "license_file_hashes": {python_relative: python_copied["sha256"]},
      }
    )
  components.extend(_system_legal_materials(root, output, python_license))

  notices_source = common_notices or resolved_policy_path.parent / "notices"
  if not notices_source.is_dir():
    notice_matches = sorted(root.rglob("AOM-PATENT-LICENSE-1.0.txt"))
    if notice_matches:
      notices_source = notice_matches[0].parent
  if not notices_source.is_dir():
    raise ComplianceError(f"common notice directory not found: {notices_source}")
  notice_names = []
  for source in sorted(notices_source.glob("*.txt")):
    _copy_file(source, output / "notices" / source.name)
    notice_names.append(source.name)
  _copy_file(resolved_policy_path, output / "policy.toml")
  sources_path = resolved_policy_path.parent / "sources.toml"
  if not sources_path.is_file():
    source_matches = sorted(root.rglob("sources.toml"))
    if source_matches:
      sources_path = source_matches[0]
  if sources_path.is_file():
    _copy_file(sources_path, output / "sources.toml")

  index_lines = [
    "rclip third-party notices",
    "=========================",
    "",
    _required_codec_text(policy, "dng"),
    "",
    "Codec patent terms",
    "------------------",
    *[f"- notices/{name}" for name in notice_names],
    "",
  ]
  rclip_component = next((component for component in components if component["name"] == "rclip"), None)
  rawpy_component = next((component for component in components if component["name"] == "rawpy"), None)
  if rawpy_component is not None and rclip_component is not None:
    source_filename = f"rawpy-{rawpy_component['version']}-corresponding-source.tar.gz"
    index_lines.extend(
      [
        "LibRaw corresponding source",
        "---------------------------",
        (
          f"https://github.com/yurijmikhalevich/rclip/releases/download/v{rclip_component['version']}/{source_filename}"
        ),
        "",
      ]
    )
  index_lines.extend(["Python distributions and runtime", "--------------------------------"])
  for component in (component for component in components if component["type"] != "system"):
    index_lines.append(f"- {component['name']} {component['version']}")
    for legal_file in sorted(set(component["license_files"])):
      index_lines.append(f"  - {legal_file}")
  system_components = [component for component in components if component["type"] == "system"]
  if system_components:
    index_lines.extend(["", "Bundled system packages", "-----------------------"])
    for component in system_components:
      index_lines.append(f"- {component['name']}")
      for legal_file in component["license_files"]:
        index_lines.append(f"  - {legal_file}")
  (output / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

  report = {
    "schema_version": 1,
    "components": components,
    "codec_notices": notice_names,
    "native_component_versions": _native_component_versions({record["name"] for record in records}),
  }
  _json_dump(output / "compliance-report.json", report)
  return report


def _native_candidates(root: Path, excluded_root: Path | None = None) -> list[Path]:
  candidates = []
  for path in root.rglob("*"):
    if not path.is_file() or (excluded_root is not None and _is_inside(path, excluded_root)):
      continue
    lower_name = path.name.lower()
    try:
      with path.open("rb") as stream:
        magic = stream.read(4)
    except OSError:
      continue
    is_native = any(magic.startswith(candidate) for candidate in NATIVE_MAGICS)
    if is_native or path.suffix.lower() in NATIVE_SUFFIXES or ".so." in lower_name or lower_name.endswith(".appimage"):
      candidates.append(path)
  return sorted(candidates)


def _binary_contains(path: Path, markers: Iterable[str], *, casefold: bool = False) -> list[str]:
  marker_bytes = [
    (marker, marker.encode("ascii").lower() if casefold else marker.encode("ascii")) for marker in markers
  ]
  overlap = max((len(value) for _, value in marker_bytes), default=1) - 1
  found: set[str] = set()
  previous = b""
  with path.open("rb") as stream:
    while current := stream.read(4 * 1024 * 1024):
      chunk = previous + current
      if casefold:
        chunk = chunk.lower()
      for marker, value in marker_bytes:
        if marker not in found and value in chunk:
          found.add(marker)
      if len(found) == len(marker_bytes):
        break
      previous = chunk[-overlap:] if overlap else b""
  return sorted(found)


def _notice_present(
  legal_dir: Path,
  filename: str,
  required_text: str | None = None,
  required_sha256: str | None = None,
) -> bool:
  path = legal_dir / "notices" / filename
  if not _confined_file(path, legal_dir):
    return False
  if required_sha256 is not None and _sha256(path) != required_sha256:
    return False
  return required_text is None or required_text in path.read_text(encoding="utf-8", errors="replace")


def _syft_python_packages(data: dict[str, Any]) -> list[dict[str, str]]:
  packages: dict[tuple[str, str], dict[str, str]] = {}
  for artifact in data.get("artifacts", []):
    artifact_type = str(artifact.get("type", "")).lower()
    language = str(artifact.get("language", "")).lower()
    purl = str(artifact.get("purl", ""))
    if artifact_type.startswith("python") or language == "python" or purl.startswith("pkg:pypi/"):
      name = artifact.get("name")
      if name:
        normalized_name = normalize_python_name(str(name))
        version = str(artifact.get("version", ""))
        packages[(normalized_name, version)] = {"name": normalized_name, "version": version}
  return [packages[key] for key in sorted(packages)]


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


def _syft_native_matches(data: dict[str, Any], patterns: Iterable[str]) -> list[str]:
  patterns = tuple(pattern.lower() for pattern in patterns)
  matches = []
  for artifact in data.get("artifacts", []):
    name = str(artifact.get("name", ""))
    version = str(artifact.get("version", ""))
    searchable = [name.lower(), str(artifact.get("purl", "")).lower()]
    for location in artifact.get("locations", []):
      if isinstance(location, dict):
        searchable.extend(str(location.get(key, "")).lower() for key in ("path", "accessPath", "realPath"))
    # Deliberately match substrings across package identifiers and paths. False positives stop the release for inspection;
    # this fail-closed bias avoids missing versioned, renamed, or unusually located codec packages.
    if any(pattern in value for pattern in patterns for value in searchable):
      description = " ".join(value for value in (name, version) if value)
      matches.append(f"Syft package: {description or '<unnamed>'}")
  return sorted(set(matches))


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


def _native_file(path: Path) -> bool:
  try:
    with path.open("rb") as stream:
      magic = stream.read(4)
  except OSError:
    return False
  return any(magic.startswith(candidate) for candidate in NATIVE_MAGICS)


def _library_dependency_aliases(path: Path) -> set[str]:
  name = path.name.lower()
  aliases = {name}
  if ".so." in name:
    prefix, version = name.split(".so.", 1)
    aliases.update({f"{prefix}.so", f"{prefix}.so.{version.split('.', 1)[0]}"})
  return aliases


def _has_replaceable_libraw(candidates: Iterable[Path]) -> bool:
  candidates = list(candidates)
  libraries = [
    path
    for path in candidates
    if _native_file(path)
    and any(marker in path.name.lower() for marker in ("libraw", "raw_r"))
    and (path.suffix.lower() in {".dll", ".dylib", ".so"} or ".so." in path.name.lower())
  ]
  bindings = [path for path in candidates if "_rawpy" in path.name.lower() and _native_file(path)]
  for binding in bindings:
    aliases = sorted({alias for library in libraries for alias in _library_dependency_aliases(library)})
    if aliases and _binary_contains(binding, aliases, casefold=True):
      return True
  return False


def _native_component_evidence(
  root: Path,
  policy: dict[str, Any],
  candidates: Iterable[Path],
  reported_versions: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
  candidates = list(candidates)
  component_policies = [
    (codec_name, component)
    for codec_name, codec_policy in policy.get("codecs", {}).items()
    for component in codec_policy.get("native_components", [])
  ]
  binary_markers = {
    str(marker) for _codec_name, component in component_policies for marker in component.get("binary_markers", [])
  }
  binary_matches = {path: set(_binary_contains(path, binary_markers)) for path in candidates}
  components = []
  for codec_name, component in component_policies:
    path_markers = [str(marker).lower() for marker in component.get("path_markers", [])]
    component_binary_markers = {str(marker) for marker in component.get("binary_markers", [])}
    paths = []
    for path in candidates:
      relative = path.relative_to(root).as_posix()
      path_match = any(marker in relative.lower() for marker in path_markers)
      binary_match = bool(component_binary_markers & binary_matches[path])
      if path_match or binary_match:
        paths.append(relative)
    if paths:
      name = str(component["name"])
      reported = reported_versions.get(name, {})
      components.append(
        {
          "codec": codec_name,
          "name": name,
          "version": reported.get("version", ""),
          "version_source": reported.get("source", ""),
          "paths": sorted(set(paths)),
        }
      )
  return sorted(components, key=lambda item: (item["name"], item["version"]))


def augment_cyclonedx(
  input_path: Path,
  output_path: Path,
  root: Path,
  legal_dir: Path,
  policy_path: Path | None,
) -> dict[str, Any]:
  data = json.loads(input_path.read_text(encoding="utf-8"))
  if data.get("bomFormat") != "CycloneDX" or not isinstance(data.get("components"), list):
    raise ComplianceError(f"unsupported CycloneDX document in {input_path}")
  root = root.resolve()
  legal_dir = legal_dir.resolve()
  policy = load_policy(policy_path)
  report, legal_errors = _validate_legal_pack(legal_dir, policy_path or _find_policy())
  if legal_errors:
    raise ComplianceError("\n".join(legal_errors))
  candidates = _native_candidates(root, legal_dir)
  evidence = _native_component_evidence(root, policy, candidates, _reported_native_versions(report))
  incomplete = [component for component in evidence if not component["version"] or not component["version_source"]]
  if incomplete:
    raise ComplianceError(
      "missing collected native versions: " + ", ".join(sorted(component["name"] for component in incomplete))
    )
  existing = {
    (str(item.get("name", "")), str(item.get("version", ""))) for item in data["components"] if isinstance(item, dict)
  }
  for component in evidence:
    key = (component["name"], component["version"])
    if key in existing:
      continue
    data["components"].append(
      {
        "type": "library",
        "bom-ref": f"pkg:generic/{component['name']}@{component['version']}",
        "name": component["name"],
        "version": component["version"],
        "purl": f"pkg:generic/{component['name']}@{component['version']}",
        "properties": [
          {"name": "rclip:codec", "value": component["codec"]},
          {"name": "rclip:version-evidence", "value": component["version_source"]},
          *[{"name": "rclip:evidence", "value": path} for path in component["paths"]],
        ],
      }
    )
  data["components"].sort(
    key=lambda item: (str(item.get("name", "")), str(item.get("version", ""))) if isinstance(item, dict) else ("", "")
  )
  _json_dump(output_path, data)
  return data


def verify_bundle(
  root: Path,
  legal_dir: Path,
  policy_path: Path | None,
  syft_json: Path | None,
  cyclonedx_json: Path | None = None,
) -> dict[str, Any]:
  root = root.resolve()
  legal_dir = legal_dir.resolve()
  resolved_policy_path = policy_path or _find_policy()
  policy = load_policy(resolved_policy_path)
  locked_versions = _locked_runtime_versions(_find_lock(resolved_policy_path))
  errors: list[str] = []
  codec_policies = policy.get("codecs", {})
  detections: dict[str, list[str]] = {codec_name: [] for codec_name in codec_policies}

  for filename in ("THIRD_PARTY_NOTICES.txt", "compliance-report.json", "policy.toml"):
    if not (legal_dir / filename).is_file():
      errors.append(f"legal pack is missing {filename}")
  report, legal_errors = _validate_legal_pack(legal_dir, resolved_policy_path)
  errors.extend(legal_errors)
  reported_packages = _reported_python_packages(legal_dir)
  try:
    _validate_python_packages(reported_packages, policy, locked_versions)
  except ComplianceError as error:
    errors.append(str(error))
  for codec_name, codec_policy in codec_policies.items():
    required_notice = codec_policy.get("required_notice")
    if required_notice and not _notice_present(
      legal_dir,
      required_notice,
      codec_policy.get("required_text"),
      codec_policy.get("required_sha256"),
    ):
      errors.append(f"legal pack is missing {required_notice} for {codec_name}")
  notices_index = legal_dir / "THIRD_PARTY_NOTICES.txt"
  dng_attribution = _required_codec_text(policy, "dng")
  if notices_index.is_file() and dng_attribution not in notices_index.read_text(encoding="utf-8", errors="replace"):
    errors.append("third-party notice index is missing the required DNG attribution")

  prohibited_files = {name.lower() for name in policy.get("prohibited_runtime_files", [])}
  for path in root.rglob("*"):
    if not path.is_file() or _is_inside(path, legal_dir):
      continue
    stem = path.stem.lower()
    if path.name.lower() in prohibited_files or stem in prohibited_files:
      errors.append(f"build-only executable is present: {path.relative_to(root)}")

  native_candidates = _native_candidates(root, legal_dir)
  native_components = _native_component_evidence(
    root,
    policy,
    native_candidates,
    _reported_native_versions(report),
  )
  for component in native_components:
    detections[component["codec"]].extend(component["paths"])
  for codec_name, codec_policy in codec_policies.items():
    filename_patterns = [value.lower() for value in codec_policy.get("forbidden_filename_patterns", [])]
    binary_markers = codec_policy.get("forbidden_binary_markers", [])
    if not filename_patterns and not binary_markers:
      continue
    for path in native_candidates:
      relative = path.relative_to(root).as_posix()
      if any(pattern in relative.lower() for pattern in filename_patterns):
        detections[codec_name].append(relative)
      if _binary_contains(path, binary_markers):
        detections[codec_name].append(f"{relative} (binary marker)")

  syft_packages: list[dict[str, str]] = []
  if syft_json is not None:
    syft_data = json.loads(syft_json.read_text(encoding="utf-8"))
    syft_packages = _syft_python_packages(syft_data)
    syft_inventory = {(package["name"], package["version"]) for package in syft_packages}
    reported_inventory = {(package["name"], package["version"]) for package in reported_packages}
    if not syft_inventory:
      errors.append("Syft inventory contains no Python distributions")
    missing_from_syft = sorted(reported_inventory - syft_inventory)
    missing_from_report = sorted(syft_inventory - reported_inventory)
    if missing_from_syft:
      errors.append(
        "Python distributions missing from Syft inventory: "
        + ", ".join(f"{name} {version or '<missing>'}" for name, version in missing_from_syft)
      )
    if missing_from_report:
      errors.append(
        "Python distributions missing from legal report: "
        + ", ".join(f"{name} {version or '<missing>'}" for name, version in missing_from_report)
      )
    for codec_name, codec_policy in codec_policies.items():
      package_patterns = codec_policy.get("forbidden_package_patterns", [])
      if package_patterns:
        detections[codec_name].extend(_syft_native_matches(syft_data, package_patterns))
    try:
      _validate_python_packages(syft_packages, policy, locked_versions)
    except ComplianceError as error:
      errors.append(str(error))

  for codec, codec_policy in codec_policies.items():
    if detections[codec]:
      if not codec_policy.get("allowed", False):
        errors.append(
          f"forbidden {codec.upper()} implementation detected: " + ", ".join(sorted(set(detections[codec])))
        )

  if detections.get("dng") and policy.get("copyleft", {}).get("libraw", {}).get("must_be_shared", False):
    if not _has_replaceable_libraw(native_candidates):
      errors.append("rawpy/LibRaw detected without a separately replaceable LibRaw shared library")

  evidenced_components = {(component["codec"], component["name"]): component for component in native_components}
  for codec, codec_paths in detections.items():
    codec_policy = codec_policies[codec]
    if not codec_paths or not codec_policy.get("allowed", False):
      continue
    for expected in codec_policy.get("native_components", []):
      key = (codec, str(expected["name"]))
      actual = evidenced_components.get(key)
      if actual is None:
        errors.append(f"missing native component evidence for {expected['name']} {expected['version']}")
      elif not actual["version"] or not actual["version_source"]:
        errors.append(f"missing collected native version for {expected['name']}")
      elif actual["version"] != str(expected["version"]):
        errors.append(
          f"disallowed native version: {expected['name']} {actual['version']} (allowed: {expected['version']})"
        )
  if cyclonedx_json is not None:
    try:
      cyclonedx = json.loads(cyclonedx_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
      errors.append(f"invalid CycloneDX inventory: {error}")
      cyclonedx = {}
    cyclonedx_components = cyclonedx.get("components")
    if not isinstance(cyclonedx_components, list):
      errors.append("CycloneDX inventory has no component list")
      cyclonedx_components = []
    inventoried = {
      (str(component.get("name", "")), str(component.get("version", "")))
      for component in cyclonedx_components
      if isinstance(component, dict)
    }
    missing_components = [
      component for component in native_components if (component["name"], component["version"]) not in inventoried
    ]
    if missing_components:
      errors.append(
        "native components missing from CycloneDX inventory: "
        + ", ".join(f"{item['name']} {item['version']}" for item in missing_components)
      )

  result = {
    "schema_version": 1,
    "detections": {key: sorted(set(value)) for key, value in detections.items()},
    "native_components": native_components,
    "syft_python_packages": syft_packages,
    "errors": sorted(set(errors)),
  }
  if errors:
    raise ComplianceError("\n".join(sorted(set(errors))))
  return result


def _deterministic_tar(source: Path, output: Path, archive_root: str, excluded: Iterable[Path] = ()) -> None:
  excluded = tuple(excluded)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("wb") as raw_stream:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as gzip_stream:
      with tarfile.open(fileobj=gzip_stream, mode="w") as archive:
        for path in sorted(source.rglob("*")):
          relative = path.relative_to(source)
          if ".git" in relative.parts or any(relative == prefix or prefix in relative.parents for prefix in excluded):
            continue
          info = archive.gettarinfo(str(path), arcname=str(Path(archive_root) / relative))
          info.uid = 0
          info.gid = 0
          info.uname = ""
          info.gname = ""
          info.mtime = 0
          if path.is_dir():
            info.mode = 0o755
          elif path.is_symlink():
            info.mode = 0o777
          else:
            info.mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
          if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
              archive.addfile(info, stream)
          else:
            archive.addfile(info)


def build_corresponding_source(manifest_path: Path, output: Path) -> None:
  with manifest_path.open("rb") as stream:
    manifest = tomllib.load(stream)
  if manifest.get("schema_version") != 1:
    raise ComplianceError(f"unsupported source manifest schema in {manifest_path}")
  rawpy = manifest.get("rawpy")
  if not isinstance(rawpy, dict):
    raise ComplianceError(f"source manifest in {manifest_path} is missing rawpy")
  description = f"rawpy source manifest in {manifest_path}"
  repository = _required_string(rawpy, "repository", description)
  revision = _required_string(rawpy, "revision", description)
  commit = _required_string(rawpy, "commit", description)
  version = _required_string(rawpy, "version", description)
  libraw_version = _required_string(rawpy, "libraw_version", description)
  submodules = rawpy.get("submodules", {})
  if not isinstance(submodules, dict):
    raise ComplianceError(f"{description} has invalid submodules")
  expected_submodules = {}
  for relative, expected_revision in submodules.items():
    if not isinstance(relative, str) or not relative or not isinstance(expected_revision, str) or not expected_revision:
      raise ComplianceError(f"{description} has invalid submodule {relative}")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
      raise ComplianceError(f"{description} has unsafe submodule {relative}")
    expected_submodules[relative] = expected_revision
  excluded_paths = []
  for key in ("excluded_submodules", "excluded_paths"):
    values = rawpy.get(key, [])
    if not isinstance(values, list) or any(not isinstance(relative, str) or not relative for relative in values):
      raise ComplianceError(f"{description} has invalid {key}")
    paths = [Path(relative) for relative in values]
    if any(path.is_absolute() or ".." in path.parts for path in paths):
      raise ComplianceError(f"{description} has unsafe {key}")
    excluded_paths.extend(paths)
  with tempfile.TemporaryDirectory(prefix="rclip-source-") as temporary:
    checkout = Path(temporary) / "rawpy"
    subprocess.run(
      [
        "git",
        "clone",
        "--branch",
        revision,
        "--depth",
        "1",
        repository,
        str(checkout),
      ],
      check=True,
    )
    rawpy_commit = subprocess.run(
      ["git", "-C", str(checkout), "rev-parse", "HEAD"],
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
    if rawpy_commit != commit:
      raise ComplianceError(f"rawpy is {rawpy_commit}, expected {commit}")
    subprocess.run(
      ["git", "-C", str(checkout), "submodule", "update", "--init", "--depth", "1", "--", *expected_submodules],
      check=True,
    )
    for relative, expected_revision in expected_submodules.items():
      actual = subprocess.run(
        ["git", "-C", str(checkout / relative), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
      ).stdout.strip()
      if actual != expected_revision:
        raise ComplianceError(f"{relative} is {actual}, expected {expected_revision}")
    libraw_header = checkout / "external" / "LibRaw" / "libraw" / "libraw_version.h"
    if not libraw_header.is_file():
      raise ComplianceError("pinned LibRaw checkout does not contain libraw_version.h")
    version_text = libraw_header.read_text(encoding="utf-8", errors="replace")
    version_parts = []
    for macro in ("LIBRAW_MAJOR_VERSION", "LIBRAW_MINOR_VERSION", "LIBRAW_PATCH_VERSION"):
      match = re.search(rf"^#define\s+{macro}\s+(\d+)\s*$", version_text, re.MULTILINE)
      if match is None:
        raise ComplianceError(f"could not read {macro} from pinned LibRaw checkout")
      version_parts.append(match.group(1))
    actual_libraw_version = ".".join(version_parts)
    if actual_libraw_version != libraw_version:
      raise ComplianceError(f"LibRaw is {actual_libraw_version}, expected {libraw_version}")
    _deterministic_tar(
      checkout,
      output,
      f"rawpy-{version}-corresponding-source",
      excluded=excluded_paths,
    )


def _path(value: str) -> Path:
  return Path(value)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)

  collect_parser = subparsers.add_parser("collect", help="collect third-party legal materials")
  collect_parser.add_argument("--root", type=_path, required=True)
  collect_parser.add_argument("--output", type=_path, required=True)
  collect_parser.add_argument("--policy", type=_path)
  collect_parser.add_argument("--common-notices", type=_path)
  collect_parser.add_argument(
    "--include-python-runtime",
    action="store_true",
    help="include the current CPython runtime and its licence",
  )

  verify_parser = subparsers.add_parser("verify", help="inspect an assembled runtime bundle")
  verify_parser.add_argument("--root", type=_path, required=True)
  verify_parser.add_argument("--legal-dir", type=_path, required=True)
  verify_parser.add_argument("--policy", type=_path)
  verify_parser.add_argument("--syft-json", type=_path)
  verify_parser.add_argument("--cyclonedx-json", type=_path)
  verify_parser.add_argument("--output", type=_path)

  source_parser = subparsers.add_parser("source-bundle", help="build LibRaw corresponding source archive")
  source_parser.add_argument("--manifest", type=_path, default=Path("compliance/sources.toml"))
  source_parser.add_argument("--output", type=_path, required=True)

  cyclonedx_parser = subparsers.add_parser("augment-cyclonedx", help="add declared native codec components to an SBOM")
  cyclonedx_parser.add_argument("--input", type=_path, required=True)
  cyclonedx_parser.add_argument("--output", type=_path, required=True)
  cyclonedx_parser.add_argument("--root", type=_path, required=True)
  cyclonedx_parser.add_argument("--legal-dir", type=_path, required=True)
  cyclonedx_parser.add_argument("--policy", type=_path)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  display_report: dict[str, object]
  try:
    if args.command == "collect":
      report = collect_legal_materials(
        args.root,
        args.output,
        args.policy,
        args.common_notices,
        args.include_python_runtime,
      )
      display_report = {"components": len(report["components"]), "output": args.output.as_posix()}
    elif args.command == "verify":
      report = verify_bundle(args.root, args.legal_dir, args.policy, args.syft_json, args.cyclonedx_json)
      if args.output:
        _json_dump(args.output, report)
      display_report = {
        "detections": {name: len(paths) for name, paths in report["detections"].items()},
        "output": args.output.as_posix() if args.output else None,
      }
    elif args.command == "source-bundle":
      build_corresponding_source(args.manifest, args.output)
      display_report = {"source_bundle": args.output.as_posix()}
    elif args.command == "augment-cyclonedx":
      report = augment_cyclonedx(args.input, args.output, args.root, args.legal_dir, args.policy)
      display_report = {"components": len(report["components"]), "output": args.output.as_posix()}
    else:  # pragma: no cover - argparse enforces this
      raise AssertionError(args.command)
  except (
    ComplianceError,
    json.JSONDecodeError,
    OSError,
    subprocess.CalledProcessError,
    tomllib.TOMLDecodeError,
  ) as error:
    print(f"compliance error: {error}", file=sys.stderr)
    return 1
  print(json.dumps(display_report, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
