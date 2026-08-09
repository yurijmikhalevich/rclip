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


DNG_ATTRIBUTION = "This product includes DNG technology under license by Adobe."
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
  b"\xce\xfa\xed\xfe",
  b"\xcf\xfa\xed\xfe",
  b"\xfe\xed\xfa\xce",
  b"\xfe\xed\xfa\xcf",
)
PYTHON_NAME_PATTERN = re.compile(r"[-_.]+")


class ComplianceError(RuntimeError):
  """Raised when a bundle does not satisfy the reviewed policy."""


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


def _find_policy(root: Path | None = None) -> Path:
  source_policy = Path(__file__).resolve().parent.parent / "compliance" / "policy.toml"
  candidates = [Path.cwd() / "compliance" / "policy.toml", source_policy]
  if root is not None:
    candidates.extend(
      path
      for path in sorted(root.rglob("policy.toml"))
      if any(part.endswith(".dist-info") for part in path.parts) and "licenses" in path.parts
    )
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  raise ComplianceError("could not find compliance/policy.toml; pass --policy explicitly")


def load_policy(path: Path | None, root: Path | None = None) -> dict[str, Any]:
  policy_path = path or _find_policy(root)
  with policy_path.open("rb") as stream:
    policy = tomllib.load(stream)
  if policy.get("schema_version") != 1:
    raise ComplianceError(f"unsupported policy schema in {policy_path}")
  return policy


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
    }
    records[(name, version)] = record
  return [records[key] for key in sorted(records)]


def _review_python_packages(records: Iterable[dict[str, Any]], policy: dict[str, Any]) -> None:
  records = list(records)
  reviewed = {normalize_python_name(name) for name in policy.get("reviewed_python_packages", [])}
  prohibited = {normalize_python_name(name) for name in policy.get("prohibited_python_packages", [])}
  installed = {record["name"] for record in records}
  rejected = sorted(installed & prohibited)
  unknown = sorted(installed - reviewed)
  errors = []
  if rejected:
    errors.append(f"prohibited Python distributions: {', '.join(rejected)}")
  if unknown:
    errors.append(f"unreviewed Python distributions: {', '.join(unknown)}")
  reviewed_versions = {
    normalize_python_name(name): {str(version) for version in versions}
    for name, versions in policy.get("reviewed_python_versions", {}).items()
  }
  unversioned = {normalize_python_name(name) for name in policy.get("unversioned_python_packages", [])}
  conflicting_version_policy = sorted(reviewed_versions.keys() & unversioned)
  missing_version_policy = sorted((installed & reviewed) - reviewed_versions.keys() - unversioned)
  if conflicting_version_policy:
    errors.append(
      "Python distributions have both reviewed and unversioned policies: " + ", ".join(conflicting_version_policy)
    )
  if missing_version_policy:
    errors.append("Python distributions without a version policy: " + ", ".join(missing_version_policy))
  version_drift = []
  for record in records:
    expected = reviewed_versions.get(record["name"])
    actual = record.get("version")
    if expected is not None and str(actual) not in expected:
      version_drift.append(f"{record['name']} {actual or '<missing>'} (reviewed: {', '.join(sorted(expected))})")
  if version_drift:
    errors.append(f"unreviewed Python versions: {', '.join(sorted(version_drift))}")
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
  override_dir = policy_dir / "license-overrides" / record["name"]
  if override_dir.is_dir():
    result.update(path for path in override_dir.rglob("*") if _confined_file(path, override_dir))
  return sorted(result)


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(source, destination)
  return {"path": destination.as_posix(), "sha256": _sha256(destination)}


def _system_legal_materials(root: Path, output: Path) -> list[dict[str, Any]]:
  """Collect Debian-style copyright files included in self-contained Linux bundles."""
  components = []
  seen: set[str] = set()
  for source in sorted(root.rglob("share/doc/*/copyright")):
    doc_dir = source.parent
    if not _confined_file(source, root) or _is_inside(source, output):
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


def collect_legal_materials(
  root: Path,
  output: Path,
  policy_path: Path | None,
  common_notices: Path | None,
) -> dict[str, Any]:
  root = root.resolve()
  output = output.resolve()
  if root == output:
    raise ComplianceError("legal output directory cannot be the scanned root")
  records = _metadata_records(root, output)
  if not records:
    raise ComplianceError(f"no Python distribution metadata found under {root}")
  resolved_policy_path = policy_path or _find_policy(root)
  policy = load_policy(resolved_policy_path, root)
  _review_python_packages(records, policy)

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
        "license_files": sorted(item["path"].replace(output.as_posix() + "/", "") for item in copied),
        "license_file_hashes": {
          item["path"].replace(output.as_posix() + "/", ""): item["sha256"]
          for item in sorted(copied, key=lambda x: x["path"])
        },
      }
    )

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

  components.extend(_system_legal_materials(root, output))

  notices_source = common_notices or resolved_policy_path.parent / "notices"
  if not notices_source.is_dir():
    notice_matches = sorted(root.rglob("AOM-PATENT-LICENSE-1.0.txt"))
    if notice_matches:
      notices_source = notice_matches[0].parent
  if not notices_source.is_dir():
    raise ComplianceError(f"common notice directory not found: {notices_source}")
  notice_names = []
  notice_hashes = {}
  for source in sorted(notices_source.glob("*.txt")):
    copied = _copy_file(source, output / "notices" / source.name)
    notice_names.append(source.name)
    notice_hashes[source.name] = copied["sha256"]
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
    DNG_ATTRIBUTION,
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
    "codec_notice_hashes": notice_hashes,
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


def _binary_contains(path: Path, markers: Iterable[str]) -> list[str]:
  marker_bytes = [(marker, marker.encode("ascii")) for marker in markers]
  overlap = max((len(value) for _, value in marker_bytes), default=1) - 1
  found: set[str] = set()
  previous = b""
  with path.open("rb") as stream:
    while current := stream.read(4 * 1024 * 1024):
      chunk = previous + current
      for marker, value in marker_bytes:
        if marker not in found and value in chunk:
          found.add(marker)
      if len(found) == len(marker_bytes):
        break
      previous = chunk[-overlap:] if overlap else b""
  return sorted(found)


def _binary_contains_casefold(path: Path, markers: Iterable[str]) -> list[str]:
  marker_bytes = [(marker, marker.encode("ascii").lower()) for marker in markers]
  overlap = max((len(value) for _, value in marker_bytes), default=1) - 1
  found: set[str] = set()
  previous = b""
  with path.open("rb") as stream:
    while current := stream.read(4 * 1024 * 1024):
      chunk = (previous + current).lower()
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
    # Deliberately match substrings across package identifiers and paths. False positives stop the release for review;
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
  for component in components:
    if not isinstance(component, dict):
      errors.append("legal pack contains an invalid component record")
      continue
    description = f"{component.get('name', '<unnamed>')} {component.get('version', '')}".rstrip()
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
    errors.append("legal pack contains a policy that differs from the reviewed policy")
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


def _has_replaceable_libraw(root: Path, legal_dir: Path) -> bool:
  candidates = _native_candidates(root, legal_dir)
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
    if aliases and _binary_contains_casefold(binding, aliases):
      return True
  return False


def _native_component_evidence(root: Path, legal_dir: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
  candidates = _native_candidates(root, legal_dir)
  components = []
  for codec_name, codec_policy in policy.get("codecs", {}).items():
    for component in codec_policy.get("native_components", []):
      path_markers = component.get("path_markers", [])
      binary_markers = component.get("binary_markers", [])
      paths = []
      for path in candidates:
        relative = path.relative_to(root).as_posix()
        path_match = any(marker.lower() in relative.lower() for marker in path_markers)
        binary_match = bool(binary_markers) and bool(_binary_contains(path, binary_markers))
        if path_match or binary_match:
          paths.append(relative)
      if paths:
        components.append(
          {
            "codec": codec_name,
            "name": str(component["name"]),
            "version": str(component["version"]),
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
  policy = load_policy(policy_path, root)
  evidence = _native_component_evidence(root.resolve(), legal_dir.resolve(), policy)
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
  resolved_policy_path = policy_path or _find_policy(root)
  policy = load_policy(resolved_policy_path, root)
  errors: list[str] = []
  detections: dict[str, list[str]] = {"av1": [], "dng": [], "hevc": [], "webp": []}

  for filename in ("THIRD_PARTY_NOTICES.txt", "compliance-report.json", "policy.toml"):
    if not (legal_dir / filename).is_file():
      errors.append(f"legal pack is missing {filename}")
  _report, legal_errors = _validate_legal_pack(legal_dir, resolved_policy_path)
  errors.extend(legal_errors)
  for codec_name, codec_policy in policy.get("codecs", {}).items():
    required_notice = codec_policy.get("required_notice")
    if required_notice and not _notice_present(
      legal_dir,
      required_notice,
      codec_policy.get("required_text"),
      codec_policy.get("required_sha256"),
    ):
      errors.append(f"legal pack is missing {required_notice} for {codec_name}")
  notices_index = legal_dir / "THIRD_PARTY_NOTICES.txt"
  if notices_index.is_file() and DNG_ATTRIBUTION not in notices_index.read_text(encoding="utf-8", errors="replace"):
    errors.append("third-party notice index is missing the required DNG attribution")

  prohibited_files = {name.lower() for name in policy.get("prohibited_runtime_files", [])}
  for path in root.rglob("*"):
    if not path.is_file() or _is_inside(path, legal_dir):
      continue
    stem = path.stem.lower()
    if path.name.lower() in prohibited_files or stem in prohibited_files:
      errors.append(f"build-only executable is present: {path.relative_to(root)}")

  hevc_policy = policy["codecs"]["hevc"]
  filename_patterns = [value.lower() for value in hevc_policy.get("forbidden_filename_patterns", [])]
  binary_markers = hevc_policy.get("forbidden_binary_markers", [])
  av1_names = ("libaom", "libavif", "_avif")
  dng_names = ("libraw", "raw_r", "_rawpy")
  webp_names = ("libwebp", "_webp")
  for path in _native_candidates(root, legal_dir):
    relative = path.relative_to(root).as_posix()
    lower = relative.lower()
    if any(pattern in lower for pattern in filename_patterns):
      detections["hevc"].append(relative)
    if any(pattern in lower for pattern in av1_names):
      detections["av1"].append(relative)
    if any(pattern in lower for pattern in dng_names):
      detections["dng"].append(relative)
    if any(pattern in lower for pattern in webp_names):
      detections["webp"].append(relative)
    found_markers = _binary_contains(path, [*binary_markers, "aom_codec", "avifDecoder", "WebPDecode"])
    if any(marker in found_markers for marker in binary_markers):
      detections["hevc"].append(f"{relative} (binary marker)")
    if any(marker in found_markers for marker in ("aom_codec", "avifDecoder")):
      detections["av1"].append(f"{relative} (binary marker)")
    if "WebPDecode" in found_markers:
      detections["webp"].append(f"{relative} (binary marker)")

  syft_packages: list[dict[str, str]] = []
  if syft_json is not None:
    syft_data = json.loads(syft_json.read_text(encoding="utf-8"))
    syft_packages = _syft_python_packages(syft_data)
    reported_packages = _reported_python_packages(legal_dir)
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
    detections["hevc"].extend(_syft_native_matches(syft_data, hevc_policy.get("forbidden_package_patterns", [])))
    try:
      _review_python_packages(syft_packages, policy)
    except ComplianceError as error:
      errors.append(str(error))

  for codec in ("av1", "dng", "webp", "hevc"):
    codec_policy = policy["codecs"][codec]
    if detections[codec]:
      if not codec_policy.get("allowed", False):
        errors.append(
          f"forbidden {codec.upper()} implementation detected: " + ", ".join(sorted(set(detections[codec])))
        )
        continue
      notice = codec_policy.get("required_notice")
      if not notice:
        continue
      required_text = codec_policy.get("required_text")
      required_sha256 = codec_policy.get("required_sha256")
      if not _notice_present(legal_dir, notice, required_text, required_sha256):
        errors.append(f"{codec.upper()} detected without required {notice}")

  if detections["dng"] and policy.get("copyleft", {}).get("libraw", {}).get("must_be_shared", False):
    if not _has_replaceable_libraw(root, legal_dir):
      errors.append("rawpy/LibRaw detected without a separately replaceable LibRaw shared library")

  native_components = _native_component_evidence(root, legal_dir, policy)
  evidenced_components = {
    (component["codec"], component["name"], component["version"]) for component in native_components
  }
  for codec, codec_paths in detections.items():
    if not codec_paths or not policy["codecs"][codec].get("allowed", False):
      continue
    for expected in policy["codecs"][codec].get("native_components", []):
      key = (codec, str(expected["name"]), str(expected["version"]))
      if key not in evidenced_components:
        errors.append(f"missing native component evidence for {expected['name']} {expected['version']}")
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


def normalize_scancode(input_path: Path, output_path: Path) -> dict[str, Any]:
  data = json.loads(input_path.read_text(encoding="utf-8"))
  packages = []
  for package in data.get("packages", []):
    packages.append(
      {
        "name": normalize_python_name(str(package.get("name", ""))),
        "version": str(package.get("version", "")),
        "declared_license_expression": package.get("declared_license_expression"),
      }
    )
  file_licenses = []
  for file_record in data.get("files", []):
    expressions = sorted(
      {
        str(detection.get("license_expression"))
        for detection in file_record.get("license_detections", [])
        if detection.get("license_expression")
      }
    )
    if expressions:
      file_licenses.append({"path": file_record.get("path", ""), "license_expressions": expressions})
  normalized = {
    "schema_version": 1,
    "packages": sorted(packages, key=lambda item: (item["name"], item["version"])),
    "file_licenses": sorted(file_licenses, key=lambda item: item["path"]),
  }
  _json_dump(output_path, normalized)
  return normalized


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
  rawpy = manifest["rawpy"]
  with tempfile.TemporaryDirectory(prefix="rclip-source-") as temporary:
    checkout = Path(temporary) / "rawpy"
    subprocess.run(
      [
        "git",
        "clone",
        "--branch",
        rawpy["revision"],
        "--depth",
        "1",
        rawpy["repository"],
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
    if rawpy_commit != rawpy["commit"]:
      raise ComplianceError(f"rawpy is {rawpy_commit}, expected {rawpy['commit']}")
    expected_submodules = rawpy.get("submodules", {})
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
    if actual_libraw_version != rawpy["libraw_version"]:
      raise ComplianceError(f"LibRaw is {actual_libraw_version}, expected {rawpy['libraw_version']}")
    excluded_submodules = [Path(relative) for relative in rawpy.get("excluded_submodules", [])]
    _deterministic_tar(
      checkout,
      output,
      f"rawpy-{rawpy['version']}-corresponding-source",
      excluded=excluded_submodules,
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

  verify_parser = subparsers.add_parser("verify", help="inspect an assembled runtime bundle")
  verify_parser.add_argument("--root", type=_path, required=True)
  verify_parser.add_argument("--legal-dir", type=_path, required=True)
  verify_parser.add_argument("--policy", type=_path)
  verify_parser.add_argument("--syft-json", type=_path)
  verify_parser.add_argument("--cyclonedx-json", type=_path)
  verify_parser.add_argument("--output", type=_path)

  scancode_parser = subparsers.add_parser("normalize-scancode", help="make ScanCode output reviewable")
  scancode_parser.add_argument("--input", type=_path, required=True)
  scancode_parser.add_argument("--output", type=_path, required=True)

  source_parser = subparsers.add_parser("source-bundle", help="build LibRaw corresponding source archive")
  source_parser.add_argument("--manifest", type=_path, default=Path("compliance/sources.toml"))
  source_parser.add_argument("--output", type=_path, required=True)

  cyclonedx_parser = subparsers.add_parser("augment-cyclonedx", help="add reviewed native codec components to an SBOM")
  cyclonedx_parser.add_argument("--input", type=_path, required=True)
  cyclonedx_parser.add_argument("--output", type=_path, required=True)
  cyclonedx_parser.add_argument("--root", type=_path, required=True)
  cyclonedx_parser.add_argument("--legal-dir", type=_path, required=True)
  cyclonedx_parser.add_argument("--policy", type=_path)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  display_report: object
  try:
    if args.command == "collect":
      report = collect_legal_materials(args.root, args.output, args.policy, args.common_notices)
      display_report = {"components": len(report["components"]), "output": args.output.as_posix()}
    elif args.command == "verify":
      report = verify_bundle(args.root, args.legal_dir, args.policy, args.syft_json, args.cyclonedx_json)
      if args.output:
        _json_dump(args.output, report)
      display_report = {
        "detections": {name: len(paths) for name, paths in report["detections"].items()},
        "output": args.output.as_posix() if args.output else None,
      }
    elif args.command == "normalize-scancode":
      report = normalize_scancode(args.input, args.output)
      display_report = {
        "file_licenses": len(report["file_licenses"]),
        "output": args.output.as_posix(),
        "packages": len(report["packages"]),
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
    KeyError,
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
