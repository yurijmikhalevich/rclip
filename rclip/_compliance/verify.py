from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rclip._compliance.common import ComplianceError
from rclip._compliance.common import _confined_file
from rclip._compliance.common import _is_inside
from rclip._compliance.common import _sha256
from rclip._compliance.licenses import _validate_python_packages
from rclip._compliance.native import _binary_contains
from rclip._compliance.native import _has_replaceable_libraw
from rclip._compliance.native import _native_candidates
from rclip._compliance.native import _native_component_evidence
from rclip._compliance.policy import _find_lock
from rclip._compliance.policy import _find_policy
from rclip._compliance.policy import _locked_runtime_versions
from rclip._compliance.policy import _required_codec_text
from rclip._compliance.policy import load_policy
from rclip._compliance.report import _reported_native_versions
from rclip._compliance.report import _reported_python_packages
from rclip._compliance.report import _validate_legal_pack
from rclip._compliance.sbom import _syft_native_matches
from rclip._compliance.sbom import _syft_python_packages


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
    if detections[codec] and not codec_policy.get("allowed", False):
      errors.append(f"forbidden {codec.upper()} implementation detected: " + ", ".join(sorted(set(detections[codec]))))

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
