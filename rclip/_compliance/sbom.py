from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from rclip._compliance.common import ComplianceError
from rclip._compliance.common import _json_dump
from rclip._compliance.common import normalize_python_name
from rclip._compliance.native import _native_candidates
from rclip._compliance.native import _native_component_evidence
from rclip._compliance.policy import _find_policy
from rclip._compliance.policy import load_policy
from rclip._compliance.report import _reported_native_versions
from rclip._compliance.report import _validate_legal_pack


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
