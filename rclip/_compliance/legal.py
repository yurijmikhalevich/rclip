from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import Any

from rclip._compliance.common import ComplianceError
from rclip._compliance.common import _confined_file
from rclip._compliance.common import _is_inside
from rclip._compliance.common import _json_dump
from rclip._compliance.common import _sha256
from rclip._compliance.licenses import _declared_license_expression
from rclip._compliance.licenses import _legal_files
from rclip._compliance.licenses import _metadata_records
from rclip._compliance.licenses import _validate_python_licenses
from rclip._compliance.licenses import _validate_python_packages
from rclip._compliance.native import _native_component_versions
from rclip._compliance.policy import _find_lock
from rclip._compliance.policy import _find_policy
from rclip._compliance.policy import _locked_runtime_versions
from rclip._compliance.policy import _required_codec_text
from rclip._compliance.policy import load_policy


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
  for prefix in (Path(sys.base_prefix), root):
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
  textual_image_component = next((component for component in components if component["name"] == "textual-image"), None)
  if rawpy_component is not None and rclip_component is not None:
    source_filename = f"rawpy-{rawpy_component['version']}-corresponding-source.tar.gz"
    index_lines.extend(
      [
        "LibRaw corresponding source",
        "---------------------------",
        f"https://github.com/yurijmikhalevich/rclip/releases/download/v{rclip_component['version']}/{source_filename}",
        "",
      ]
    )
  if textual_image_component is not None and rclip_component is not None:
    version = textual_image_component["version"]
    release = f"https://github.com/yurijmikhalevich/rclip/releases/download/v{rclip_component['version']}"
    application = f"https://github.com/yurijmikhalevich/rclip/archive/refs/tags/v{rclip_component['version']}.tar.gz"
    index_lines.extend(
      [
        "textual-image (LGPL-3.0-or-later)",
        "----------------------------------",
        f"This product uses textual-image {version} under LGPL-3.0-or-later.",
        f"{release}/textual_image-{version}.tar.gz",
        f"Corresponding application source: {application}",
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
