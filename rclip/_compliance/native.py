from __future__ import annotations

from collections.abc import Iterable
import importlib
from pathlib import Path
import re
from typing import Any

from rclip._compliance.common import ComplianceError
from rclip._compliance.common import _is_inside


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
FORBIDDEN_RAWPY_FEATURES = ("DEMOSAIC_PACK_GPL2", "DEMOSAIC_PACK_GPL3")


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
