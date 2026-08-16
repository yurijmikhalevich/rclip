from __future__ import annotations

from collections.abc import Iterable
import gzip
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile
import tomllib

from rclip.compliance.common import ComplianceError
from rclip.compliance.common import _required_string


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
