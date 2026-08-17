import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tomllib

import pytest

from rclip._compliance.common import ComplianceError
from rclip._compliance.legal import collect_legal_materials
from rclip._compliance.licenses import _declared_license_expression
from rclip._compliance.licenses import _validate_python_packages
from rclip._compliance.native import _binary_contains
from rclip._compliance.native import _native_candidates
from rclip._compliance.native import _native_component_versions
from rclip._compliance.policy import _locked_runtime_versions
from rclip._compliance.policy import load_policy
from rclip._compliance.sbom import augment_cyclonedx
from rclip._compliance.source import _deterministic_tar
from rclip._compliance.source import build_corresponding_source
from rclip._compliance.verify import verify_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY = REPO_ROOT / "compliance" / "policy.toml"
NOTICES = REPO_ROOT / "compliance" / "notices"


def dng_attribution() -> str:
  with POLICY.open("rb") as stream:
    return tomllib.load(stream)["codecs"]["dng"]["required_text"]


def write_distribution(
  root: Path,
  name: str,
  version: str = "1.0",
  include_license: bool = True,
  license_expression: str | None = "MIT",
  legacy_license: str | None = None,
) -> None:
  dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
  dist_info.mkdir(parents=True)
  metadata = f"Name: {name}\nVersion: {version}\n"
  if license_expression is not None:
    metadata += f"License-Expression: {license_expression}\n"
  if legacy_license is not None:
    metadata += f"License: {legacy_license}\n"
  if include_license:
    metadata += "License-File: LICENSE\n"
    licenses = dist_info / "licenses"
    licenses.mkdir()
    (licenses / "LICENSE").write_text(f"Licence for {name}\n", encoding="utf-8")
  (dist_info / "METADATA").write_text(metadata, encoding="utf-8")


def copy_notice(legal_dir: Path, filename: str) -> None:
  target = legal_dir / "notices" / filename
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_bytes((NOTICES / filename).read_bytes())


def write_legal_pack(legal_dir: Path) -> None:
  legal_dir.mkdir(parents=True, exist_ok=True)
  for notice in NOTICES.glob("*.txt"):
    copy_notice(legal_dir, notice.name)
  (legal_dir / "THIRD_PARTY_NOTICES.txt").write_text(
    dng_attribution() + "\n",
    encoding="utf-8",
  )
  license_path = legal_dir / "licenses/rclip-3.3.0/LICENSE"
  license_path.parent.mkdir(parents=True)
  license_path.write_text("rclip licence\n", encoding="utf-8")
  relative = license_path.relative_to(legal_dir).as_posix()
  with POLICY.open("rb") as stream:
    policy = tomllib.load(stream)
  native_versions = {
    component["name"]: component["version"]
    for codec in policy["codecs"].values()
    for component in codec.get("native_components", [])
  }
  report = {
    "schema_version": 1,
    "components": [
      {
        "type": "python",
        "name": "rclip",
        "version": "3.3.0",
        "license_expression": policy["approved_python_licenses"]["rclip"],
        "license_files": [relative],
        "license_file_hashes": {relative: hashlib.sha256(license_path.read_bytes()).hexdigest()},
      }
    ],
    "native_component_versions": [
      {"name": name, "version": version, "source": "test fixture"} for name, version in sorted(native_versions.items())
    ],
  }
  (legal_dir / "compliance-report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
  (legal_dir / "policy.toml").write_bytes(POLICY.read_bytes())


def test_collects_namespaced_licenses_and_common_notices(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  write_distribution(root, "rclip", "3.3.0")
  output = root / "share" / "doc" / "rclip"

  report = collect_legal_materials(root, output, POLICY, NOTICES)

  assert "root" not in report
  assert "codec_notice_hashes" not in report
  assert report["components"][0]["name"] == "rclip"
  assert report["components"][0]["license_expression"] == "MIT"
  assert (output / "licenses/rclip-3.3.0/licenses/LICENSE").is_file()
  assert (output / "notices/AOM-PATENT-LICENSE-1.0.txt").is_file()
  assert dng_attribution() in (output / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")


def test_collects_and_indexes_bundled_system_licenses(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  write_distribution(root, "rclip", "3.3.0")
  copyright_file = root / "usr/share/doc/libffi7/copyright"
  copyright_file.parent.mkdir(parents=True)
  copyright_file.write_text("libffi licence\n", encoding="utf-8")
  output = root / "usr/share/doc/rclip"

  report = collect_legal_materials(root, output, POLICY, NOTICES)

  assert any(component["type"] == "system" and component["name"] == "libffi7" for component in report["components"])
  assert (output / "licenses/system/libffi7/copyright").is_file()
  assert "Bundled system packages" in (output / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
  assert not verify_bundle(root, output, POLICY, None)["errors"]


def test_does_not_collect_python_runtime_license_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  root = tmp_path / "runtime"
  write_distribution(root, "rclip", "3.3.0")
  python_license = root / "usr/share/doc/python3.11/copyright"
  python_license.parent.mkdir(parents=True)
  python_license.write_text("CPython licence\n", encoding="utf-8")
  monkeypatch.setattr(
    "rclip._compliance.legal._python_runtime_license",
    lambda _root: ("3.11.0", python_license),
  )
  output = root / "usr/share/doc/rclip"

  report = collect_legal_materials(root, output, POLICY, NOTICES, include_python_runtime=True)

  assert [(component["type"], component["name"]) for component in report["components"]].count(
    ("runtime", "cpython")
  ) == 1
  assert not (output / "licenses/system/python3.11/copyright").exists()
  notices = (output / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
  assert notices.count("- cpython 3.11.0") == 1
  assert "- python3.11" not in notices


def test_omits_external_python_runtime_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  root = tmp_path / "runtime"
  write_distribution(root, "rclip", "3.3.0")
  monkeypatch.setattr(
    "rclip._compliance.legal._python_runtime_license",
    lambda _root: pytest.fail("external Python runtime should not be inspected"),
  )

  report = collect_legal_materials(root, root / "legal", POLICY, NOTICES)

  assert not any(component["name"] == "cpython" for component in report["components"])


@pytest.mark.parametrize("name", ["unknown-package", "pi-heif", "pillow-heif"])
def test_collection_fails_closed_for_disallowed_or_prohibited_packages(tmp_path: Path, name: str) -> None:
  write_distribution(tmp_path, name)

  with pytest.raises(ComplianceError):
    collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)


def test_collection_requires_a_license_file(tmp_path: Path) -> None:
  write_distribution(tmp_path, "rclip", include_license=False)

  with pytest.raises(ComplianceError, match="does not provide a licence file"):
    collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)


def test_collection_uses_only_the_locked_version_license_override(tmp_path: Path) -> None:
  write_distribution(
    tmp_path,
    "flatbuffers",
    version="25.12.19",
    include_license=False,
    license_expression="Apache-2.0",
  )

  collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)

  assert (tmp_path / "legal/licenses/flatbuffers-25.12.19/LICENSE.txt").is_file()


def test_collection_rejects_escaping_declared_license(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  write_distribution(root, "rclip", include_license=False)
  secret = tmp_path / "secret.txt"
  secret.write_text("build secret\n", encoding="utf-8")
  metadata = root / "rclip-1.0.dist-info/METADATA"
  metadata.write_text(
    f"Name: rclip\nVersion: 1.0\nLicense-Expression: MIT\nLicense-File: {secret}\n",
    encoding="utf-8",
  )

  with pytest.raises(ComplianceError, match="unsafe licence file"):
    collect_legal_materials(root, root / "legal", POLICY, NOTICES)


def test_collection_rejects_disallowed_version_changes(tmp_path: Path) -> None:
  write_distribution(tmp_path, "anyio", version="999")

  with pytest.raises(ComplianceError, match="disallowed Python versions"):
    collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)


def test_collection_rejects_unapproved_or_unknown_licences(tmp_path: Path) -> None:
  write_distribution(tmp_path, "anyio", version="4.14.2", license_expression="GPL-3.0-only")
  with pytest.raises(ComplianceError, match="unapproved Python licence"):
    collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)

  root = tmp_path / "unknown"
  write_distribution(root, "anyio", version="4.14.2", license_expression=None)
  with pytest.raises(ComplianceError, match="unknown Python licence declaration"):
    collect_legal_materials(root, root / "legal", POLICY, NOTICES)


def test_collection_normalizes_legacy_licence_metadata(tmp_path: Path) -> None:
  write_distribution(
    tmp_path,
    "coremltools",
    version="9.0",
    license_expression=None,
    legacy_license="BSD",
  )

  collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)

  report = json.loads((tmp_path / "legal/compliance-report.json").read_text(encoding="utf-8"))
  assert report["components"][0]["license_expression"] == "BSD-3-Clause"


@pytest.mark.parametrize(
  "expression",
  ["Apache-2.0", "BSD-3-Clause", "MIT", "MPL-2.0", "MPL-2.0 AND MIT", "WTFPL"],
)
def test_accepts_spdx_expressions_from_legacy_licence_metadata(expression: str) -> None:
  record = {
    "declared_license_expression": "",
    "legacy_license": expression,
    "license_classifiers": [],
  }

  assert _declared_license_expression(record) == expression


def test_unversioned_package_accepts_any_version() -> None:
  policy = {
    "unversioned_python_packages": ["example"],
    "approved_python_licenses": {"example": "MIT"},
  }

  _validate_python_packages([{"name": "example", "version": "999"}], policy, {})


def test_policy_covers_locked_runtime_closure_on_every_platform() -> None:
  locked_versions = _locked_runtime_versions(REPO_ROOT / "uv.lock")
  with POLICY.open("rb") as stream:
    policy = tomllib.load(stream)

  unversioned = set(policy["unversioned_python_packages"])
  assert set(policy["approved_python_licenses"]) == locked_versions.keys() | unversioned


def test_rawpy_source_manifest_matches_allowed_runtime_version() -> None:
  with (REPO_ROOT / "compliance/sources.toml").open("rb") as stream:
    source = tomllib.load(stream)["rawpy"]
  with POLICY.open("rb") as stream:
    policy = tomllib.load(stream)

  assert _locked_runtime_versions(REPO_ROOT / "uv.lock")["rawpy"] == {source["version"]}
  assert policy["approved_python_licenses"]["rawpy"] == "MIT"
  assert policy["codecs"]["dng"]["native_components"][0]["version"] == source["libraw_version"]


def test_policy_requires_native_component_fields(tmp_path: Path) -> None:
  policy = tmp_path / "policy.toml"
  policy.write_text(
    POLICY.read_text(encoding="utf-8").replace('name = "libavif"\n', "", 1),
    encoding="utf-8",
  )

  with pytest.raises(ComplianceError, match="av1 native component is missing name"):
    load_policy(policy)


@pytest.mark.parametrize(
  ("original", "replacement", "error"),
  [
    ("schema_version = 1", "schema_version = true", "unsupported policy schema"),
    ("allowed = true", 'allowed = "false"', "invalid allowed"),
    ('anyio = "MIT"', 'anyio = ["MIT"]', "invalid approved licence expression"),
    ('path_markers = ["libavif", "_avif"]', 'path_markers = "libavif"', "invalid path_markers"),
    ('required_notice = "AOM-PATENT-LICENSE-1.0.txt"', 'unexpected = "value"', "unknown fields"),
  ],
)
def test_policy_rejects_invalid_field_types_and_unknown_fields(
  tmp_path: Path,
  original: str,
  replacement: str,
  error: str,
) -> None:
  policy = tmp_path / "policy.toml"
  policy.write_text(POLICY.read_text(encoding="utf-8").replace(original, replacement, 1), encoding="utf-8")

  with pytest.raises(ComplianceError, match=error):
    load_policy(policy)


def test_collects_native_versions_from_runtime_apis() -> None:
  with POLICY.open("rb") as stream:
    policy = tomllib.load(stream)
  expected = {
    component["name"]: component["version"]
    for codec in policy["codecs"].values()
    for component in codec.get("native_components", [])
  }

  reported = {component["name"]: component["version"] for component in _native_component_versions({"pillow", "rawpy"})}

  assert reported == expected


@pytest.mark.parametrize(
  ("flags", "error"),
  [
    (
      {"DEMOSAIC_PACK_GPL2": True, "DEMOSAIC_PACK_GPL3": False},
      "rawpy forbidden feature is not disabled: DEMOSAIC_PACK_GPL2=True",
    ),
    (
      {"DEMOSAIC_PACK_GPL2": False, "DEMOSAIC_PACK_GPL3": True},
      "rawpy forbidden feature is not disabled: DEMOSAIC_PACK_GPL3=True",
    ),
    (
      {"DEMOSAIC_PACK_GPL2": False},
      "rawpy does not report required feature flag DEMOSAIC_PACK_GPL3",
    ),
  ],
)
def test_rejects_enabled_or_unreported_rawpy_gpl_features(
  monkeypatch: pytest.MonkeyPatch, flags: dict[str, bool], error: str
) -> None:
  class Rawpy:
    libraw_version = (0, 22, 0)

    def __init__(self, feature_flags: dict[str, bool]) -> None:
      self.flags = feature_flags

  rawpy = Rawpy(flags)
  monkeypatch.setattr("rclip._compliance.native.importlib.import_module", lambda name: rawpy)

  with pytest.raises(ComplianceError, match=error):
    _native_component_versions({"rawpy"})


def test_sdist_includes_homebrew_compliance_inputs(tmp_path: Path) -> None:
  subprocess.run(
    ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
    cwd=REPO_ROOT,
    check=True,
    capture_output=True,
    text=True,
  )
  archives = list(tmp_path.glob("rclip-*.tar.gz"))
  assert len(archives) == 1

  with tarfile.open(archives[0]) as archive:
    names = {Path(name).relative_to(Path(name).parts[0]).as_posix() for name in archive.getnames()}

  expected = {
    "compliance/policy.toml",
    "uv.lock",
    *{
      path.relative_to(REPO_ROOT).as_posix()
      for directory in (REPO_ROOT / "compliance/notices", REPO_ROOT / "compliance/license-overrides")
      for path in directory.rglob("*")
      if path.is_file()
    },
  }
  assert expected <= names, f"sdist is missing compliance inputs: {sorted(expected - names)}"


def test_av1_requires_aom_patent_notice(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "libaom.so").write_bytes(b"native aom_codec dav1d avifDecoder")
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  (legal / "notices/AOM-PATENT-LICENSE-1.0.txt").unlink()

  with pytest.raises(ComplianceError, match="AOM-PATENT-LICENSE"):
    verify_bundle(root, legal, POLICY, None)

  copy_notice(legal, "AOM-PATENT-LICENSE-1.0.txt")
  assert "libaom.so" in verify_bundle(root, legal, POLICY, None)["detections"]["av1"]


def test_dng_requires_attribution_and_replaceable_libraw(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "_rawpy.so").write_bytes(b"\x7fELFnative libraw_r.so")
  legal = tmp_path / "legal"
  write_legal_pack(legal)

  with pytest.raises(ComplianceError, match="separately replaceable"):
    verify_bundle(root, legal, POLICY, None)

  (root / "libraw_r.so").write_bytes(b"\x7fELFnative")
  assert verify_bundle(root, legal, POLICY, None)["detections"]["dng"] == ["_rawpy.so", "libraw_r.so"]


def test_dng_rejects_an_unreferenced_libraw_decoy(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "_rawpy.so").write_bytes(b"\x7fELFstatically linked")
  (root / "libraw_r.so").write_bytes(b"\x7fELFunused")
  legal = tmp_path / "legal"
  write_legal_pack(legal)

  with pytest.raises(ComplianceError, match="separately replaceable"):
    verify_bundle(root, legal, POLICY, None)


def test_rejects_actual_hevc_implementation_but_not_libheif_api(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  (root / "libheif.so").write_bytes(b"VIPS_FOREIGN_HEIF_COMPRESSION_HEVC")

  assert not verify_bundle(root, legal, POLICY, None)["detections"]["hevc"]

  (root / "libde265.so").write_bytes(b"native")
  with pytest.raises(ComplianceError, match="forbidden HEVC"):
    verify_bundle(root, legal, POLICY, None)


def test_binary_markers_are_detected_across_read_boundaries(tmp_path: Path) -> None:
  binary = tmp_path / "codec.so"
  marker = b"x265_encoder_open"
  binary.write_bytes(b"x" * (4 * 1024 * 1024 - len(marker) // 2) + marker)

  assert _binary_contains(binary, [marker.decode("ascii")]) == [marker.decode("ascii")]


def test_binary_markers_can_be_matched_case_insensitively(tmp_path: Path) -> None:
  binary = tmp_path / "binding.so"
  binary.write_bytes(b"native LIBRAW_R.SO dependency")

  assert _binary_contains(binary, ["libraw_r.so"], casefold=True) == ["libraw_r.so"]


def test_native_candidates_confine_symlink_targets_to_root(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  bundled = root / "bundled.so"
  bundled.write_bytes(b"\x7fELFnative")
  host = tmp_path / "host.so"
  host.write_bytes(b"\x7fELFnative")
  bundled_link = root / "bundled-link.so"
  host_link = root / "host-link.so"
  try:
    bundled_link.symlink_to(bundled.name)
    host_link.symlink_to(host)
  except OSError:
    pytest.skip("symlinks are unavailable")

  assert set(_native_candidates(root)) == {bundled, bundled_link}


@pytest.mark.parametrize(
  "magic",
  (b"\x7fELF", b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf", b"\xbe\xba\xfe\xca", b"\xbf\xba\xfe\xca"),
)
def test_scans_extensionless_native_executables_for_hevc(tmp_path: Path, magic: bytes) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "ffmpeg").write_bytes(magic + b" x265_encoder_open")
  legal = tmp_path / "legal"
  write_legal_pack(legal)

  with pytest.raises(ComplianceError, match="forbidden HEVC"):
    verify_bundle(root, legal, POLICY, None)


def test_rejects_build_tool_in_runtime(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "uv.exe").write_bytes(b"binary")
  legal = tmp_path / "legal"
  write_legal_pack(legal)

  with pytest.raises(ComplianceError, match="build-only executable"):
    verify_bundle(root, legal, POLICY, None)


def test_syft_inventory_is_checked_against_dependency_policy(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  syft = tmp_path / "syft.json"
  syft.write_text(
    json.dumps({"artifacts": [{"name": "surprise", "version": "1", "type": "python", "purl": "pkg:pypi/surprise@1"}]}),
    encoding="utf-8",
  )

  with pytest.raises(ComplianceError, match="disallowed Python distributions"):
    verify_bundle(root, legal, POLICY, syft)


def test_syft_inventory_must_cover_the_legal_report(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  syft = tmp_path / "syft.json"
  syft.write_text(json.dumps({"artifacts": []}), encoding="utf-8")

  with pytest.raises(ComplianceError, match="Syft inventory contains no Python distributions"):
    verify_bundle(root, legal, POLICY, syft)


def test_syft_inventory_matches_the_legal_report(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  syft = tmp_path / "syft.json"
  syft.write_text(
    json.dumps({"artifacts": [{"name": "rclip", "version": "3.3.0", "type": "python"}]}),
    encoding="utf-8",
  )

  assert verify_bundle(root, legal, POLICY, syft)["syft_python_packages"] == [{"name": "rclip", "version": "3.3.0"}]


def test_syft_native_inventory_rejects_hevc_packages(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  syft = tmp_path / "syft.json"
  syft.write_text(
    json.dumps(
      {
        "artifacts": [
          {
            "name": "x265",
            "version": "4.1",
            "type": "deb",
            "purl": "pkg:deb/ubuntu/x265@4.1",
          }
        ]
      }
    ),
    encoding="utf-8",
  )

  with pytest.raises(ComplianceError, match="Syft package: x265 4.1"):
    verify_bundle(root, legal, POLICY, syft)


def test_verification_checks_component_licence_hashes(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  (legal / "licenses/rclip-3.3.0/LICENSE").write_text("altered\n", encoding="utf-8")

  with pytest.raises(ComplianceError, match="altered licence file"):
    verify_bundle(root, legal, POLICY, None)


def test_verification_checks_reviewed_licence_expressions(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  report_path = legal / "compliance-report.json"
  report = json.loads(report_path.read_text(encoding="utf-8"))
  report["components"][0]["license_expression"] = "GPL-3.0-only"
  report_path.write_text(json.dumps(report), encoding="utf-8")

  with pytest.raises(ComplianceError, match="unapproved licence GPL-3.0-only"):
    verify_bundle(root, legal, POLICY, None)


def test_verification_checks_bundled_policy_contents(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  (legal / "policy.toml").write_text("schema_version = 1\n", encoding="utf-8")

  with pytest.raises(ComplianceError, match="differs from the distribution policy"):
    verify_bundle(root, legal, POLICY, None)


def test_verification_checks_complete_notice_contents(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "libaom.so").write_bytes(b"native aom_codec dav1d avifDecoder")
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  (legal / "notices/AOM-PATENT-LICENSE-1.0.txt").write_text("", encoding="utf-8")

  with pytest.raises(ComplianceError, match="AOM-PATENT-LICENSE"):
    verify_bundle(root, legal, POLICY, None)


def test_all_codec_allowed_flags_are_enforced(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "libaom.so").write_bytes(b"native aom_codec dav1d avifDecoder")
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  denied_policy = tmp_path / "policy.toml"
  denied_policy.write_text(
    POLICY.read_text(encoding="utf-8").replace("[codecs.av1]\nallowed = true", "[codecs.av1]\nallowed = false"),
    encoding="utf-8",
  )
  (legal / "policy.toml").write_bytes(denied_policy.read_bytes())

  with pytest.raises(ComplianceError, match="forbidden AV1"):
    verify_bundle(root, legal, denied_policy, None)


def test_augments_and_checks_native_cyclonedx_components(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "libaom.so").write_bytes(b"native aom_codec dav1d avifDecoder")
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  source = tmp_path / "source.cdx.json"
  output = tmp_path / "output.cdx.json"
  source.write_text(json.dumps({"bomFormat": "CycloneDX", "specVersion": "1.6", "components": []}), encoding="utf-8")

  with pytest.raises(ComplianceError, match="native components missing from CycloneDX"):
    verify_bundle(root, legal, POLICY, None, source)

  result = augment_cyclonedx(source, output, root, legal, POLICY)

  assert ("libaom", "3.14.1") in {(item["name"], item["version"]) for item in result["components"]}
  assert verify_bundle(root, legal, POLICY, None, output)["native_components"]


def test_native_sbom_uses_reported_version_and_rejects_policy_drift(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "libaom.so").write_bytes(b"native aom_codec")
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  report_path = legal / "compliance-report.json"
  report = json.loads(report_path.read_text(encoding="utf-8"))
  next(component for component in report["native_component_versions"] if component["name"] == "libaom")["version"] = (
    "999"
  )
  report_path.write_text(json.dumps(report), encoding="utf-8")
  source = tmp_path / "source.cdx.json"
  output = tmp_path / "output.cdx.json"
  source.write_text(json.dumps({"bomFormat": "CycloneDX", "components": []}), encoding="utf-8")

  result = augment_cyclonedx(source, output, root, legal, POLICY)

  assert ("libaom", "999") in {(item["name"], item["version"]) for item in result["components"]}
  with pytest.raises(ComplianceError, match=r"libaom 999 \(allowed: 3\.14\.1\)"):
    verify_bundle(root, legal, POLICY, None, output)


def test_source_archive_is_deterministic_and_excludes_every_manifest_path(tmp_path: Path) -> None:
  with (REPO_ROOT / "compliance/sources.toml").open("rb") as stream:
    rawpy = tomllib.load(stream)["rawpy"]
  excluded_submodules = [Path(path) for path in rawpy["excluded_submodules"]]
  excluded_files = [Path(path) for path in rawpy["excluded_paths"]]
  excluded = [*excluded_submodules, *excluded_files]
  source = tmp_path / "source"
  (source / "external/LibRaw").mkdir(parents=True)
  (source / "external/LibRaw/COPYRIGHT").write_text("LibRaw\n", encoding="utf-8")
  for relative in excluded_submodules:
    directory = source / relative
    directory.mkdir(parents=True)
    (directory / "code.c").write_text("unused\n", encoding="utf-8")
  for relative in excluded_files:
    path = source / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture\n", encoding="utf-8")
  first = tmp_path / "first.tar.gz"
  second = tmp_path / "second.tar.gz"

  _deterministic_tar(source, first, "source", excluded=excluded)
  _deterministic_tar(source, second, "source", excluded=excluded)

  assert first.read_bytes() == second.read_bytes()
  with tarfile.open(first) as archive:
    archived = [Path(name) for name in archive.getnames()]
  assert Path("source/external/LibRaw/COPYRIGHT") in archived
  for relative in excluded:
    prefix = Path("source") / relative
    assert not any(path == prefix or prefix in path.parents for path in archived)


def test_source_archive_normalizes_non_executable_modes(tmp_path: Path) -> None:
  first_source = tmp_path / "first-source"
  second_source = tmp_path / "second-source"
  first_source.mkdir()
  second_source.mkdir()
  (first_source / "file.txt").write_text("same\n", encoding="utf-8")
  (second_source / "file.txt").write_text("same\n", encoding="utf-8")
  (first_source / "file.txt").chmod(0o600)
  (second_source / "file.txt").chmod(0o644)
  first = tmp_path / "first-mode.tar.gz"
  second = tmp_path / "second-mode.tar.gz"

  _deterministic_tar(first_source, first, "source")
  _deterministic_tar(second_source, second, "source")

  assert first.read_bytes() == second.read_bytes()


def test_source_manifest_schema_is_validated(tmp_path: Path) -> None:
  manifest = tmp_path / "sources.toml"
  manifest.write_text("schema_version = 999\n", encoding="utf-8")

  with pytest.raises(ComplianceError, match="unsupported source manifest schema"):
    build_corresponding_source(manifest, tmp_path / "source.tar.gz")


def test_source_manifest_requires_rawpy_fields(tmp_path: Path) -> None:
  manifest = tmp_path / "sources.toml"
  manifest.write_text("schema_version = 1\n[rawpy]\n", encoding="utf-8")

  with pytest.raises(ComplianceError, match="rawpy source manifest .* is missing repository"):
    build_corresponding_source(manifest, tmp_path / "source.tar.gz")


def test_pyinstaller_legal_pack_path_uses_project_root() -> None:
  spec = (REPO_ROOT / "release-utils/windows/pyinstaller.spec").read_text(encoding="utf-8")

  assert "project_root = Path(SPEC).resolve().parents[2]" in spec
  assert "Path(__file__)" not in spec
  assert "(str(legal_dir), 'legal')" in spec
