import hashlib
import json
from pathlib import Path
import tarfile
import tomllib

import pytest

from rclip._compliance import ComplianceError
from rclip._compliance import augment_cyclonedx
from rclip._compliance import _binary_contains
from rclip._compliance import _deterministic_tar
from rclip._compliance import _review_python_packages
from rclip._compliance import build_corresponding_source
from rclip._compliance import collect_legal_materials
from rclip._compliance import normalize_scancode
from rclip._compliance import verify_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY = REPO_ROOT / "compliance" / "policy.toml"
NOTICES = REPO_ROOT / "compliance" / "notices"


def write_distribution(root: Path, name: str, version: str = "1.0", include_license: bool = True) -> None:
  dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
  dist_info.mkdir(parents=True)
  metadata = f"Name: {name}\nVersion: {version}\n"
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
    "This product includes DNG technology under license by Adobe.\n",
    encoding="utf-8",
  )
  license_path = legal_dir / "licenses/rclip-3.3.0/LICENSE"
  license_path.parent.mkdir(parents=True)
  license_path.write_text("rclip licence\n", encoding="utf-8")
  relative = license_path.relative_to(legal_dir).as_posix()
  report = {
    "schema_version": 1,
    "components": [
      {
        "type": "python",
        "name": "rclip",
        "version": "3.3.0",
        "license_files": [relative],
        "license_file_hashes": {relative: hashlib.sha256(license_path.read_bytes()).hexdigest()},
      }
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
  assert report["components"][0]["name"] == "rclip"
  assert (output / "licenses/rclip-3.3.0/licenses/LICENSE").is_file()
  assert (output / "notices/AOM-PATENT-LICENSE-1.0.txt").is_file()
  assert "This product includes DNG technology under license by Adobe." in (
    output / "THIRD_PARTY_NOTICES.txt"
  ).read_text(encoding="utf-8")


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


@pytest.mark.parametrize("name", ["unknown-package", "pi-heif"])
def test_collection_fails_closed_for_unreviewed_or_prohibited_packages(tmp_path: Path, name: str) -> None:
  write_distribution(tmp_path, name)

  with pytest.raises(ComplianceError):
    collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)


def test_collection_requires_a_license_file(tmp_path: Path) -> None:
  write_distribution(tmp_path, "rclip", include_license=False)

  with pytest.raises(ComplianceError, match="does not provide a licence file"):
    collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)


def test_collection_rejects_escaping_declared_license(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  write_distribution(root, "rclip", include_license=False)
  secret = tmp_path / "secret.txt"
  secret.write_text("build secret\n", encoding="utf-8")
  metadata = root / "rclip-1.0.dist-info/METADATA"
  metadata.write_text(f"Name: rclip\nVersion: 1.0\nLicense-File: {secret}\n", encoding="utf-8")

  with pytest.raises(ComplianceError, match="unsafe licence file"):
    collect_legal_materials(root, root / "legal", POLICY, NOTICES)


def test_collection_requires_review_for_version_changes(tmp_path: Path) -> None:
  write_distribution(tmp_path, "anyio", version="999")

  with pytest.raises(ComplianceError, match="unreviewed Python versions"):
    collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)


def test_reviewed_package_requires_a_version_policy() -> None:
  policy = {"reviewed_python_packages": ["example"]}

  with pytest.raises(ComplianceError, match="without a version policy: example"):
    _review_python_packages([{"name": "example", "version": "1"}], policy)

  policy["unversioned_python_packages"] = ["example"]
  _review_python_packages([{"name": "example", "version": "1"}], policy)


def test_policy_covers_locked_runtime_closure_on_every_platform() -> None:
  with (REPO_ROOT / "uv.lock").open("rb") as stream:
    locked_packages = {package["name"]: package for package in tomllib.load(stream)["package"]}
  with POLICY.open("rb") as stream:
    policy = tomllib.load(stream)

  closure: set[str] = set()
  pending = ["rclip"]
  while pending:
    name = pending.pop()
    if name in closure:
      continue
    closure.add(name)
    pending.extend(dependency["name"] for dependency in locked_packages[name].get("dependencies", []))

  reviewed = set(policy["reviewed_python_packages"])
  assert closure <= reviewed
  unversioned = set(policy["unversioned_python_packages"])
  reviewed_versions = policy["reviewed_python_versions"]
  for name in closure - unversioned:
    assert locked_packages[name]["version"] in reviewed_versions[name]


def test_rawpy_source_manifest_matches_reviewed_runtime_version() -> None:
  with (REPO_ROOT / "compliance/sources.toml").open("rb") as stream:
    source = tomllib.load(stream)["rawpy"]
  with POLICY.open("rb") as stream:
    policy = tomllib.load(stream)

  assert policy["reviewed_python_versions"]["rawpy"] == [source["version"]]
  assert policy["codecs"]["dng"]["native_components"][0]["version"] == source["libraw_version"]


def test_av1_requires_aom_patent_notice(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "libaom.so").write_bytes(b"native aom_codec dav1d avifDecoder")
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  (legal / "notices/AOM-PATENT-LICENSE-1.0.txt").unlink()

  with pytest.raises(ComplianceError, match="AV1 detected"):
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


def test_scans_extensionless_native_executables_for_hevc(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "ffmpeg").write_bytes(b"\x7fELF x265_encoder_open")
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

  with pytest.raises(ComplianceError, match="unreviewed Python distributions"):
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


def test_verification_checks_bundled_policy_contents(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  (legal / "policy.toml").write_text("schema_version = 1\n", encoding="utf-8")

  with pytest.raises(ComplianceError, match="differs from the reviewed policy"):
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


def test_normalizes_scancode_output(tmp_path: Path) -> None:
  source = tmp_path / "scancode.json"
  target = tmp_path / "normalized.json"
  source.write_text(
    json.dumps(
      {
        "packages": [{"name": "Typing_Extensions", "version": "4", "declared_license_expression": "apache-2.0"}],
        "files": [
          {
            "path": "z/LICENSE",
            "license_detections": [{"license_expression": "apache-2.0"}, {"license_expression": "apache-2.0"}],
          }
        ],
      }
    ),
    encoding="utf-8",
  )

  normalized = normalize_scancode(source, target)

  assert normalized["packages"][0]["name"] == "typing-extensions"
  assert normalized["file_licenses"][0]["license_expressions"] == ["apache-2.0"]
  assert json.loads(target.read_text(encoding="utf-8")) == normalized


def test_source_archive_is_deterministic_and_excludes_disabled_submodules(tmp_path: Path) -> None:
  source = tmp_path / "source"
  (source / "external/LibRaw").mkdir(parents=True)
  (source / "external/LibRaw/COPYRIGHT").write_text("LibRaw\n", encoding="utf-8")
  disabled = source / "external/LibRaw-demosaic-pack-GPL2"
  disabled.mkdir()
  (disabled / "code.c").write_text("unused\n", encoding="utf-8")
  first = tmp_path / "first.tar.gz"
  second = tmp_path / "second.tar.gz"

  _deterministic_tar(source, first, "source", excluded=[Path("external/LibRaw-demosaic-pack-GPL2")])
  _deterministic_tar(source, second, "source", excluded=[Path("external/LibRaw-demosaic-pack-GPL2")])

  assert first.read_bytes() == second.read_bytes()
  with tarfile.open(first) as archive:
    names = archive.getnames()
  assert "source/external/LibRaw/COPYRIGHT" in names
  assert not any("demosaic-pack-GPL2" in name for name in names)


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


def test_pyinstaller_legal_pack_path_uses_project_root() -> None:
  spec = (REPO_ROOT / "release-utils/windows/pyinstaller.spec").read_text(encoding="utf-8")

  assert "project_root = Path(__file__).resolve().parents[2]" in spec
  assert "(str(legal_dir), 'legal')" in spec
