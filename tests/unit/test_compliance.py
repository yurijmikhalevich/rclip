import json
from pathlib import Path
import tarfile
import tomllib

import pytest

from rclip._compliance import ComplianceError
from rclip._compliance import _binary_contains
from rclip._compliance import _deterministic_tar
from rclip._compliance import _review_python_packages
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
  (legal_dir / "compliance-report.json").write_text("{}\n", encoding="utf-8")
  (legal_dir / "policy.toml").write_bytes(POLICY.read_bytes())


def test_collects_namespaced_licenses_and_common_notices(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  write_distribution(root, "rclip", "3.3.0")
  output = root / "share" / "doc" / "rclip"

  report = collect_legal_materials(root, output, POLICY, NOTICES)

  assert report["components"][0]["name"] == "rclip"
  assert (output / "licenses/rclip-3.3.0/licenses/LICENSE").is_file()
  assert (output / "notices/AOM-PATENT-LICENSE-1.0.txt").is_file()
  assert "This product includes DNG technology under license by Adobe." in (
    output / "THIRD_PARTY_NOTICES.txt"
  ).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["unknown-package", "pi-heif"])
def test_collection_fails_closed_for_unreviewed_or_prohibited_packages(tmp_path: Path, name: str) -> None:
  write_distribution(tmp_path, name)

  with pytest.raises(ComplianceError):
    collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)


def test_collection_requires_a_license_file(tmp_path: Path) -> None:
  write_distribution(tmp_path, "rclip", include_license=False)

  with pytest.raises(ComplianceError, match="does not provide a licence file"):
    collect_legal_materials(tmp_path, tmp_path / "legal", POLICY, NOTICES)


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
    source_version = tomllib.load(stream)["rawpy"]["version"]
  with POLICY.open("rb") as stream:
    reviewed_versions = tomllib.load(stream)["reviewed_python_versions"]

  assert reviewed_versions["rawpy"] == [source_version]


def test_av1_requires_aom_patent_notice(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "libaom.so").write_bytes(b"native")
  legal = tmp_path / "legal"
  write_legal_pack(legal)
  (legal / "notices/AOM-PATENT-LICENSE-1.0.txt").unlink()

  with pytest.raises(ComplianceError, match="AV1 detected"):
    verify_bundle(root, legal, POLICY, None)

  copy_notice(legal, "AOM-PATENT-LICENSE-1.0.txt")
  assert verify_bundle(root, legal, POLICY, None)["detections"]["av1"] == ["libaom.so"]


def test_dng_requires_attribution_and_replaceable_libraw(tmp_path: Path) -> None:
  root = tmp_path / "runtime"
  root.mkdir()
  (root / "_rawpy.so").write_bytes(b"native")
  legal = tmp_path / "legal"
  write_legal_pack(legal)

  with pytest.raises(ComplianceError, match="separately replaceable"):
    verify_bundle(root, legal, POLICY, None)

  (root / "libraw_r.so").write_bytes(b"native")
  assert verify_bundle(root, legal, POLICY, None)["detections"]["dng"] == ["_rawpy.so", "libraw_r.so"]


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
