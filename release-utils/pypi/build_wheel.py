"""Builds a wheel that installs the rclip binary, so that `pip install rclip` keeps working.

Usage: build_wheel.py <directory with the binary and its notices> <target triple> <output directory>"""

import base64
import hashlib
import re
import sys
import tomllib
import zipfile
from pathlib import Path

PLATFORM_TAGS = {
  # the Linux binaries are built on Ubuntu 22.04, which has glibc 2.35
  "x86_64-unknown-linux-gnu": "manylinux_2_35_x86_64",
  "aarch64-unknown-linux-gnu": "manylinux_2_35_aarch64",
  "aarch64-apple-darwin": "macosx_11_0_arm64",
  "x86_64-pc-windows-msvc": "win_amd64",
}
CLASSIFIERS = [
  "Environment :: Console",
  "Intended Audience :: End Users/Desktop",
  "Operating System :: MacOS",
  "Operating System :: Microsoft :: Windows",
  "Operating System :: POSIX :: Linux",
  "Programming Language :: Rust",
  "Topic :: Multimedia :: Graphics",
  "Topic :: Scientific/Engineering :: Image Recognition",
  "Topic :: Utilities",
]

package_dir, target, output_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
repository = Path(__file__).resolve().parents[2]
package = tomllib.loads((repository / "Cargo.toml").read_text())["package"]
# Cargo versions are SemVer ("4.2.0-rc.1"), while wheels need PEP 440 ("4.2.0rc1")
version = re.sub(r"-(a)lpha\.|-(b)eta\.|-(rc)\.", r"\1\2\3", package["version"])
tag = f"py3-none-{PLATFORM_TAGS[target]}"
dist_info = f"rclip-{version}.dist-info"

metadata = [
  "Metadata-Version: 2.4",
  "Name: rclip",
  f"Version: {version}",
  f"Summary: {package['description']}",
  f"Author-email: {package['authors'][0]}",
  f"License-Expression: {package['license']}",
  "License-File: LICENSE",
  "License-File: THIRD_PARTY_NOTICES.txt",
  f"Keywords: {','.join(package['keywords'])}",
  *(f"Classifier: {classifier}" for classifier in CLASSIFIERS),
  f"Project-URL: Repository, {package['repository']}",
  "Requires-Python: >=3.8",
  "Description-Content-Type: text/markdown",
  "",
  (repository / "README.md").read_text(),
]
wheel = ["Wheel-Version: 1.0", "Generator: rclip", "Root-Is-Purelib: false", f"Tag: {tag}", ""]
files = {
  f"{dist_info}/METADATA": "\n".join(metadata).encode(),
  f"{dist_info}/WHEEL": "\n".join(wheel).encode(),
  f"{dist_info}/licenses/LICENSE": (package_dir / "LICENSE").read_bytes(),
  f"{dist_info}/licenses/THIRD_PARTY_NOTICES.txt": (package_dir / "THIRD_PARTY_NOTICES.txt").read_bytes(),
}
# pip copies everything under scripts/ into the environment's bin directory
binary = next(path for path in package_dir.iterdir() if path.stem == "rclip")
files[f"rclip-{version}.data/scripts/{binary.name}"] = binary.read_bytes()
record = [
  f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()},{len(data)}"
  for name, data in files.items()
]
files[f"{dist_info}/RECORD"] = "\n".join([*record, f"{dist_info}/RECORD,,", ""]).encode()

output_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(output_dir / f"rclip-{version}-{tag}.whl", "w", zipfile.ZIP_DEFLATED) as archive:
  for name, data in files.items():
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_DEFLATED
    # a regular file, executable in the case of the binary; pip ignores the mode without the file type
    info.external_attr = (0o100755 if "/scripts/" in name else 0o100644) << 16
    archive.writestr(info, data)
