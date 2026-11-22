#!/usr/bin/env bash
# Packages the binary built by `cargo auditable build --release` into dist/: an archive with the
# binary and the notices that must accompany it, a wheel for PyPI, and a CycloneDX SBOM.
#
# Usage: release-utils/package.sh <target triple>
# Needs cargo-about, jq, Syft, and Python 3.11 or newer.
set -euo pipefail
cd "$(dirname "$0")/.."

target=$1
version=$(cargo metadata --locked --no-deps --format-version 1 | jq -r '.packages[0].version' | tr -d '\r')
name=rclip-$version-$target
binary=target/release/rclip
if [[ $target == *windows* ]]; then
  binary+=.exe
fi

rm -rf "dist/$name"
mkdir -p "dist/$name"
cp "$binary" LICENSE "dist/$name"
scripts/third_party_notices.sh --target "$target" > "dist/$name/THIRD_PARTY_NOTICES.txt"

# cargo-auditable embeds the list of crates into the binary; LibRaw and ONNX Runtime aren't crates
syft scan "file:$binary" -o cyclonedx-json |
  jq --slurpfile native compliance/native_components.json '.components += $native[0]' > "dist/$name.cdx.json"
if ! jq -e 'any(.components[]; .name == "rusqlite")' "dist/$name.cdx.json" > /dev/null; then
  echo "error: the SBOM doesn't list the crates compiled into $binary" >&2
  exit 1
fi

"$(command -v python3 || command -v python)" -I release-utils/pypi/build_wheel.py "dist/$name" "$target" dist
tar -C dist -czf "dist/$name.tar.gz" "$name"
rm -r "dist/${name:?}"
