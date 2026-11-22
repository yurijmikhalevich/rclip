#!/usr/bin/env bash
# Prints the licences and notices of everything compiled into the rclip binary, and fails when a
# dependency breaks the distribution policy described in compliance/README.md.
#
# Usage: scripts/third_party_notices.sh [cargo-about options, such as --target <triple>]
# Needs cargo-about and jq.
set -euo pipefail
cd "$(dirname "$0")/.."

# rclip must not ship an HEVC decoder: HEIC images are decoded by the operating system's codec.
if grep -E '^name = "(libde265-sys|libheif-rs|libheif-sys)"$' Cargo.lock >&2; then
  echo "error: Cargo.lock contains an HEVC or HEIF decoder" >&2
  exit 1
fi

# the Rust crates, grouped by licence; fails when a crate needs a licence about.toml doesn't accept
cargo about generate --locked --fail "$@" about.hbs

rule=$(printf '=%.0s' {1..80})
packages=$(cargo metadata --locked --format-version 1)

# Prints a titled section with the contents of the given files.
section() {
  printf '%s\n%s\n%s\n\n' "$rule" "$1" "$rule"
  shift
  for file in "$@"; do
    cat "$file"
    printf '\n\n'
  done
}

# Prints the source directory of a crate, where it keeps the licences of the library it bundles.
crate() {
  dirname "$(jq -r --arg name "$1" '.packages[] | select(.name == $name).manifest_path' <<< "$packages")"
}

# the libraries compiled from sources that the crates above bundle
section "libjpeg-turbo (bundled in turbojpeg-sys)" "$(crate turbojpeg-sys)"/libjpeg-turbo/{LICENSE.md,README.ijg}
section "OpenJPEG (bundled in openjpeg-sys)" "$(crate openjpeg-sys)/vendor/LICENSE"
section "zlib (bundled in libz-sys)" "$(crate libz-sys)/src/zlib/LICENSE"
section "SQLite (bundled in libsqlite3-sys)" <(echo "SQLite is in the public domain: https://www.sqlite.org/copyright.html")

# the libraries rclip itself brings
section "LibRaw 0.22.3, used under the Common Development and Distribution License 1.0. \
Its source code is at https://github.com/LibRaw/LibRaw/tree/0.22.3" \
  vendor/libraw/COPYRIGHT vendor/libraw/LICENSE.CDDL compliance/notices/ADOBE-DNG-PATENT-LICENSE.txt
section "ONNX Runtime 1.31.0" compliance/onnxruntime-1.31.0/{LICENSE,ThirdPartyNotices.txt}
