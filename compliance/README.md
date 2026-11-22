# Distribution compliance

`policy.toml` is the allowlist for software shipped in rclip release
artifacts. `notices/` contains legal terms that must accompany codec
implementations. Dependency and packaging changes must keep this policy and the
collected legal materials accurate.

`license-overrides/` contains complete licence texts only when an upstream
wheel omits them. Overrides are scoped as `<package>/<version>/` and must be
checked against that exact locked dependency version.

`model_repository/` contains the model card and legal materials published with
the converted model. Required OpenCLIP and OpenAI CLIP notices are included in
rclip distributions, and rclip downloads the complete set from Hugging Face
alongside the model artifacts.

Builds use `python -m rclip._compliance` to collect dependency licences and
native versions, create a third-party notice index, inspect native codec
binaries, enrich and verify Syft inventories, and validate every collected
legal file by SHA-256. Python versions come from the runtime dependency closure
in `uv.lock`; their declared licence expressions are checked against the
reviewed expressions in `policy.toml`. Native versions come from the installed
bindings' runtime APIs and are checked against `policy.toml`.
Disallowed Python distributions and forbidden codecs fail release policy checks.
Complete ScanCode output is available for human inspection; CI does not
interpret it or require separate sign-off.

The DNG notice required by Adobe is:

> This product includes DNG technology under license by Adobe.

The rawpy corresponding-source archive deliberately initializes only LibRaw
and LibRaw-cmake. The optional GPL2/GPL3 demosaic packs are disabled in the
wheel build and excluded from the archive. Release collection fails unless
rawpy reports both `DEMOSAIC_PACK_GPL2` and `DEMOSAIC_PACK_GPL3` as disabled.
Large test image fixtures, which are not needed to build or modify rawpy or
LibRaw, are also excluded.

textual-image is redistributed under LGPL-3.0-or-later. Self-contained release
artifacts include the LGPLv3 and GPLv3 texts, and each GitHub release publishes
the exact textual-image source archive from `uv.lock`. The release's tagged
rclip source archive provides the corresponding application code needed to
modify or relink that library.

## Dependency policy workflow

When `uv.lock`, a dependency declaration, or packaging changes:

1. Check the new component's copyright licence, patent terms, notices, and
   native libraries.
2. Add new distributions and their reviewed SPDX licence expressions to
   `approved_python_licenses` in `policy.toml`. Existing versions come directly
   from `uv.lock`; update the policy only when the package set or a declared
   licence changes. A wheel that omits a required licence needs a file under
   `license-overrides/<package>/<version>/`. Use
   `unversioned_python_packages` only for the rclip project distribution itself
   or distributions supplied outside its locked runtime dependency closure.
3. Update `sources.toml` when rawpy or LibRaw changes. Keep the textual-image
   source download and its generated notice aligned with `uv.lock`.
4. Run `uv run pytest tests/unit/test_compliance.py`.

Release builds run Syft against each assembled payload, add declared native
codec components, and attach CycloneDX SBOMs. The compliance reporting workflow
runs ScanCode on dependency or packaging pull requests, every Monday, and
against the latest released AppImage and MSI. The policy and exact-version
checks fail closed before a release is published.

## Rust version

The Rust version of rclip is a single statically linked binary, so its policy is checked on the
sources it is built from:

- `about.toml` lists the licences its crates may use.
- `scripts/third_party_notices.sh` fails when a crate needs any other licence or when `Cargo.lock`
  contains an HEVC or HEIF decoder. Otherwise, it prints the notices that must accompany the binary:
  the licence texts of the crates, of the libraries some crates bundle (libjpeg-turbo, OpenJPEG,
  zlib, and SQLite), and of LibRaw and ONNX Runtime.
- The compliance workflow runs the script whenever the dependencies, the LibRaw submodule, or the
  policy change.

HEIC images are decoded only by the operating system's codec. Never add an HEVC or HEIF decoder.

### LibRaw

rclip uses LibRaw 0.22.3, a git submodule in `vendor/libraw`, under the CDDL 1.0, one of the two
licences LibRaw offers; the other is the LGPL 2.1. The CDDL covers only LibRaw's own files, so it
puts no conditions on the rest of a statically linked binary. It requires that LibRaw's source code
stays available to everyone who gets the binary; the notices shipped with the binary say where to
get it.

- Check the licence of every new LibRaw version before moving the submodule to it. Its README says:
  "We do not guarantee that the licensing will not change in future versions."
- Don't patch LibRaw's files. If that becomes unavoidable, publish the changes under both of its
  licences, so that either stays usable.
- Never add LibRaw's GPL demosaic packs, and never add GPL code to the binary: the CDDL and the GPL
  are incompatible.
- When the submodule moves, update the LibRaw version in `scripts/third_party_notices.sh` and
  `vendor/README.md`.

### Dependency policy workflow

When `Cargo.lock`, the LibRaw submodule, or the ONNX Runtime version changes:

1. Check the copyright licence, patent terms, and notices of every new component, including the
   native libraries a crate bundles.
2. Add a new licence to `accepted` in `about.toml` only after reviewing it.
3. Add the licence files of a newly bundled native library to `scripts/third_party_notices.sh`.
4. When ONNX Runtime changes, replace `onnxruntime-<version>` with the `LICENSE` and
   `ThirdPartyNotices.txt` of the new version and update the script.
5. Run `scripts/third_party_notices.sh > /dev/null`.
