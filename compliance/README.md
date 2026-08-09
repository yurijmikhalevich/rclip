# Distribution compliance

`policy.toml` is the allowlist for software shipped in rclip release
artifacts. `notices/` contains legal terms that must accompany codec
implementations. Dependency and packaging changes must keep this policy and the
collected legal materials accurate.

`license-overrides/` contains complete licence texts only when an upstream
wheel omits them. Each override must be checked against that exact locked
dependency version.

Builds use `python -m rclip._compliance` to collect dependency licences and
native versions, create a third-party notice index, inspect native codec
binaries, enrich and verify Syft inventories, and validate every collected
legal file by SHA-256. Native versions come from the installed bindings'
runtime APIs and are checked against `policy.toml`.
Disallowed Python distributions and forbidden codecs fail release policy checks.
Complete ScanCode output is available for human inspection; CI does not
interpret it or require separate sign-off.

The DNG notice required by Adobe is:

> This product includes DNG technology under license by Adobe.

The rawpy corresponding-source archive deliberately initializes only LibRaw
and LibRaw-cmake. The optional GPL2/GPL3 demosaic packs are disabled in the
wheel build and excluded from the archive. Large test image fixtures, which
are not needed to build or modify rawpy or LibRaw, are also excluded.

## Dependency policy workflow

When `uv.lock`, a dependency declaration, or packaging changes:

1. Check the new component's copyright licence, patent terms, notices, and
   native libraries.
2. Update `allowed_python_versions` in `policy.toml`. A wheel that omits a
   required licence needs a version-specific file under `license-overrides/`.
   Use `unversioned_python_packages` only for the rclip project distribution
   itself or distributions supplied outside its locked runtime dependency
   closure.
3. Update `sources.toml` when rawpy or LibRaw changes.
4. Run `uv run pytest tests/unit/test_compliance.py`.

Release builds run Syft against each assembled payload, add declared native
codec components, and attach CycloneDX SBOMs. The compliance reporting workflow
runs ScanCode on dependency or packaging pull requests, every Monday, and
against the latest released AppImage and MSI. The policy and exact-version
checks fail closed before a release is published.
