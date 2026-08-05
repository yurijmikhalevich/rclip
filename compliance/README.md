# Distribution compliance

`policy.toml` is the reviewed allowlist for software shipped in rclip release
artifacts. `notices/` contains legal terms that must accompany codec
implementations. Do not update either mechanically: dependency and packaging
changes require review.

`license-overrides/` contains complete licence texts only when an upstream
wheel omits them. Each override must be checked against that exact locked
dependency version during review.

Builds use `python -m rclip._compliance` to collect dependency licences,
create a third-party notice index, inspect native codec binaries, and verify
Syft inventories. ScanCode output is normalized into a smaller review report.
Unknown Python distributions and forbidden codecs fail the audit.

The DNG notice required by Adobe is:

> This product includes DNG technology under license by Adobe.

The rawpy corresponding-source archive deliberately initializes only LibRaw
and LibRaw-cmake. The optional GPL2/GPL3 demosaic packs are disabled in the
wheel build and excluded from the archive.

## Dependency review workflow

When `uv.lock`, a dependency declaration, or packaging changes:

1. Review the new component's copyright licence, patent terms, notices, and
   native libraries.
2. Update both `reviewed_python_packages` and `reviewed_python_versions` in
   `policy.toml`. A wheel that omits a required licence needs a version-reviewed
   file under `license-overrides/`. Use `unversioned_python_packages` only for
   distributions supplied outside rclip's locked runtime dependency closure.
3. Update `sources.toml` when rawpy or LibRaw changes.
4. Run `uv run pytest tests/unit/test_compliance.py`.

Release builds run Syft against each assembled payload and attach CycloneDX
SBOMs. The deep-audit workflow runs ScanCode on dependency or packaging pull
requests, every Monday, and against the latest released AppImage and MSI. The
policy and exact-version checks fail closed before a release is published.
