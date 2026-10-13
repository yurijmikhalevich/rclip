# Vendored sources

## LibRaw

`libraw/` contains the library sources, headers, and licenses of
[LibRaw 0.22.0](https://github.com/LibRaw/LibRaw/releases/tag/0.22.0), the version
the Python version of rclip uses through rawpy. LibRaw is dual-licensed under the
LGPL 2.1 and the CDDL 1.0; see `libraw/LICENSE.LGPL` and `libraw/LICENSE.CDDL`.

It was extracted from the release archive
(`https://github.com/LibRaw/LibRaw/archive/refs/tags/0.22.0.tar.gz`, SHA-256
`5a11327a9cef2496d6a4335e8da30a1604460b6c545a30fe7588cf4c00a0fcae`), keeping only
`libraw/`, `internal/`, `src/` (without the `*_ph.cpp` placeholders, which are not
part of the library), and the license files. `build.rs` compiles it.
