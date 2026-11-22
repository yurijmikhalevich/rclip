# Vendored sources

## LibRaw

`libraw/` is a git submodule pinned to [LibRaw 0.22.3](https://github.com/LibRaw/LibRaw/releases/tag/0.22.3).
Check it out with `git submodule update --init`; `build.rs` compiles it.

LibRaw is dual-licensed under the LGPL 2.1 and the CDDL 1.0; see `libraw/LICENSE.LGPL` and
`libraw/LICENSE.CDDL`. rclip uses it under the CDDL 1.0. Read the LibRaw section of
[`compliance/README.md`](../compliance/README.md) before upgrading or changing it.
