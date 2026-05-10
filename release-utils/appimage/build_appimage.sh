#!/usr/bin/env bash

set -euo pipefail

app_version=${APP_VERSION:?APP_VERSION is required}
recipe=./release-utils/appimage/appimage-builder.yml
appdir=./AppDir
artifact="rclip-${app_version}-x86_64.AppImage"
update_information='gh-releases-zsync|AppImageCrafters|python-appimage-example|latest|python-appimage-*x86_64.AppImage.zsync'

rm -rf "$appdir" appimage-build .bundle.yml "$artifact" "${artifact}.zsync"

appimage-builder --skip-tests --skip-appimage --recipe "$recipe"
ARCH=x86_64 appimagetool --comp zstd --mksquashfs-opt -Xcompression-level --mksquashfs-opt 3 --mksquashfs-opt -b --mksquashfs-opt 128K --no-appstream --updateinformation "$update_information" "$appdir" "$artifact"
