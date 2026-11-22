#!/usr/bin/env bash
# Wraps the Linux binary from a release archive into an AppImage in the current directory.
#
# Usage: release-utils/appimage/build_appimage.sh <rclip-<version>-x86_64-unknown-linux-gnu.tar.gz>
# Needs appimagetool.
set -euo pipefail

archive=$1
version=$(basename "$archive" -x86_64-unknown-linux-gnu.tar.gz)
version=${version#rclip-}
appdir=AppDir
here=$(dirname "$0")

rm -rf "$appdir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/doc/rclip" "$appdir/usr/share/icons/hicolor/256x256/apps"
tar xzf "$archive" --strip-components=1 -C "$appdir/usr/share/doc/rclip"
mv "$appdir/usr/share/doc/rclip/rclip" "$appdir/usr/bin/"
cp "$here/logo@256.png" "$appdir/usr/share/icons/hicolor/256x256/apps/rclip.png"
cp "$here/logo@256.png" "$appdir/rclip.png"
ln -s usr/bin/rclip "$appdir/AppRun"
cat > "$appdir/rclip.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=rclip
Comment=Semantic photo search for the command line
Exec=rclip
Icon=rclip
Terminal=true
Categories=Graphics;Utility;
DESKTOP

ARCH=x86_64 appimagetool --comp zstd --no-appstream "$appdir" "rclip-$version-x86_64.AppImage"
