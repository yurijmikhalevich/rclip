#!/usr/bin/env bash

set -euo pipefail

export PYTHONHOME="$APPDIR/usr"
export PYTHONPATH="$APPDIR/usr/lib/python3/dist-packages:$APPDIR/usr/lib/python3.11:$APPDIR/usr/lib/python3.11/site-packages:$APPDIR/usr/local/lib/python3.11/dist-packages"
export LD_LIBRARY_PATH="$APPDIR/usr/lib/x86_64-linux-gnu"

RCLIP_BUILD_TOOLS_DIR=$(mktemp -d)
trap 'rm -rf "$RCLIP_BUILD_TOOLS_DIR"' EXIT
python3.11 -m pip install --isolated --no-input --target="$RCLIP_BUILD_TOOLS_DIR" uv==0.11.12
export PYTHONPATH="$RCLIP_BUILD_TOOLS_DIR:$PYTHONPATH"

python3.11 -m uv build
python3.11 -m uv export --locked --format requirements.txt --no-dev --no-editable --no-emit-project --no-hashes --output-file requirements.txt
python3.11 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" -r requirements.txt
python3.11 -m pip install --no-dependencies --isolated --no-input --prefix="$APPDIR/usr" dist/*.whl
python3.11 -m pip install --upgrade --isolated --no-input --ignore-installed --target="$APPDIR/usr/lib/python3.11/site-packages" certifi

# The bundled pip, setuptools, and wheel packages are build tools inherited
# from the Ubuntu Python packages. They are not needed by the application.
for site_packages in \
  "$APPDIR/usr/lib/python3/dist-packages" \
  "$APPDIR/usr/lib/python3.11/site-packages" \
  "$APPDIR/usr/local/lib/python3.11/dist-packages"; do
  if [[ -d "$site_packages" ]]; then
    find "$site_packages" -maxdepth 1 \
      \( -name pip -o -name 'pip-*.dist-info' \
      -o -name setuptools -o -name 'setuptools-*.dist-info' -o -name 'setuptools-*.egg-info' \
      -o -name wheel -o -name 'wheel-*.dist-info' -o -name 'wheel-*.egg-info' \
      -o -name pkg_resources -o -name _distutils_hack \) \
      -exec rm -rf '{}' +
  fi
done
for bin_dir in "$APPDIR/usr/bin" "$APPDIR/usr/local/bin"; do
  if [[ -d "$bin_dir" ]]; then
    find "$bin_dir" -maxdepth 1 -type f \( -name pip -o -name 'pip[0-9]*' -o -name wheel \) -delete
  fi
done

python3.11 -m rclip._compliance collect \
  --root "$APPDIR" \
  --output "$APPDIR/usr/share/doc/rclip" \
  --policy compliance/policy.toml \
  --common-notices compliance/notices \
  --include-python-runtime
