#!/usr/bin/env bash

set -e

PYTHONHOME=$APPDIR/usr \
PYTHONPATH=$APPDIR/usr/lib/python3/dist-packages:$APPDIR/usr/lib/python3.11 \
LD_LIBRARY_PATH=$APPDIR/usr/lib/x86_64-linux-gnu \
which python3.11 &&
python3.11 -m pip install uv==0.11.12 &&
python3.11 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" setuptools wheel &&
python3.11 -m uv build &&
python3.11 -m uv export --locked --format requirements.txt --no-dev --no-editable --no-emit-project --no-hashes --output-file requirements.txt &&
python3.11 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" -r requirements.txt &&
python3.11 -m pip install --no-dependencies --isolated --no-input --prefix="$APPDIR/usr" dist/*.whl &&
python3.11 -m pip install --upgrade --isolated --no-input --ignore-installed --target="$APPDIR/usr/lib/python3.11/site-packages" certifi
