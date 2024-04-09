#!/usr/bin/env bash

set -e

PYTHONHOME=$APPDIR/usr \
PYTHONPATH=$APPDIR/usr/lib/python3.9/site-packages:$APPDIR/usr/lib/python3.9 \
LD_LIBRARY_PATH=$APPDIR/usr/lib/x86_64-linux-gnu \
python3.9 -m pip install poetry==1.8.2 &&
python3.9 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" certifi setuptools wheel &&
python3.9 -m poetry build &&
python3.9 -m poetry export --output requirements.txt &&
python3.9 -m pip install --extra-index-url https://download.pytorch.org/whl/cpu --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" -r requirements.txt &&
python3.9 -m pip install --no-dependencies --isolated --no-input --prefix="$APPDIR/usr" dist/*.whl
