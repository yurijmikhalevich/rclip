#!/usr/bin/env bash

set -e

PYTHONHOME=$APPDIR/usr \
PYTHONPATH=$APPDIR/usr/lib/python3.8/site-packages:$APPDIR/usr/lib/python3.8 \
LD_LIBRARY_PATH=$APPDIR/usr/lib/x86_64-linux-gnu \
python3.8 -m pip install poetry==1.7.1 &&
python3.8 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" certifi setuptools wheel &&
python3.8 -m poetry build &&
python3.8 -m poetry export --output requirements.txt &&
python3.8 -m pip install --extra-index-url https://download.pytorch.org/whl/cpu --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" -r requirements.txt &&
python3.8 -m pip install --no-dependencies --isolated --no-input --prefix="$APPDIR/usr" dist/*.whl
