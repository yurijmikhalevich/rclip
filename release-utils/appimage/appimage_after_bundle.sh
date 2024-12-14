#!/usr/bin/env bash

set -e

PYTHONHOME=$APPDIR/usr \
PYTHONPATH=$APPDIR/usr/lib/python3/dist-packages:$APPDIR/usr/lib/python3.10 \
LD_LIBRARY_PATH=$APPDIR/usr/lib/x86_64-linux-gnu \
which python3.10 &&
python3.10 -m pip install poetry==1.8.4 &&
python3.10 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" certifi setuptools wheel &&
python3.10 -m poetry build &&
python3.10 -m poetry export --output requirements.txt &&
python3.10 -m pip install --extra-index-url https://download.pytorch.org/whl/cpu --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" -r requirements.txt &&
python3.10 -m pip install --no-dependencies --isolated --no-input --prefix="$APPDIR/usr" dist/*.whl
