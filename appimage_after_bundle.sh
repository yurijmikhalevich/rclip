#!/usr/bin/env bash

set -e

if [[ "$GITHUB_ACTIONS" ]]; then
  # Install git into the appimage-builder docker image
  apt-get update && apt-get install -y git
fi

PYTHONHOME=$APPDIR/usr \
PYTHONPATH=$APPDIR/usr/lib/python3.8/site-packages:$APPDIR/usr/lib/python3.8 \
LD_LIBRARY_PATH=$APPDIR/usr/lib/x86_64-linux-gnu \
python3.8 -m pip install poetry==1.3.2 &&
python3.8 -m poetry export --without dev -f requirements.txt --output requirements.txt &&
python3.8 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" certifi setuptools wheel &&
python3.8 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" -r ./requirements.txt
