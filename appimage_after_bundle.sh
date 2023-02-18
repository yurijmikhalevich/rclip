#!/usr/bin/env bash

set -e

if [[ "$GITHUB_ACTIONS" ]]; then
  # Install git into the appimage-builder docker image
  # And reinstall Python to ensure compatibility with libffi7
  apt-get update && apt-get purge -y python3 && apt-get install -y git python3 python3-pip python3-setuptools
fi

PYTHONHOME=$APPDIR/usr \
PYTHONPATH=$APPDIR/usr/lib/python3.8/site-packages:$APPDIR/usr/lib/python3.8 \
LD_LIBRARY_PATH=$APPDIR/usr/lib/x86_64-linux-gnu \
cat /etc/lsb-release &&
whomai &&
which python3.8 &&
python3.8 -m pip install setuptools &&
python3.8 -m pip install poetry==1.3.2 &&
python3.8 -m poetry export --without-hashes --without dev -f requirements.txt --output requirements.txt &&
python3.8 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" certifi setuptools wheel &&
python3.8 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="$APPDIR/usr" -r ./requirements.txt
