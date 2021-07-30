#!/usr/bin/env bash

set -e

echo $PATH
ls -lah /usr/bin/git
git --version
which git

PYTHONHOME=${APPDIR}/usr
PYTHONPATH=${APPDIR}/usr/lib/python3.8/site-packages:${APPDIR}/usr/lib/python3.8
LD_LIBRARY_PATH=${APPDIR}/usr/lib/x86_64-linux-gnu
python3.8 -m pip install pipenv &&
python3.8 -m pipenv lock -r > requirements.txt &&
python3.8 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="${APPDIR}/usr" certifi wheel &&
python3.8 -m pip install --upgrade --isolated --no-input --ignore-installed --prefix="${APPDIR}/usr" -r ./requirements.txt
