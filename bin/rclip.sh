#!/usr/bin/env sh

BASEDIR=$(dirname "$0")/..

PYTHONPATH="$BASEDIR" python3 "$BASEDIR/rclip/main.py" "$@"
