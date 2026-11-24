#!/usr/bin/env bash
# Builds the static ONNX Runtime libraries that rclip links on x86-64 Linux. ort's prebuilt ones are
# compiled for AVX2 and crash on older CPUs; this build targets baseline x86-64 and still picks the
# fastest kernels the CPU supports at run time.
#
# Usage: scripts/build_onnxruntime.sh <output directory>
# Then build rclip with ORT_LIB_LOCATION=<output directory>.
set -euo pipefail

output=$(realpath -m "$1")
work=$(mktemp -d)
python3 -I "$(dirname "$0")/fetch_onnxruntime_source.py" "$work/source"
# The second line leaves out what rclip's models don't use: traditional ML and text generation
# operators, data types other than dense numeric tensors, models with external data, and GPU
# discovery. The client package build doesn't keep idle threads spinning, as in ort's prebuilt
# ONNX Runtime that rclip links on macOS and Windows.
cmake -S "$work/source/cmake" -B "$work/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -Donnxruntime_BUILD_SHARED_LIB=OFF -Donnxruntime_BUILD_UNIT_TESTS=OFF -Donnxruntime_USE_TELEMETRY=OFF \
  -Donnxruntime_DISABLE_ML_OPS=ON -Donnxruntime_DISABLE_GENERATION_OPS=ON -Donnxruntime_DISABLE_SPARSE_TENSORS=ON \
  -Donnxruntime_DISABLE_OPTIONAL_TYPE=ON -Donnxruntime_DISABLE_FLOAT8_TYPES=ON -Donnxruntime_DISABLE_FLOAT4_TYPES=ON \
  -Donnxruntime_DISABLE_STRING_TYPE=ON -Donnxruntime_DISABLE_EXTERNAL_INITIALIZERS=ON \
  -Donnxruntime_DISABLE_DEVICE_DISCOVERY=ON -Donnxruntime_CLIENT_PACKAGE_BUILD=ON
cmake --build "$work/build"
# ort links re2, which the default target doesn't build
cmake --build "$work/build" --target re2
mkdir -p "$output"
# ort finds the libraries by their paths in the build directory
(cd "$work/build" && find . -name '*.a' -exec cp --parents {} "$output/" \;)
rm -rf "${work:?}"
