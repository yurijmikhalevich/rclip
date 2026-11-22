"""Downloads the ONNX Runtime 1.31.0 sources from Software Heritage and verifies every file.

microsoft/onnxruntime has been private since 2026-11-17, so the sources come from the archive of its
v1.31.0 tag (revision 402979ac1bef89dcd55add41eaa7277c407599eb, archived on 2026-11-14)."""

import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import urllib.request

VAULT = "https://archive.softwareheritage.org/api/1/vault/flat/swh:1:dir:6ebffa882d87ca25e05568f369892e4036124e14/"
# the git tree hash of the archived directory without its three submodule links, which the build doesn't use
TREE_HASH = "9d3f9620881393cfc9c233a8a09d9ff12c0b9629"


def blob(data: bytes) -> bytes:
  return hashlib.sha1(b"blob %d\0" % len(data) + data).digest()


def tree(path: str) -> bytes:
  entries = []
  for name in os.listdir(path):
    full = os.path.join(path, name)
    info = os.lstat(full)
    if stat.S_ISLNK(info.st_mode):
      mode, digest = b"120000", blob(os.readlink(full).encode())
    elif stat.S_ISDIR(info.st_mode):
      mode, digest = b"40000", tree(full)
    else:
      mode = b"100755" if info.st_mode & 0o111 else b"100644"
      with open(full, "rb") as file:
        digest = blob(file.read())
    # git sorts directories as if their names ended with a slash
    entries.append((name.encode() + (b"/" if mode == b"40000" else b""), mode, name.encode(), digest))
  body = b"".join(mode + b" " + name + b"\0" + digest for _, mode, name, digest in sorted(entries))
  return hashlib.sha1(b"tree %d\0" % len(body) + body).digest()


def call(method: str) -> dict:
  return json.load(urllib.request.urlopen(urllib.request.Request(VAULT, method=method), timeout=120))


destination = sys.argv[1]
status = call("POST")
while status["status"] not in ("done", "failed"):
  print("waiting for the archive:", status["status"], flush=True)
  time.sleep(15)
  status = call("GET")
assert status["status"] == "done", status
os.makedirs(destination)
archive = destination + ".tar.gz"
subprocess.run(
  ["curl", "-sSfL", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", archive, status["fetch_url"]], check=True
)
subprocess.run(["tar", "xzf", archive, "-C", destination, "--strip-components=1"], check=True)
actual = tree(destination).hex()
assert actual == TREE_HASH, f"the sources hash to {actual}, expected {TREE_HASH}"
print("verified the ONNX Runtime sources:", actual)
