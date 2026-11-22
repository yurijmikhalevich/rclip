"""Times rclip built with the prebuilt ONNX Runtime against rclip built with the baseline one, alternating."""

import os
import statistics
import subprocess
import sys
import time

work, repo = sys.argv[1], os.getcwd()
VARIANTS = ("prebuilt", "baseline")


def run(variant: str, cwd: str, args: list[str], datadir: str, fresh: bool) -> dict:
  directory = f"{work}/data/{variant}-{datadir}"
  database = f"{directory}/db.sqlite3"
  if fresh and os.path.exists(database):
    os.remove(database)
  started = time.perf_counter()
  process = subprocess.Popen(
    [f"{work}/rclip-{variant}", *args],
    cwd=cwd,
    env={**os.environ, "RCLIP_DATADIR": directory},
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  stdout, stderr = process.communicate()
  wall = time.perf_counter() - started
  assert process.returncode == 0, stderr.decode()[-3000:]
  return {"wall": wall, "stdout": stdout}


SCENARIOS = [
  ("Index 100 12 MP JPEGs from scratch", f"{work}/images/photos", ["cat"], "data", True, 3),
  ("Index 600 640x480 JPEGs from scratch", f"{work}/images/small", ["cat"], "data", True, 3),
  ("Re-check 600 unchanged images and search", f"{work}/images/small", ["cat"], "data", False, 7),
  ("Search 600 images by text, --no-indexing", f"{work}/images/small", ["-n", "cat"], "data", False, 9),
  ("Search 300,000 images by text, --no-indexing", f"{work}/library", ["-n", "cat"], "big", False, 7),
  (
    "Search 300,000 images by image, --no-indexing",
    f"{work}/library",
    ["-n", f"{repo}/tests/e2e/images/cat.jpg"],
    "big",
    False,
    7,
  ),
]

lines = [
  "| Scenario | Prebuilt (AVX2) | Baseline | Baseline / prebuilt | Same output |",
  "|---|---|---|---|---|",
]
for name, cwd, args, datadir, fresh, runs in SCENARIOS:
  samples: dict[str, list[dict]] = {variant: [] for variant in VARIANTS}
  for _ in range(runs):
    for variant in VARIANTS:
      samples[variant].append(run(variant, cwd, args, datadir, fresh))
  cells = []
  medians = {}
  for variant in VARIANTS:
    walls = [sample["wall"] for sample in samples[variant]]
    medians[variant] = statistics.median(walls)
    cells.append(f"{medians[variant]:.2f} s ({min(walls):.2f}-{max(walls):.2f})")
  same = samples["prebuilt"][-1]["stdout"] == samples["baseline"][-1]["stdout"]
  lines.append(f"| {name} | {cells[0]} | {cells[1]} | {medians['baseline'] / medians['prebuilt']:.2f}x | {same} |")
  print(lines[-1], flush=True)

summary = "\n".join(lines) + "\n\nMedian of the runs, with the fastest and slowest in parentheses.\n"
with open(os.environ.get("GITHUB_STEP_SUMMARY", os.devnull), "a") as file:
  file.write(summary)
