"""Build-time licence collection and codec compliance checks.

The implementation intentionally uses only the Python standard library so
release builders can run it inside partially assembled application bundles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from rclip._compliance.common import ComplianceError
from rclip._compliance.common import _json_dump
from rclip._compliance.legal import collect_legal_materials
from rclip._compliance.sbom import augment_cyclonedx
from rclip._compliance.source import build_corresponding_source
from rclip._compliance.verify import verify_bundle


def _path(value: str) -> Path:
  return Path(value)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)

  collect_parser = subparsers.add_parser("collect", help="collect third-party legal materials")
  collect_parser.add_argument("--root", type=_path, required=True)
  collect_parser.add_argument("--output", type=_path, required=True)
  collect_parser.add_argument("--policy", type=_path)
  collect_parser.add_argument("--common-notices", type=_path)
  collect_parser.add_argument(
    "--include-python-runtime",
    action="store_true",
    help="include the current CPython runtime and its licence",
  )

  verify_parser = subparsers.add_parser("verify", help="inspect an assembled runtime bundle")
  verify_parser.add_argument("--root", type=_path, required=True)
  verify_parser.add_argument("--legal-dir", type=_path, required=True)
  verify_parser.add_argument("--policy", type=_path)
  verify_parser.add_argument("--syft-json", type=_path)
  verify_parser.add_argument("--cyclonedx-json", type=_path)
  verify_parser.add_argument("--output", type=_path)

  source_parser = subparsers.add_parser("source-bundle", help="build LibRaw corresponding source archive")
  source_parser.add_argument("--manifest", type=_path, default=Path("compliance/sources.toml"))
  source_parser.add_argument("--output", type=_path, required=True)

  cyclonedx_parser = subparsers.add_parser("augment-cyclonedx", help="add declared native codec components to an SBOM")
  cyclonedx_parser.add_argument("--input", type=_path, required=True)
  cyclonedx_parser.add_argument("--output", type=_path, required=True)
  cyclonedx_parser.add_argument("--root", type=_path, required=True)
  cyclonedx_parser.add_argument("--legal-dir", type=_path, required=True)
  cyclonedx_parser.add_argument("--policy", type=_path)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  display_report: dict[str, object]
  try:
    if args.command == "collect":
      report = collect_legal_materials(
        args.root,
        args.output,
        args.policy,
        args.common_notices,
        args.include_python_runtime,
      )
      display_report = {"components": len(report["components"]), "output": args.output.as_posix()}
    elif args.command == "verify":
      report = verify_bundle(args.root, args.legal_dir, args.policy, args.syft_json, args.cyclonedx_json)
      if args.output:
        _json_dump(args.output, report)
      display_report = {
        "detections": {name: len(paths) for name, paths in report["detections"].items()},
        "output": args.output.as_posix() if args.output else None,
      }
    elif args.command == "source-bundle":
      build_corresponding_source(args.manifest, args.output)
      display_report = {"source_bundle": args.output.as_posix()}
    elif args.command == "augment-cyclonedx":
      report = augment_cyclonedx(args.input, args.output, args.root, args.legal_dir, args.policy)
      display_report = {"components": len(report["components"]), "output": args.output.as_posix()}
    else:  # pragma: no cover - argparse enforces this
      raise AssertionError(args.command)
  except (
    ComplianceError,
    json.JSONDecodeError,
    OSError,
    subprocess.CalledProcessError,
    tomllib.TOMLDecodeError,
  ) as error:
    print(f"compliance error: {error}", file=sys.stderr)
    return 1
  print(json.dumps(display_report, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
