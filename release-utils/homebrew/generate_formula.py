import hashlib
import importlib.metadata
from collections import OrderedDict
from collections.abc import Mapping
from time import sleep
from typing import AbstractSet, Optional, TypedDict, cast

import jinja2
from packaging.markers import default_environment
from packaging.requirements import Requirement
import requests
import sys

env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

REQUEST_TIMEOUT = 60  # seconds
TARGET_PYTHON_VERSION = "3.13"
TARGET_PYTHON_FULL_VERSION = f"{TARGET_PYTHON_VERSION}.0"
TARGET_PYTHON_TAG = f"cp{TARGET_PYTHON_VERSION.replace('.', '')}"


TEMPLATE = env.from_string("""class Rclip < Formula
  include Language::Python::Virtualenv

  desc "AI-Powered Command-Line Photo Search Tool"
  homepage "https://github.com/yurijmikhalevich/rclip"
  url "{{ package.url }}"
  sha256 "{{ package.checksum }}"
  license "MIT"

  if OS.linux?
    depends_on "patchelf" => :build # for rawpy
    depends_on "zlib-ng-compat" # rawpy bundled libs link against libz
  end
  depends_on "certifi"
  depends_on "libheif"
  depends_on "libraw"
  depends_on "libyaml"
  depends_on "numpy"
  depends_on "pillow"
  depends_on "python@{{ target_python_version }}"

{{ resources }}

{{ wheel_resources }}

  def install
    # Fix for ZIP timestamp issue with files having dates before 1980
    ENV["SOURCE_DATE_EPOCH"] = "315532800" # 1980-01-01

    virtualenv_install_with_resources without: %w[{{ wheel_names }}]
{% for pkg in wheel_packages %}

    resource("{{ pkg.name }}").stage do
      wheel = Dir["*.whl"].first
      valid_wheel = wheel.sub(/^.*--/, "")
      File.rename(wheel, valid_wheel)
      system "python{{ target_python_version }}", "-m", "pip", "--python=#{libexec}/bin/python", "install", "--no-deps", valid_wheel
    end
{% if pkg.patchelf %}

    if OS.linux?
{% for path, rpath in pkg.patchelf.items() %}
      targets = Dir[libexec/"lib/python{{ target_python_version }}/site-packages/{{ path }}"]
      if targets.empty?
        odie "Failed to find any files to patch with patchelf for pattern: " \\
             "#{libexec}/lib/python{{ target_python_version }}/site-packages/{{ path }}"
      end
      targets.each do |so|
        next if File.symlink?(so)

        system "patchelf", "--set-rpath", "{{ rpath }}", so
      end
{% endfor %}
    end
{% endif %}
{% endfor %}

  end

  test do
    output = shell_output("#{bin}/rclip cat")
    assert_match("score\\tfilepath", output)
  end
end
""")  # noqa


RESOURCE_TEMPLATE = env.from_string(
  '  resource "{{ resource.name }}" do\n'
  '    url "{{ resource.url }}"\n'
  '    {{ resource.checksum_type }} "{{ resource.checksum }}"\n'
  "  end"
)

# These deps are handled by Homebrew formulas (excluded from virtualenv resources)
BREW_DEPS = ["numpy", "pillow", "certifi"]


class WheelInfo(TypedDict):
  url: str
  sha256: str


class PlatformWheels(TypedDict):
  mac_arm: WheelInfo
  linux_arm: WheelInfo
  linux_x86: WheelInfo


class PackageResource(TypedDict):
  name: str
  version: str
  url: str
  checksum: str
  checksum_type: str
  homepage: str


class _WheelPackageRequired(TypedDict):
  name: str
  tag: str


class WheelPackage(_WheelPackageRequired, total=False):
  patchelf: dict[str, str]


# These deps are installed from platform-specific wheels (excluded from virtualenv resources)
WHEEL_PACKAGES: list[WheelPackage] = [
  {
    "name": "rawpy",
    "tag": TARGET_PYTHON_TAG,
    "patchelf": {
      "rawpy/_rawpy*.so": "$ORIGIN/../rawpy.libs",
      "rawpy.libs/*.so*": "$ORIGIN",
    },
  },
  {"name": "hf-xet", "tag": "abi3"},
  {"name": "onnxruntime", "tag": TARGET_PYTHON_TAG},
]

RESOURCE_URL_OVERRIDES = {}

MAKE_GRAPH_IGNORED = {"pip", "setuptools", "wheel", "argparse", "wsgiref"}

EXTRA_MACOS_RESOURCES = ["coremltools"]
EXTRA_MACOS_RESOURCE_KEYS = {name.lower().replace("-", "_") for name in EXTRA_MACOS_RESOURCES}

MarkerEnvironment = dict[str, str | AbstractSet[str]]


def get_marker_environment(overrides: Mapping[str, str]) -> MarkerEnvironment:
  environment: MarkerEnvironment = {}
  for key, value in default_environment().items():
    if isinstance(value, str):
      environment[key] = value
  environment.update(
    {
      "python_version": TARGET_PYTHON_VERSION,
      "python_full_version": TARGET_PYTHON_FULL_VERSION,
      "platform_python_implementation": "CPython",
      "implementation_name": "cpython",
      "implementation_version": TARGET_PYTHON_FULL_VERSION,
    }
  )
  environment.update(overrides)
  return environment


def make_graph(package_name: str, skip_pypi_packages: set[str]):
  marker_env = get_marker_environment({"sys_platform": "linux", "platform_system": "Linux"})
  result = OrderedDict()
  queue = [package_name]
  visited = set()
  while queue:
    pkg = queue.pop(0)
    key = pkg.lower().replace("-", "_")
    if key in visited:
      continue
    visited.add(key)
    if key in EXTRA_MACOS_RESOURCE_KEYS:
      continue
    try:
      dist = importlib.metadata.distribution(pkg)
    except importlib.metadata.PackageNotFoundError:
      continue
    actual_name = dist.metadata["Name"]
    version = dist.metadata["Version"]
    if actual_name.lower() in MAKE_GRAPH_IGNORED:
      continue
    if actual_name.lower() in skip_pypi_packages:
      result[actual_name.lower().replace("_", "-")] = {
        "name": actual_name,
        "version": version,
      }
    else:
      resp = requests.get(f"https://pypi.org/pypi/{actual_name}/{version}/json", timeout=REQUEST_TIMEOUT)
      resp.raise_for_status()
      data = resp.json()
      sdist = next((url_entry for url_entry in data["urls"] if url_entry["packagetype"] == "sdist"), None)
      url_info = sdist or next(iter(data["urls"]), None)
      result[actual_name.lower().replace("_", "-")] = {
        "name": actual_name,
        "version": version,
        "url": url_info["url"] if url_info else "",
        "checksum": url_info["digests"]["sha256"] if url_info else "",
        "checksum_type": "sha256",
        "homepage": data["info"]["home_page"] or "",
      }
    for req_str in dist.requires or []:
      req = Requirement(req_str)
      if "extra" not in str(req.marker or "") and (req.marker is None or req.marker.evaluate(marker_env)):
        queue.append(req.name)
  return result


def get_pypi_resource(package_name: str, version: str) -> PackageResource:
  resp = requests.get(f"https://pypi.org/pypi/{package_name}/{version}/json", timeout=REQUEST_TIMEOUT)
  resp.raise_for_status()
  data = resp.json()
  sdist = next((url_entry for url_entry in data["urls"] if url_entry["packagetype"] == "sdist"), None)
  url_info = sdist or next(iter(data["urls"]), None)
  if url_info is None:
    raise RuntimeError(f"No downloadable files found on PyPI for {package_name!r} {version}")
  return {
    "name": package_name,
    "version": version,
    "url": url_info["url"],
    "checksum": url_info["digests"]["sha256"],
    "checksum_type": "sha256",
    "homepage": data["info"]["home_page"] or "",
  }


def get_macos_only_resources() -> OrderedDict[str, PackageResource]:
  marker_env = get_marker_environment({"sys_platform": "darwin", "platform_system": "Darwin"})
  resources: dict[str, PackageResource] = {}
  queue = list(EXTRA_MACOS_RESOURCES)
  visited = set()
  while queue:
    package_name = queue.pop(0)
    key = package_name.lower().replace("-", "_")
    if key in visited:
      continue
    visited.add(key)

    dist = importlib.metadata.distribution(package_name)
    actual_name = dist.metadata["Name"]
    normalized_name = actual_name.lower().replace("_", "-")
    resources[normalized_name] = get_pypi_resource(actual_name, dist.metadata["Version"])

    for req_str in dist.requires or []:
      req = Requirement(req_str)
      if req.name.lower() in BREW_DEPS:
        continue
      if "extra" in str(req.marker or ""):
        continue
      if req.marker is not None and not req.marker.evaluate(marker_env):
        continue
      queue.append(req.name)

  return OrderedDict(sorted(resources.items()))


def get_wheels(package_name: str, tag: Optional[str] = None, resolved_version: Optional[str] = None) -> PlatformWheels:
  """Fetch platform-specific wheel URLs/SHA256 from PyPI, keyed by mac_arm/linux_arm/linux_x86."""
  if resolved_version is not None:
    version = resolved_version
  else:
    try:
      dist = importlib.metadata.distribution(package_name)
      version = dist.metadata["Version"]
    except importlib.metadata.PackageNotFoundError as exc:
      raise RuntimeError(
        f"Package {package_name!r} is not installed and no resolved version was provided; "
        f"install rclip and its dependencies before running the formula generator."
      ) from exc

  resp = requests.get(f"https://pypi.org/pypi/{package_name}/{version}/json", timeout=REQUEST_TIMEOUT)
  resp.raise_for_status()
  data = resp.json()

  result = {}
  for url_info in data["urls"]:
    if url_info["packagetype"] != "bdist_wheel":
      continue
    parts = url_info["filename"][:-4].split("-")  # strip .whl
    if len(parts) < 5:
      continue
    py, abi, plat = parts[-3], parts[-2], parts[-1]
    py_tags = py.split(".")
    abi_tags = abi.split(".")
    if tag == "abi3":
      if "abi3" not in abi_tags:
        continue
    elif tag:
      if tag not in py_tags:
        continue
      if tag not in abi_tags and "abi3" not in abi_tags:
        continue
    info = {"url": url_info["url"], "sha256": url_info["digests"]["sha256"]}
    if "macosx" in plat and "arm64" in plat:
      result["mac_arm"] = info
    elif "manylinux" in plat and "aarch64" in plat:
      result["linux_arm"] = info
    elif "manylinux" in plat and "x86_64" in plat:
      result["linux_x86"] = info
  missing = {"mac_arm", "linux_arm", "linux_x86"} - result.keys()
  if missing:
    raise RuntimeError(f"Missing {package_name!r} wheels for platforms: {missing}")
  return cast(PlatformWheels, result)


def render_macos_resource_blocks(resources: list[PackageResource]) -> str:
  if not resources:
    return ""
  lines = ["  if OS.mac?"]
  for resource in resources:
    lines.extend(RESOURCE_TEMPLATE.render(resource=resource).splitlines())
    lines.append("")
  if lines[-1] == "":
    lines.pop()
  lines.append("  end")
  return "\n".join(lines)


def render_wheel_resource_block(name: str, wheels: PlatformWheels) -> str:
  """Generate Ruby conditional block for platform-specific wheel resources."""

  def resource_block(wheel_info: WheelInfo, indent: str) -> list[str]:
    return [
      f'{indent}resource "{name}" do',
      f'{indent}  url "{wheel_info["url"]}", using: :nounzip',
      f'{indent}  sha256 "{wheel_info["sha256"]}"',
      f"{indent}end",
    ]

  lines: list[str] = []
  lines.append("  if OS.mac?")
  lines.append("    if Hardware::CPU.arm?")
  lines.extend(resource_block(wheels["mac_arm"], "      "))
  lines.append("    else")
  lines.append('      raise "Unknown CPU architecture, only arm64 is supported on macOS"')
  lines.append("    end")
  lines.append("  elsif OS.linux?")
  lines.append("    if Hardware::CPU.arm?")
  lines.extend(resource_block(wheels["linux_arm"], "      "))
  lines.append("    elsif Hardware::CPU.intel?")
  lines.extend(resource_block(wheels["linux_x86"], "      "))
  lines.append("    else")
  lines.append('      raise "Unknown CPU architecture, only amd64 and arm64 are supported"')
  lines.append("    end")
  lines.append("  end")

  return "\n".join(lines)


def main():
  if len(sys.argv) != 2:
    print("Usage: generate_formula.py <version>")
    sys.exit(1)

  target_version = sys.argv[1]
  skip_pypi_packages = {dep.lower() for dep in BREW_DEPS + [pkg["name"] for pkg in WHEEL_PACKAGES]}
  deps = get_deps_for_requested_rclip_version_or_die(target_version, skip_pypi_packages)
  macos_only_resources = get_macos_only_resources()

  wheel_versions = {
    pkg["name"]: deps[pkg["name"].lower()]["version"] for pkg in WHEEL_PACKAGES if pkg["name"].lower() in deps
  }

  excluded_deps = BREW_DEPS + [pkg["name"] for pkg in WHEEL_PACKAGES]
  for dep in excluded_deps:
    deps.pop(dep, None)
  for dep, url in RESOURCE_URL_OVERRIDES.items():
    new_url = url.render(version=deps[dep]["version"])
    deps[dep]["url"] = new_url
    deps[dep]["checksum"] = compute_checksum(new_url)
  for dep in deps.values():
    dep["name"] = dep["name"].lower().replace("_", "-")
  for dep in macos_only_resources.values():
    dep["name"] = dep["name"].lower().replace("_", "-")
  for dep in list(macos_only_resources):
    if dep in deps or dep in skip_pypi_packages:
      macos_only_resources.pop(dep)

  rclip_metadata = deps.pop("rclip")

  all_wheels = []
  for pkg in WHEEL_PACKAGES:
    wheels = get_wheels(pkg["name"], tag=pkg.get("tag"), resolved_version=wheel_versions.get(pkg["name"]))
    all_wheels.append(wheels)

  wheel_resources = "\n\n".join(
    render_wheel_resource_block(pkg["name"], wheels) for pkg, wheels in zip(WHEEL_PACKAGES, all_wheels)
  )
  wheel_names = " ".join(pkg["name"] for pkg in WHEEL_PACKAGES)
  resources = "\n\n".join(
    [
      *[RESOURCE_TEMPLATE.render(resource=dep) for dep in deps.values()],
      *([render_macos_resource_blocks(list(macos_only_resources.values()))] if macos_only_resources else []),
    ]
  )
  print(
    TEMPLATE.render(
      package=rclip_metadata,
      resources=resources,
      target_python_version=TARGET_PYTHON_VERSION,
      wheel_resources=wheel_resources,
      wheel_names=wheel_names,
      wheel_packages=WHEEL_PACKAGES,
    )
  )


def compute_checksum(url: str):
  with requests.get(url, timeout=REQUEST_TIMEOUT, stream=True) as response:
    response.raise_for_status()
    sha256 = hashlib.sha256()
    for chunk in response.iter_content(chunk_size=8192):
      if chunk:
        sha256.update(chunk)
    return sha256.hexdigest()


def get_deps_for_requested_rclip_version_or_die(target_version: str, skip_pypi_packages: set[str]):
  deps = make_graph("rclip", skip_pypi_packages)
  rclip_metadata = deps["rclip"]
  target_tarball = f"rclip-{target_version}.tar.gz"

  # it takes a few seconds for a published wheel appear in PyPI
  retries_left = 5
  while not rclip_metadata["url"].endswith(target_tarball):
    if retries_left == 0:
      print(f"Version mismatch: {rclip_metadata['version']} != {target_version}. Exiting.", file=sys.stderr)
      sys.exit(1)
    retries_left -= 1
    print(
      f"Version mismatch: {rclip_metadata['url'].split('/')[-1]} != {target_tarball}. Retrying in 10 seconds.",
      file=sys.stderr,
    )
    # it takes a few seconds for a published wheel appear in PyPI
    sleep(10)
    deps = make_graph("rclip", skip_pypi_packages)
    rclip_metadata = deps["rclip"]

  return deps


if __name__ == "__main__":
  main()
