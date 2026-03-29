import hashlib
import importlib.metadata
from collections import OrderedDict
from time import sleep
from typing import Optional, TypedDict
import jinja2
from packaging.requirements import Requirement
import requests
import sys

env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

REQUEST_TIMEOUT = 60  # seconds


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
  depends_on "rust" => :build # for safetensors
  depends_on "certifi"
  depends_on "libheif"
  depends_on "libraw"
  depends_on "libyaml"
  depends_on "numpy"
  depends_on "pillow"
  depends_on "python@3.14"
  depends_on "pytorch"
  depends_on "sentencepiece"
  depends_on "torchvision"

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
      system "python3.14", "-m", "pip", "--python=#{libexec}/bin/python", "install", "--no-deps", valid_wheel
    end
{% if pkg.patchelf %}

    if OS.linux?
{% for path, rpath in pkg.patchelf.items() %}
      targets = Dir[libexec/"lib/python3.14/site-packages/{{ path }}"]
      odie "Failed to find any files to patch with patchelf for pattern: #{libexec}/lib/python3.14/site-packages/{{ path }}" if targets.empty?
      targets.each do |so|
        next if File.symlink?(so)
        system "patchelf", "--set-rpath", "{{ rpath }}", so
      end
{% endfor %}
    end
{% endif %}
{% endfor %}

    # link dependent virtualenvs to this one
    site_packages = Language::Python.site_packages("python3.14")
    paths = %w[pytorch torchvision].map do |package_name|
      package = Formula[package_name].opt_libexec
      package/site_packages
    end
    (libexec/site_packages/"homebrew-deps.pth").write paths.join("\\n")
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
BREW_DEPS = ["numpy", "pillow", "certifi", "torch", "torchvision"]


class _WheelPackageRequired(TypedDict):
  name: str
  tag: str


class WheelPackage(_WheelPackageRequired, total=False):
  patchelf: dict[str, str]


# These deps are installed from platform-specific wheels (excluded from virtualenv resources)
WHEEL_PACKAGES: list[WheelPackage] = [
  {
    "name": "rawpy",
    "tag": "cp314",
    "patchelf": {
      "rawpy/_rawpy*.so": "$ORIGIN/../rawpy.libs",
      "rawpy.libs/*.so*": "$ORIGIN",
    },
  },
  {"name": "hf-xet", "tag": "abi3"},
]

RESOURCE_URL_OVERRIDES = {
  # open-clip-torch publishes an incomplete tarball to pypi, so we will fetch one from GitHub
  "open-clip-torch": env.from_string(
    "https://github.com/mlfoundations/open_clip/archive/refs/tags/v{{ version }}.tar.gz"
  ),
}

_MAKE_GRAPH_IGNORED = {"pip", "setuptools", "wheel", "argparse", "wsgiref"}


def make_graph(package_name: str):
  result = OrderedDict()
  queue = [package_name]
  visited = set()
  while queue:
    pkg = queue.pop(0)
    key = pkg.lower().replace("-", "_")
    if key in visited:
      continue
    visited.add(key)
    try:
      dist = importlib.metadata.distribution(pkg)
    except importlib.metadata.PackageNotFoundError:
      continue
    actual_name = dist.metadata["Name"]
    version = dist.metadata["Version"]
    if actual_name.lower() in _MAKE_GRAPH_IGNORED:
      continue
    resp = requests.get(f"https://pypi.org/pypi/{actual_name}/{version}/json", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    sdist = next((u for u in data["urls"] if u["packagetype"] == "sdist"), None)
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
      if "extra" not in str(req.marker or "") and (req.marker is None or req.marker.evaluate({})):
        queue.append(req.name)
  return result


def get_wheels(package_name: str, tag: Optional[str] = None, resolved_version: Optional[str] = None):
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
    if tag and tag not in f"{py}-{abi}":
      continue
    info = {"url": url_info["url"], "sha256": url_info["digests"]["sha256"]}
    if "macosx" in plat and "arm64" in plat:
      result["mac_arm"] = info
    elif "linux" in plat and "aarch64" in plat:
      result["linux_arm"] = info
    elif "linux" in plat and "x86_64" in plat:
      result["linux_x86"] = info
  return result


def render_wheel_resource_block(name: str, wheels: dict[str, dict[str, str]]) -> str:
  """Generate Ruby conditional block for platform-specific wheel resources."""
  mac_arm = wheels.get("mac_arm")
  linux_arm = wheels.get("linux_arm")
  linux_x86 = wheels.get("linux_x86")

  has_mac = bool(mac_arm)
  has_linux = bool(linux_arm or linux_x86)

  if not has_mac and not has_linux:
    return ""

  def resource_block(wheel_info: dict[str, str], indent: str) -> list[str]:
    return [
      f'{indent}resource "{name}" do',
      f'{indent}  url "{wheel_info["url"]}", using: :nounzip',
      f'{indent}  sha256 "{wheel_info["sha256"]}"',
      f"{indent}end",
    ]

  lines = []

  if has_mac:
    lines.append("  if OS.mac?")
    first = True
    if mac_arm:
      lines.append("    if Hardware::CPU.arm?")
      lines.extend(resource_block(mac_arm, "      "))
      first = False
    lines.append("    else")
    lines.append('      raise "Unknown CPU architecture, only arm64 is supported on macOS"')
    lines.append("    end")

  if has_linux:
    os_keyword = "  elsif" if has_mac else "  if"
    lines.append(f"{os_keyword} OS.linux?")
    first = True
    if linux_arm:
      lines.append("    if Hardware::CPU.arm?")
      lines.extend(resource_block(linux_arm, "      "))
      first = False
    if linux_x86:
      cpu_keyword = "    elsif" if not first else "    if"
      lines.append(f"{cpu_keyword} Hardware::CPU.intel?")
      lines.extend(resource_block(linux_x86, "      "))
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
  deps = get_deps_for_requested_rclip_version_or_die(target_version)

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
  for _, dep in deps.items():
    dep["name"] = dep["name"].lower().replace("_", "-")

  rclip_metadata = deps.pop("rclip")

  all_wheels = [
    get_wheels(pkg["name"], tag=pkg.get("tag"), resolved_version=wheel_versions.get(pkg["name"]))
    for pkg in WHEEL_PACKAGES
  ]
  for pkg, wheels in zip(WHEEL_PACKAGES, all_wheels):
    if not wheels:
      print(f"No wheels found for {pkg['name']!r} (tag={pkg.get('tag')!r}). Exiting.", file=sys.stderr)
      sys.exit(1)
  wheel_packages = list(WHEEL_PACKAGES)

  wheel_resources = "\n\n".join(
    [render_wheel_resource_block(pkg["name"], wheels) for pkg, wheels in zip(WHEEL_PACKAGES, all_wheels)]
  )
  wheel_names = " ".join(pkg["name"] for pkg in WHEEL_PACKAGES)
  resources = "\n\n".join([RESOURCE_TEMPLATE.render(resource=dep) for dep in deps.values()])
  print(
    TEMPLATE.render(
      package=rclip_metadata,
      resources=resources,
      wheel_resources=wheel_resources,
      wheel_names=wheel_names,
      wheel_packages=wheel_packages,
    )
  )


def compute_checksum(url: str):
  response = requests.get(url, timeout=REQUEST_TIMEOUT)
  return hashlib.sha256(response.content).hexdigest()


def get_deps_for_requested_rclip_version_or_die(target_version: str):
  deps = make_graph("rclip")
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
    deps = make_graph("rclip")
    rclip_metadata = deps["rclip"]

  return deps


if __name__ == "__main__":
  main()
