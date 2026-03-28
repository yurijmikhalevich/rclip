import hashlib
import importlib.metadata
from collections import OrderedDict
from time import sleep
import jinja2
from packaging.requirements import Requirement
import requests
import sys

env = jinja2.Environment(trim_blocks=True)


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

  if OS.mac?
    if Hardware::CPU.arm?
      resource "rawpy" do
        url "{{ rawpy_wheels.mac_arm.url }}", using: :nounzip
        sha256 "{{ rawpy_wheels.mac_arm.sha256 }}"
      end
    else
      raise "Unknown CPU architecture, only arm64 is supported on macOS"
    end
  elsif OS.linux?
    if Hardware::CPU.arm?
      resource "rawpy" do
        url "{{ rawpy_wheels.linux_arm.url }}", using: :nounzip
        sha256 "{{ rawpy_wheels.linux_arm.sha256 }}"
      end
    elsif Hardware::CPU.intel?
      resource "rawpy" do
        url "{{ rawpy_wheels.linux_x86.url }}", using: :nounzip
        sha256 "{{ rawpy_wheels.linux_x86.sha256 }}"
      end
    else
      raise "Unknown CPU architecture, only amd64 and arm64 are supported"
    end
  end

  if OS.mac?
    if Hardware::CPU.arm?
      resource "hf-xet" do
        url "{{ hf_xet_wheels.mac_arm.url }}", using: :nounzip
        sha256 "{{ hf_xet_wheels.mac_arm.sha256 }}"
      end
    elsif Hardware::CPU.intel?
      resource "hf-xet" do
        url "{{ hf_xet_wheels.mac_intel.url }}", using: :nounzip
        sha256 "{{ hf_xet_wheels.mac_intel.sha256 }}"
      end
    else
      raise "Unknown CPU architecture, only amd64 and arm64 are supported"
    end
  elsif OS.linux?
    if Hardware::CPU.arm?
      resource "hf-xet" do
        url "{{ hf_xet_wheels.linux_arm.url }}", using: :nounzip
        sha256 "{{ hf_xet_wheels.linux_arm.sha256 }}"
      end
    elsif Hardware::CPU.intel?
      resource "hf-xet" do
        url "{{ hf_xet_wheels.linux_x86.url }}", using: :nounzip
        sha256 "{{ hf_xet_wheels.linux_x86.sha256 }}"
      end
    else
      raise "Unknown CPU architecture, only amd64 and arm64 are supported"
    end
  end

  def install
    # Fix for ZIP timestamp issue with files having dates before 1980
    ENV["SOURCE_DATE_EPOCH"] = "315532800" # 1980-01-01

    virtualenv_install_with_resources without: %w[hf-xet rawpy]

    resource("rawpy").stage do
      wheel = Dir["*.whl"].first
      valid_wheel = wheel.sub(/^.*--/, "")
      File.rename(wheel, valid_wheel)
      system "python3.14", "-m", "pip", "--python=#{libexec}/bin/python", "install", "--no-deps", valid_wheel
    end


    if OS.linux?
      rawpy_so = Dir[libexec/"lib/python3.14/site-packages/rawpy/_rawpy*.so"].first
      raise "rawpy shared object not found" unless rawpy_so

      system "patchelf", "--set-rpath", "$ORIGIN/../rawpy.libs", rawpy_so

      Dir[libexec/"lib/python3.14/site-packages/rawpy.libs/*.so*"].each do |lib|
        next if File.symlink?(lib)

        system "patchelf", "--set-rpath", "$ORIGIN", lib
      end
    end

    resource("hf-xet").stage do
      wheel = Dir["*.whl"].first
      valid_wheel = wheel.sub(/^.*--/, "")
      File.rename(wheel, valid_wheel)
      system "python3.14", "-m", "pip", "--python=#{libexec}/bin/python", "install", "--no-deps", valid_wheel
    end

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

# These deps are handled separately (brew-managed or platform-specific wheels)
WHEEL_DEPS = ["numpy", "pillow", "certifi", "rawpy", "torch", "torchvision", "hf-xet"]
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
    resp = requests.get(f"https://pypi.org/pypi/{actual_name}/{version}/json")
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


def get_wheels(package_name: str, tag: str = None):
  """Fetch platform-specific wheel URLs/SHA256 from PyPI, keyed by mac_arm/mac_intel/linux_arm/linux_x86."""
  try:
    dist = importlib.metadata.distribution(package_name)
    version = dist.metadata["Version"]
  except importlib.metadata.PackageNotFoundError:
    resp = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    resp.raise_for_status()
    version = resp.json()["info"]["version"]

  resp = requests.get(f"https://pypi.org/pypi/{package_name}/{version}/json")
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
    elif "macosx" in plat and "x86_64" in plat:
      result["mac_intel"] = info
    elif "linux" in plat and "aarch64" in plat:
      result["linux_arm"] = info
    elif "linux" in plat and "x86_64" in plat:
      result["linux_x86"] = info
  return result


def main():
  if len(sys.argv) != 2:
    print("Usage: generate_formula.py <version>")
    sys.exit(1)

  target_version = sys.argv[1]
  deps = get_deps_for_requested_rclip_version_or_die(target_version)

  for dep in WHEEL_DEPS:
    deps.pop(dep, None)
  for dep, url in RESOURCE_URL_OVERRIDES.items():
    new_url = url.render(version=deps[dep]["version"])
    deps[dep]["url"] = new_url
    deps[dep]["checksum"] = compute_checksum(new_url)
  for _, dep in deps.items():
    dep["name"] = dep["name"].lower().replace("_", "-")

  rclip_metadata = deps.pop("rclip")
  rawpy_wheels = get_wheels("rawpy", tag="cp314")
  hf_xet_wheels = get_wheels("hf-xet", tag="abi3")
  resources = "\n\n".join([RESOURCE_TEMPLATE.render(resource=dep) for dep in deps.values()])
  print(TEMPLATE.render(package=rclip_metadata, resources=resources, rawpy_wheels=rawpy_wheels, hf_xet_wheels=hf_xet_wheels))


def compute_checksum(url: str):
  response = requests.get(url)
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
