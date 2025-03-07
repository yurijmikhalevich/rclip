import hashlib
from time import sleep
import jinja2
import poet
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
  end

  depends_on "rust" => :build # for safetensors
  depends_on "certifi"
  depends_on "libyaml"
  depends_on "numpy"
  depends_on "pillow"
  depends_on "python@3.12"
  depends_on "pytorch-python312@2.5.1"
  depends_on "sentencepiece"
  depends_on "torchvision-python312@0.20.1"

{{ resources }}

  if OS.mac?
    if Hardware::CPU.arm?
      resource "rawpy" do
        url "https://files.pythonhosted.org/packages/87/75/610a34caf048aa87248f8393e70073610146f379fdda8194a988ba286d5b/rawpy-0.24.0-cp312-cp312-macosx_11_0_arm64.whl", using: :nounzip
        sha256 "1097b10eed4027e5b50006548190602e1adba9c824526b45f7a37781cfa01818"
      end
    elsif Hardware::CPU.intel?
      resource "rawpy" do
        url "https://files.pythonhosted.org/packages/27/1c/59024e87c20b325e10b43e3b709929681a0ed23bda3885c7825927244fcc/rawpy-0.24.0-cp312-cp312-macosx_10_9_x86_64.whl", using: :nounzip
        sha256 "ed639b0dc91c3e85d6c39303a1523b7e1edc4f4b0381c376ed0ff99febb306e4"
      end
    else
      raise "Unknown CPU architecture, only amd64 and arm64 are supported"
    end
  elsif OS.linux?
    if Hardware::CPU.arm?
      resource "rawpy" do
        url "https://files.pythonhosted.org/packages/9c/c4/576853c0eea14d62a2776f683dae23c994572dfc2dcb47fd1a1473b7b18a/rawpy-0.24.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl", using: :nounzip
        sha256 "17a970fd8cdece57929d6e99ce64503f21b51c00ab132bad53065bd523154892"
      end
    elsif Hardware::CPU.intel?
      resource "rawpy" do
        url "https://files.pythonhosted.org/packages/fe/35/5d6765359ce6e06fe0aee5a3e4e731cfe08c056df093d97c292bdc02132a/rawpy-0.24.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", using: :nounzip
        sha256 "a12fc4e6c5879b88c6937abb9f3f6670dd34d126b4a770ad4566e9f747e306fb"
      end
    else
      raise "Unknown CPU architecture, only amd64 and arm64 are supported"
    end
  end

  def install
    # Fix for ZIP timestamp issue with files having dates before 1980
    ENV["SOURCE_DATE_EPOCH"] = "315532800" # 1980-01-01

    virtualenv_install_with_resources without: "rawpy"

    resource("rawpy").stage do
      wheel = Dir["*.whl"].first
      valid_wheel = wheel.sub(/^.*--/, "")
      File.rename(wheel, valid_wheel)
      system "python3.12", "-m", "pip", "--python=#{libexec}/bin/python", "install", "--no-deps", valid_wheel
    end

    if OS.linux?
      rawpy_so = Dir[libexec/"lib/python3.12/site-packages/rawpy/_rawpy*.so"].first
      raise "rawpy shared object not found" unless rawpy_so

      system "patchelf", "--set-rpath", "$ORIGIN/../rawpy.libs", rawpy_so

      libraw_so = Dir[libexec/"lib/python3.12/site-packages/rawpy.libs/libraw*.so.*"].first
      raise "libraw shared object not found" unless libraw_so

      system "patchelf", "--set-rpath", "$ORIGIN", libraw_so
    end

    # link dependent virtualenvs to this one
    site_packages = Language::Python.site_packages("python3.12")
    paths = %w[pytorch-python312@2.5.1 torchvision-python312@0.20.1].map do |package_name|
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


# These deps are being installed from brew
DEPS_TO_IGNORE = ["numpy", "pillow", "certifi", "rawpy", "torch", "torchvision"]
RESOURCE_URL_OVERRIDES = {
  # open-clip-torch publishes an incomplete tarball to pypi, so we will fetch one from GitHub
  "open-clip-torch": env.from_string(
    "https://github.com/mlfoundations/open_clip/archive/refs/tags/v{{ version }}.tar.gz"
  ),
}


def main():
  if len(sys.argv) != 2:
    print("Usage: generate_formula.py <version>")
    sys.exit(1)

  target_version = sys.argv[1]
  deps = get_deps_for_requested_rclip_version_or_die(target_version)

  for dep in DEPS_TO_IGNORE:
    deps.pop(dep, None)
  for dep, url in RESOURCE_URL_OVERRIDES.items():
    new_url = url.render(version=deps[dep]["version"])
    deps[dep]["url"] = new_url
    deps[dep]["checksum"] = compute_checksum(new_url)
  for _, dep in deps.items():
    dep["name"] = dep["name"].lower()

  rclip_metadata = deps.pop("rclip")
  resources = "\n\n".join([poet.RESOURCE_TEMPLATE.render(resource=dep) for dep in deps.values()])
  print(TEMPLATE.render(package=rclip_metadata, resources=resources))


def compute_checksum(url: str):
  response = requests.get(url)
  return hashlib.sha256(response.content).hexdigest()


def get_deps_for_requested_rclip_version_or_die(target_version: str):
  deps = poet.make_graph("rclip")
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
    deps = poet.make_graph("rclip")
    rclip_metadata = deps["rclip"]

  return deps


if __name__ == "__main__":
  main()
