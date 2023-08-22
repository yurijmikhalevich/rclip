import hashlib
from time import sleep
import jinja2
import poet
import requests
import sys

env = jinja2.Environment(trim_blocks=True)


TEMPLATE = env.from_string('''class Rclip < Formula
  include Language::Python::Virtualenv

  desc "AI-Powered Command-Line Photo Search Tool"
  homepage "https://github.com/yurijmikhalevich/rclip"
  url "{{ package.url }}"
  sha256 "{{ package.checksum }}"
  license "MIT"

  depends_on "rust" => :build # for safetensors
  depends_on "numpy"
  depends_on "pillow"
  depends_on "python-certifi"
  depends_on "python@3.11"
  depends_on "pytorch"
  depends_on "sentencepiece"
  depends_on "torchvision"

{{ resources }}

  def install
    virtualenv_install_with_resources

    # link dependent virtualenvs to this one
    site_packages = Language::Python.site_packages("python3.11")
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
''')


# These deps are being installed from brew
DEPS_TO_IGNORE = ['numpy', 'pillow', 'certifi', 'torch', 'torchvision']
RESOURCE_URL_OVERRIDES = {
  # open-clip-torch publishes an incomplete tarball to pypi, so we will fetch one from GitHub
  'open-clip-torch': env.from_string(
    'https://github.com/mlfoundations/open_clip/archive/refs/tags/v{{ version }}.tar.gz'
  ),
}


def main():
  if len(sys.argv) != 2:
    print('Usage: generate_formula.py <version>')
    sys.exit(1)

  target_version = sys.argv[1]
  deps = get_deps_for_requested_rclip_version_or_die(target_version)

  for dep in DEPS_TO_IGNORE:
    deps.pop(dep, None)
  for dep, url in RESOURCE_URL_OVERRIDES.items():
    new_url = url.render(version=deps[dep]['version'])
    deps[dep]['url'] = new_url
    deps[dep]['checksum'] = compute_checksum(new_url)
  for _, dep in deps.items():
    dep['name'] = dep['name'].lower()

  rclip_metadata = deps.pop('rclip')
  resources = '\n\n'.join([poet.RESOURCE_TEMPLATE.render(resource=dep) for dep in deps.values()])
  print(TEMPLATE.render(package=rclip_metadata, resources=resources))


def compute_checksum(url: str):
  response = requests.get(url)
  return hashlib.sha256(response.content).hexdigest()


def get_deps_for_requested_rclip_version_or_die(target_version: str):
  deps = poet.make_graph('rclip')
  rclip_metadata = deps['rclip']
  retries_left = 5
  while rclip_metadata['version'] != target_version:
    if retries_left == 0:
      print(f'Version mismatch: {rclip_metadata["version"]} != {target_version}. Exiting.', file=sys.stderr)
      sys.exit(1)
    retries_left -= 1
    print(
      f'Version mismatch: {rclip_metadata["version"]} != {target_version}. Retrying in 10 seconds.',
      file=sys.stderr,
    )
    # it takes a few seconds for a published wheel appear in PyPI
    sleep(10)
    deps = poet.make_graph('rclip')
    rclip_metadata = deps['rclip']
  return deps


if __name__ == '__main__':
  main()
