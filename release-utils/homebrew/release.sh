#!/bin/bash
set -euo pipefail

# Updates the rclip formula in https://github.com/yurijmikhalevich/homebrew-tap to the version in
# Cargo.toml, whose binaries must already be published in a GitHub release.
# The script requires gh cli and git to be installed and configured,
# and push permissions to the tap.

ORIG_PWD=$(pwd)
VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)
RELEASE_URL="https://github.com/yurijmikhalevich/rclip/releases/download/v$VERSION"

TMP_DIR=$(mktemp -d -t release-rclip-brew-XXXXXXXXXX)
cd "$TMP_DIR"
echo "Working in $TMP_DIR"

function handle_exit() {
  cd "$ORIG_PWD"
  rm -rf "$TMP_DIR"
  echo "Removed $TMP_DIR"
}
trap handle_exit 0 SIGHUP SIGINT SIGQUIT SIGABRT SIGTERM

# Prints the URL and the checksum of the release archive for a platform.
function archive() {
  local name="rclip-$VERSION-$1.tar.gz"
  curl -fsSL -o "$name" "$RELEASE_URL/$name"
  echo "url \"$RELEASE_URL/$name\""
  echo "      sha256 \"$(shasum -a 256 "$name" | cut -d " " -f 1)\""
}

cat > rclip.rb <<FORMULA
class Rclip < Formula
  desc "Semantic photo search for the command-line"
  homepage "https://github.com/yurijmikhalevich/rclip"
  version "$VERSION"
  license "MIT"

  on_macos do
    depends_on arch: :arm64

    on_arm do
      $(archive aarch64-apple-darwin)
    end
  end

  on_linux do
    on_intel do
      $(archive x86_64-unknown-linux-gnu)
    end

    on_arm do
      $(archive aarch64-unknown-linux-gnu)
    end
  end

  def install
    bin.install "rclip"
    doc.install "LICENSE", "THIRD_PARTY_NOTICES.txt"
  end

  test do
    assert_match "rclip #{version}", shell_output("#{bin}/rclip --version")
  end
end
FORMULA

if [[ "${GITHUB_ACTIONS:-}" ]]; then
  # gh prefers GH_TOKEN over GITHUB_TOKEN, which holds the bot token with access to the tap
  if [[ "${GH_TOKEN:-}" ]]; then
    echo "GH_TOKEN must not be set, gh must use GITHUB_TOKEN"
    exit 1
  fi
  gh auth setup-git
  git clone https://github.com/yurijmikhalevich/homebrew-tap.git homebrew-tap
else
  git clone git@github.com:yurijmikhalevich/homebrew-tap.git homebrew-tap
fi
cd homebrew-tap

PR_BRANCH="release-rclip-$VERSION"
PR_TITLE="rclip $VERSION"

git checkout -b "$PR_BRANCH"
cp ../rclip.rb Formula/rclip.rb
git commit -am "$PR_TITLE"
git push origin "$PR_BRANCH"
gh pr create --title "$PR_TITLE" --body "Automated commit updating **rclip** formula to $VERSION" --base main --head "$PR_BRANCH"
# it takes a few seconds for GHA to start checks on the PR
sleep 20
gh pr checks "$PR_BRANCH" --watch --fail-fast
gh pr edit "$PR_BRANCH" --add-label pr-pull
# it takes a few seconds for GHA to start checks on the PR
sleep 20
gh pr checks "$PR_BRANCH" --watch --fail-fast

# assert that PR_STATE was closed as it should
PR_STATE=$(gh pr view "$PR_BRANCH" --json state -q .state)
if [ "$PR_STATE" != "CLOSED" ]; then
  echo "PR \"$PR_TITLE\" is not closed"
  exit 1
fi

echo "Released rclip $VERSION"
