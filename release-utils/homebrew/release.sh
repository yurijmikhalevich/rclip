#!/bin/bash
set -e

# the script requires gh cli and git to be installed and configured
# and push permissions to https://github.com/yurijmikhalevich/homebrew-tap

ORIG_PWD=$(pwd)
VERSION=$(poetry version -s)

TMP_DIR=$(mktemp -d -t release-rclip-brew-XXXXXXXXXX)
cd $TMP_DIR
echo "Working in $TMP_DIR"

function handle_exit() {
  cd "$ORIG_PWD"
  rm -rf $TMP_DIR
  echo "Removed $TMP_DIR"
}
trap handle_exit 0 SIGHUP SIGINT SIGQUIT SIGABRT SIGTERM

if [[ "$GITHUB_ACTIONS" ]]; then
  git clone "https://$GITHUB_TOKEN@github.com/yurijmikhalevich/homebrew-tap.git" homebrew-tap
else
  git clone git@github.com:yurijmikhalevich/homebrew-tap.git homebrew-tap
fi
cd homebrew-tap

PR_BRANCH="release-rclip-$VERSION"
PR_TITLE="rclip $VERSION"

git checkout -b "$PR_BRANCH"
python "$ORIG_PWD/release-utils/homebrew/generate_formula.py" ${VERSION} > Formula/rclip.rb
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
