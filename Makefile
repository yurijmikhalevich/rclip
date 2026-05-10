# the `release` action uses `sed` expressions which are only compatible with the GNU sed
ifeq ($(shell which gsed),)
    SED := sed
else
    SED := gsed
endif

build-appimage:
	appimage-builder --recipe ./release-utils/appimage/appimage-builder.yml

lint-style:
	uv run ruff check

fix-style:
	uv run ruff check --fix
	uv run ruff format

lint-types:
	uv run ty check

lint: lint-style lint-types

test:
	uv run pytest tests

test-system-rclip:
	RCLIP_TEST_RUN_SYSTEM_RCLIP=true uv run --no-sync pytest tests/e2e

build-docker:
	DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build . -t rclip

# CI runs release-brew as part of the `release` action
build-windows:
	uv run --exact --no-dev --with pyinstaller==6.10.0 pyinstaller -y ./release-utils/windows/pyinstaller.spec

# CI runs release-brew as part of the `release` action
release-brew:
	uv run ./release-utils/homebrew/release.sh

release:
	@test $(VERSION) || (echo "VERSION arg is required (e.g. VERSION=major|minor|patch|alpha|beta|rc)" && exit 1)
	uv version --no-sync --bump $(VERSION)
	$(SED) -i "s/version: .*/version: $$(uv version --short)/" snap/snapcraft.yaml
	git commit -am "release: v$$(uv version --short)"
	git push origin $$(git branch --show-current)
	git tag v$$(uv version --short)
	git push origin v$$(uv version --short)
