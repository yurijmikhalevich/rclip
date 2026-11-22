# the `release` action uses `sed` expressions which are only compatible with the GNU sed
ifeq ($(shell which gsed),)
    SED := sed
else
    SED := gsed
endif

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
	RCLIP_TEST_RUN_SYSTEM_RCLIP=true cargo test --release --locked --test e2e

build-docker:
	DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build . -t rclip

# CI runs release-brew as part of the `release` action
release-brew:
	./release-utils/homebrew/release.sh

release:
	@test $(VERSION) || (echo "VERSION arg is required (e.g. VERSION=5.0.0 or VERSION=5.0.0-rc.1)" && exit 1)
	$(SED) -i '0,/^version = .*/s//version = "$(VERSION)"/' Cargo.toml
	cargo update --offline --workspace
	$(SED) -i "s/version: .*/version: $(VERSION)/" snap/snapcraft.yaml
	$(SED) -i -e "s/^version: .*/version: $(VERSION)/" -e "s/^date-released: .*/date-released: $$(date +%F)/" CITATION.cff
	git commit -am "release: v$(VERSION)"
	git push origin $$(git branch --show-current)
	git tag v$(VERSION)
	git push origin v$(VERSION)
