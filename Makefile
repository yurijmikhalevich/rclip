# the `release` action uses `sed` expressions which are only compatible with the GNU sed
ifeq ($(shell which gsed),)
    SED := sed
else
    SED := gsed
endif

build-appimage:
	poetry run appimage-builder --recipe ./release-utils/appimage/appimage-builder.yml

lint-style:
	poetry run pycodestyle .

lint-types:
	poetry run pyright .

install-pyright:
	npm i -g pyright@1.1.185

lint: lint-style lint-types

test:
	poetry run pytest tests

build-docker:
	DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build . -t rclip

release:
	@test $(VERSION) || (echo "VERSION arg is required" && exit 1)
	poetry version $(VERSION)
	$(SED) -i "s/version: .*/version: $$(poetry version -s)/" snap/snapcraft.yaml
	$(SED) -i "s/source: .*/source: .\/snap\/local\/rclip-$$(poetry version -s).tar.gz/" snap/snapcraft.yaml
	git commit -am "release: v$$(poetry version -s)"
	git push origin $$(git branch --show-current)
	git tag v$$(poetry version -s)
	git push origin v$$(poetry version -s)
