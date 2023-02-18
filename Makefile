build-appimage:
	poetry run appimage-builder --recipe appimage-builder.yml

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
