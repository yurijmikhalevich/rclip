# rclip - AI-Powered Photo Search CLI Tool

<div align="center">
  <img alt="rclip logo" src="logo_transparent.png" width="600px" />
</div>

**rclip** is a command-line photo search tool based on the awesome OpenAI's [CLIP](https://github.com/openai/CLIP) neural network.

## Installation

Currently, pre-built distributable is available only for Linux x86_64.

1. Download the AppImage from the latest [release](https://github.com/yurijmikhalevich/rclip/releases).

2. Execute following commands:

```bash
$ chmod +x <downloaded AppImage filename>
$ sudo mv <downloaded AppImage filename> /usr/local/bin/rclip
```

## Usage

```bash
$ cd photos && rclip "search query"
```

### How do I preview the results?

The command from below will open top-5 results for "kitty" in your default image viewer. For this to work, you'll have to index the directory beforehand by running rclip in it without the `-n` key.

```bash
$ rclip -nf -t 5 "kitty" | xargs -d '\n' -n 1 xdg-open
```

## Help

```bash
$ rclip --help
```

## Contributing

This repository follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard.

Please, execute `pipenv shell` before running `pipenv sync` or `pipenv install` to set `PIP_FIND_LINKS`.

## License

MIT
