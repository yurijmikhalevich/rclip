# rclip - AI-powered image search tool

**rclip** is a command-line image search tool based on the awesome OpenAI's [CLIP](https://github.com/openai/CLIP) neural network.

## Installation

Currently, pre-built distributable is available only for Linux x86_64.

```bash
$ wget -c https://github.com/yurijmikhalevich/rclip/releases/download/v0.0.1-alpha/rclip-0.0.1-alpha-x86_64.AppImage
$ chmod +x rclip-0.0.1-alpha-x86_64.AppImage
$ sudo mv rclip-0.0.1-alpha-x86_64.AppImage /usr/local/bin/rclip
```

## Usage

```bash
$ cd photos && rclip "search query"
```

## Help

```bash
$ rclip --help
```

## Contributing

This repository follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard.

## License

MIT
