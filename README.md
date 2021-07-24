# rclip - AI-powered photo search tool

**rclip** is a command-line photo search tool based on the awesome OpenAI's [CLIP](https://github.com/openai/CLIP) neural network.

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

## License

MIT
