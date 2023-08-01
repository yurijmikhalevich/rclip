# rclip - AI-Powered Command-Line Photo Search Tool
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[[Blog]](https://mikhalevi.ch/rclip-an-ai-powered-command-line-photo-search-tool/) [[Demo on YouTube]](https://www.youtube.com/watch?v=tAJHXOkHidw) [[Paper]](https://www.thinkmind.org/index.php?view=article&articleid=content_2023_1_20_60011)

<div align="center">
  <img alt="rclip logo" src="https://raw.githubusercontent.com/yurijmikhalevich/rclip/main/resources/logo-transparent.png" width="600px" />
</div>

**rclip** is a command-line photo search tool based on the awesome OpenAI's [CLIP](https://github.com/openai/CLIP) neural network.

## Installation

<details>
  <summary><strong>Linux x86_64</strong></summary>
  
  On Linux x86_64, you can install **rclip** as a self-containing binary.

  1. Download the AppImage from the latest [release](https://github.com/yurijmikhalevich/rclip/releases).

  2. Execute the following commands:

  ```bash
  chmod +x <downloaded AppImage filename>
  sudo mv <downloaded AppImage filename> /usr/local/bin/rclip
  ```
</details>

<details>
  <summary><strong>macOS/Windows/Linux (using <code>pip</code>)</strong></summary>

  ```bash
  pip install rclip
  ```
</details>

## Usage

```bash
cd photos && rclip "search query"
```

<img alt="rclip usage demo" src="https://raw.githubusercontent.com/yurijmikhalevich/rclip/main/resources/rclip-usage.gif" width="640px" />

When you run **rclip** for the first time in a particular directory, it's going to extract features from the photos, and this takes time. How long it takes depends on your CPU and the number of pictures you will search through. It took about a day to process 73 thousand photos on my NAS, which runs an old-ish Intel Celeron J3455.

For a detailed demonstration, watch the video: https://www.youtube.com/watch?v=tAJHXOkHidw.

### Similar image search

You can use another image as a query by passing a file path or even an URL to the image file, and **rclip** will find the images most similar to the one you used as a query. If you are referencing a local image via a relative path, you **must** prefix it with `./`. For example:

```bash
cd photos && rclip ./cat.jpg

# or use URL
cd photos && rclip https://raw.githubusercontent.com/yurijmikhalevich/rclip/main/tests/e2e/images/cat.jpg
```

Check this video out for the image-to-image search demo: https://www.youtube.com/watch?v=1YQZKeCBxWM.

### Combining multiple queries

You can add and subtract image and text queries from each other; here are a few usage examples:

```bash
cd photos && rclip horse + stripes
cd photos && rclip apple - fruit
cd photos && rclip "./new york city.jpg" + night
cd photos && rclip "2:golden retriever" + "./swimming pool.jpg"
cd photos && rclip "./racing car.jpg" - "2:sports car" + "2:snow"
```

If you want to see how these queries perform when executed on the 1.28 million images ImageNet-1k dataset, check out the demo on YouTube: https://www.youtube.com/watch?v=MsTgYdOpgcQ.

### How do I preview the results?

The command from below will open top-5 results for "kitty" in your default image viewer:

```bash
rclip -f -t 5 kitty | xargs -d '\n' -n 1 xdg-open
```

I prefer to use `feh`'s thumbnail mode to preview multiple results:

```bash
rclip -f -t 5 kitty | feh -f - -t
```

## Help

```bash
rclip --help
```

## Contributing

This repository follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard.

### Running locally from the source code

To run **rclip** locally from the source code, you must have [Python](https://www.python.org/downloads/) and [Poetry](https://python-poetry.org/) installed.

Then do:
```bash
# clone the source code repository
git clone git@github.com:yurijmikhalevich/rclip.git

# install dependencies and rclip
cd rclip
poetry install

# activate the new poetry environment
poetry shell
```

If the poetry environment is active, you can use **rclip** locally, as described in the [Usage](#usage) section above.

## Contributors ✨

Thanks go to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/ramayer"><img src="https://avatars.githubusercontent.com/u/72320?v=4?s=100" width="100px;" alt=""/><br /><sub><b>ramayer</b></sub></a><br /><a href="https://github.com/yurijmikhalevich/rclip/commits?author=ramayer" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

## License

MIT
