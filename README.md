# rclip – semantic photo search for the command line
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[[Blog]](https://mikhalevi.ch/rclip-an-ai-powered-command-line-photo-search-tool/) [[Demo on YouTube]](https://www.youtube.com/watch?v=tAJHXOkHidw) [[Paper]](https://www.thinkmind.org/index.php?view=article&articleid=content_2023_1_20_60011)

<div align="center">
  <img alt="rclip logo" src="https://raw.githubusercontent.com/yurijmikhalevich/rclip/main/resources/logo-transparent.png" width="600px" />
</div>

**rclip** is a semantic photo search tool for the command line, powered by [OpenCLIP's top-performing ViT-B/32 AI model](https://github.com/mlfoundations/open_clip/blob/55794d65a14dfc547a9ed3514145dd68ccc939e9/README.md). Search a local photo library with natural-language queries, similar image search, or mixed text and image queries – entirely on your machine, with no cloud and no uploads. It builds on the CLIP architecture introduced by OpenAI.

## Features

- **Natural-language search** – find photos by describing them, e.g. `rclip "two parrots on a branch"`.
- **Reverse / image-to-image search** – search by an example image from a local path or a URL.
- **Combined & arithmetic queries** – mix and weight text and image queries, e.g. `rclip "2:golden retriever" + "./pool.jpg" - fruit`.
- **Local & private** – works fully offline; your photos never leave your computer.
- **Wide format support** – `jpg`, `png`, `webp`, `heic`, `tiff`, `gif`, and more, plus experimental RAW (`arw`, `cr2`, `dng`).
- **Fast incremental indexing** – only new and changed images are reprocessed on subsequent runs.
- **Terminal previews** – view images inline in iTerm2, Konsole, wezterm, Mintty, and mlterm.
- **Cross-platform** – Linux, macOS (Apple Silicon), and Windows.

## Installation

### Linux

```bash
sudo snap install rclip
```

<details>
  <summary>Alternative options (AppImage and <code>pip</code>)</summary>

  If your Linux distribution doesn't support [snap](https://snapcraft.io/rclip), you can use one of the alternative installation options:

  #### AppImage (self-contained x86_64 executable)

  On Linux x86_64, you can install **rclip** as a self-contained executable.

  1. Download the AppImage from the [latest release](https://github.com/yurijmikhalevich/rclip/releases/latest).

  2. Execute the following commands:

  ```bash
  chmod +x <downloaded AppImage filename>
  sudo mv <downloaded AppImage filename> /usr/local/bin/rclip
  ```

  #### Using <code>pip</code>

  ```bash
  pip install --extra-index-url https://download.pytorch.org/whl/cpu rclip
  ```
</details>

### macOS

```bash
brew install yurijmikhalevich/tap/rclip
```

**Note:** We only support Apple Silicon (arm64) on macOS.

<details>
  <summary>Alternative option (<code>pip</code>)</summary>

  ```bash
  pip install rclip
  ```
</details>

### Windows

1. Download the "*.msi" from the [latest release](https://github.com/yurijmikhalevich/rclip/releases/latest).
2. Install **rclip** by running the installer.

<details>
  <summary>Alternative option (<code>pip</code>)</summary>

  ```bash
  pip install rclip
  ```
</details>

## Usage

Search the current directory with a natural-language query:

```bash
cd photos && rclip "search query"
```

<details>
  <summary>Example output</summary>

  ```text
  score  filepath
  0.297  "/photos/sunrise-beach.jpg"
  0.286  "/photos/dawn-walk.png"
  0.274  "/photos/morning-hike.heic"
  ```
</details>

<img alt="rclip usage demo" src="https://raw.githubusercontent.com/yurijmikhalevich/rclip/main/resources/rclip-usage.gif" width="640px" />

The first time you run **rclip** in a directory, it extracts features from your images to build the search index. How long this takes depends on your CPU and the number of images you search. On my hardware, it took 15 hours to process 84,725 photos on a NAS with an old Intel Celeron J3455, 7 minutes to index 50,000 images on a MacBook with an M1 Max, and 3 hours to process 1.28 million images on the same MacBook.

For a detailed demonstration, watch the video: https://www.youtube.com/watch?v=tAJHXOkHidw.

### Similar image search (image-to-image search)

You can also use an image as the query by passing a file path or image URL. **rclip** will return the images most similar to that query image. If you use a relative path to a local image, you **must** prefix it with `./`. For example:

```bash
cd photos && rclip ./cat.jpg

# or use URL
cd photos && rclip https://raw.githubusercontent.com/yurijmikhalevich/rclip/main/tests/e2e/images/cat.jpg
```

Check this video out for the image-to-image search demo: https://www.youtube.com/watch?v=1YQZKeCBxWM.

### Combining multiple queries

You can combine and subtract image and text queries; here are a few examples:

```bash
cd photos && rclip horse + stripes
cd photos && rclip apple - fruit
cd photos && rclip "./new york city.jpg" + night
cd photos && rclip "2:golden retriever" + "./swimming pool.jpg"
cd photos && rclip "./racing car.jpg" - "2:sports car" + "2:snow"
```

If you want to see how these queries perform when executed on the 1.28 million images ImageNet-1k dataset, check out the demo on YouTube: https://www.youtube.com/watch?v=MsTgYdOpgcQ.

### Which formats does **rclip** support?

**rclip** always indexes the following image formats: `jpg`, `jpeg`, `png`, `webp`, `heic`, `tiff`, `tif`, `bmp`, `gif`, `jp2`, `pnm`, `pbm`, `pgm`, and `ppm`.

RAW formats (`arw`, `cr2`, and `dng`) are supported when you pass the `--experimental-raw-support` flag:

```bash
rclip --experimental-raw-support cat
```

When this flag is enabled, a RAW file is skipped if a processed image (e.g., a
JPEG) with the same name sits alongside it, so previews and exported variants
are indexed instead of the RAW original.

### How do I preview the results?

If you are using either [iTerm2](https://iterm2.com/), [Konsole](https://konsole.kde.org/) (version 22.04 and higher), [wezterm](https://wezfurlong.org/wezterm/), [Mintty](https://mintty.github.io/), or [mlterm](https://mlterm.sourceforge.net/), all you need to do is pass the `--preview` (or `-p`) flag to **rclip**:

```bash
rclip -p kitty
```

<details>
  <summary>Using a different terminal or viewer</summary>

  If you use another terminal or want to open the results in a viewer of your choice, you can pipe **rclip**'s output into it. For example, on Linux, the command below will open the top 5 results for "kitty" in your default image viewer:

  ```bash
  rclip -f -t 5 kitty | xargs -d '\n' -n 1 xdg-open
  ```

  The `-f` or `--filepath-only` flag makes **rclip** print only file paths, without scores or the header, which makes it ideal for use with a custom viewer as in the example.
  
  I prefer to use **feh**'s thumbnail mode to preview multiple results:

  ```bash
  rclip -f -t 5 kitty | feh -f - -t
  ```
</details>

### Can I use **rclip** to copy images matching a certain query?

Yes. You can pipe **rclip**'s output to another tool to copy matching images. For example, to copy the top 3 images matching "search query" to `/path/to/destination` on macOS, Linux, or WSL:

```sh
rclip -f -t 3 "search query" | xargs -I {} cp {} /path/to/destination
```

### How does **rclip** update the index?

When you run **rclip** in a directory that has already been processed, it indexes
only the new images added since the last run and removes deleted images from its
index. This makes consecutive runs much faster.

If you know no images have been added or deleted since the last run, you can use
the `--no-indexing` (or `-n`) flag to skip indexing entirely and speed up the
search even more.

```bash
rclip -n cat
```

## Get help

https://github.com/yurijmikhalevich/rclip/discussions/new/choose

## Contributing

This repository follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard.

### Running locally from the source code

To run **rclip** locally from source, you must have [Python](https://www.python.org/downloads/) and [uv](https://docs.astral.sh/uv/) installed.

Then run:
```bash
# clone the source code repository
git clone git@github.com:yurijmikhalevich/rclip.git

# install dependencies and rclip
cd rclip
uv sync
```

Then use `uv run rclip`, as described in the [Usage](#usage) section above.

## Contributors ✨

Thanks go to these wonderful people and organizations ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ramayer"><img src="https://avatars.githubusercontent.com/u/72320?v=4?s=100" width="100px;" alt="ramayer"/><br /><sub><b>ramayer</b></sub></a><br /><a href="https://github.com/yurijmikhalevich/rclip/commits?author=ramayer" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.caphyon.com"><img src="https://avatars.githubusercontent.com/u/15829334?v=4?s=100" width="100px;" alt="Caphyon"/><br /><sub><b>Caphyon</b></sub></a><br /><a href="#infra-Caphyon" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://abidkhan484.github.io"><img src="https://avatars.githubusercontent.com/u/15053047?v=4?s=100" width="100px;" alt="AbId KhAn"/><br /><sub><b>AbId KhAn</b></sub></a><br /><a href="https://github.com/yurijmikhalevich/rclip/commits?author=abidkhan484" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://cl4r1ty.dev"><img src="https://avatars.githubusercontent.com/u/136800640?v=4?s=100" width="100px;" alt="Ben"/><br /><sub><b>Ben</b></sub></a><br /><a href="https://github.com/yurijmikhalevich/rclip/commits?author=Cl4r1ty-1" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://techtracer.pages.dev"><img src="https://avatars.githubusercontent.com/u/48885301?v=4?s=100" width="100px;" alt="Tanmay Chaudhari"/><br /><sub><b>Tanmay Chaudhari</b></sub></a><br /><a href="https://github.com/yurijmikhalevich/rclip/commits?author=tanmayc07" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://leoauri.com"><img src="https://avatars.githubusercontent.com/u/10868855?v=4?s=100" width="100px;" alt="Leo Auri"/><br /><sub><b>Leo Auri</b></sub></a><br /><a href="https://github.com/yurijmikhalevich/rclip/commits?author=leoauri" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

Thanks to [Caphyon](https://github.com/Caphyon) and the Advanced Installer team for generously supplying the **rclip** project with the Professional Advanced Installer license for creating the Windows installer.

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

## License

MIT
