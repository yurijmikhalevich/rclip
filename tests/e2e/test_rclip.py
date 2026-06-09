from pathlib import Path
from typing import NamedTuple
from io import BytesIO
from io import TextIOWrapper
import os
import subprocess
import sys
import tempfile

import pytest


class SearchResult(NamedTuple):
  filepath: str
  score: float


def set_argv(*args: str):
  script_name = sys.argv[0]
  sys.argv.clear()
  sys.argv.append(script_name)
  sys.argv.extend(args)


@pytest.fixture
def test_images_dir():
  return Path(__file__).parent / "images"


@pytest.fixture
def test_empty_dir():
  return Path(__file__).parent / "empty_directory"


@pytest.fixture
def test_dir_with_nested_directories():
  return Path(__file__).parent / "images nested directories"


@pytest.fixture
def test_dir_with_raw_images():
  return Path(__file__).parent / "images raw"


@pytest.fixture
def test_dir_with_unicode_filenames():
  return Path(__file__).parent / "images unicode"


@pytest.fixture(scope="session")
def shared_datadir():
  with tempfile.TemporaryDirectory() as tmpdirname:
    yield tmpdirname


@pytest.fixture(scope="session")
def shared_model_cache_dir():
  with tempfile.TemporaryDirectory() as tmpdirname:
    yield tmpdirname


@pytest.fixture(autouse=True)
def use_shared_datadir(monkeypatch: pytest.MonkeyPatch, shared_datadir: str):
  Path(shared_datadir, "db.sqlite3").unlink(missing_ok=True)
  monkeypatch.setenv("RCLIP_DATADIR", shared_datadir)


def _assert_output_snapshot(
  images_dir: Path, request: pytest.FixtureRequest, capfd: pytest.CaptureFixture[str], encoding: str | None = None
):
  out, _ = capfd.readouterr()
  snapshot_path = Path(__file__).parent / "output_snapshots" / f"{request.node.name}.txt"
  snapshot = (
    out.replace(str(images_dir) + os.path.sep, "<test_images_dir>")
    .replace("./", "<test_images_dir>")
    .replace("." + os.path.sep, "<test_images_dir>")
    .replace(os.path.sep, "/")
    .replace("\r\n", "\n")
    # Stripping the BOM marker we are adding on Windows systems when the output is being piped to a file.
    # Otherwise, the output won't be encoded correctly.
  ).lstrip("\ufeff")
  if not snapshot_path.exists():
    snapshot_path.write_text(snapshot)
  assert snapshot == snapshot_path.read_text(encoding=encoding)


@pytest.fixture
def assert_output_snapshot(test_images_dir: Path, request: pytest.FixtureRequest, capfd: pytest.CaptureFixture[str]):
  yield
  _assert_output_snapshot(test_images_dir, request, capfd)


@pytest.fixture
def assert_output_snapshot_nested_directories(
  test_dir_with_nested_directories: Path,
  request: pytest.FixtureRequest,
  capfd: pytest.CaptureFixture[str],
):
  yield
  _assert_output_snapshot(test_dir_with_nested_directories, request, capfd)


@pytest.fixture
def assert_output_snapshot_raw_images(
  test_dir_with_raw_images: Path,
  request: pytest.FixtureRequest,
  capfd: pytest.CaptureFixture[str],
):
  yield
  _assert_output_snapshot(test_dir_with_raw_images, request, capfd)


@pytest.fixture
def assert_output_snapshot_unicode_filepaths(
  test_dir_with_unicode_filenames: Path, request: pytest.FixtureRequest, capfd: pytest.CaptureFixture[str]
):
  yield
  _assert_output_snapshot(test_dir_with_unicode_filenames, request, capfd, "utf-8-sig")


def execute_query(
  test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str, *args: str
) -> list[SearchResult]:
  from io import StringIO

  run_system_rclip = os.getenv("RCLIP_TEST_RUN_SYSTEM_RCLIP")
  if run_system_rclip:
    completed_run = subprocess.run(
      ["rclip", *args],
      cwd=test_images_dir,
      env={
        **os.environ,
        "RCLIP_MODEL_CACHE_DIR": shared_model_cache_dir,
        "RCLIP_TEST_RUN_SYSTEM_RCLIP": "",
      },
      capture_output=True,
      text=True,
    )
    output = completed_run.stdout
    sys.stdout.write(output)
    sys.stdout.flush()
    if completed_run.returncode != 0:
      raise SystemExit(completed_run.returncode)
  else:
    monkeypatch.setenv("RCLIP_MODEL_CACHE_DIR", shared_model_cache_dir)

    from rclip.main import main

    monkeypatch.chdir(test_images_dir)
    set_argv(*args)

    old_stdout = sys.stdout
    captured_buffer = BytesIO()
    sys.stdout = captured = TextIOWrapper(captured_buffer, encoding="utf-8")
    try:
      main()
    finally:
      captured.flush()
      output = captured_buffer.getvalue().decode("utf-8")
      sys.stdout = old_stdout
      sys.stdout.write(output)
      sys.stdout.flush()

  results = []
  for line in output.strip().split("\n")[1:]:
    if line.strip():
      parts = line.split("\t")
      if len(parts) >= 2:
        score = float(parts[0])
        filepath = parts[1].strip('"')
        results.append(SearchResult(filepath=filepath, score=score))

  return sorted(results, key=lambda r: (-r.score, r.filepath))


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "kitty")


@pytest.mark.usefixtures("assert_output_snapshot")
@pytest.mark.parametrize(
  "query,expected_ext",
  [
    ("tree", "webp"),
    ("boats on a lake", "png"),
    ("bee", "heic"),
    ("chess knight", "tiff"),
    ("chess pawns", "bmp"),
    ("chess queen", "gif"),
    ("duck", "jp2"),
    ("kookaburras", "pnm"),
    ("magpie", "pbm"),
    ("parrot and flowers", "pgm"),
    ("parrot looks into the camera", "ppm"),
  ],
  ids=["webp", "png", "heic", "tiff", "bmp", "gif", "jp2", "pnm", "pbm", "pgm", "ppm"],
)
def test_search_format(
  test_images_dir: Path,
  monkeypatch: pytest.MonkeyPatch,
  shared_model_cache_dir: str,
  query: str,
  expected_ext: str,
):
  results = execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, query)
  assert any(expected_ext in result.filepath for result in results)


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_animated_gif(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  results = execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "bee animated", "--top", "15")
  assert any("bee_animated.gif" in result.filepath for result in results)


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_by_image(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, str(test_images_dir / "cat.jpg"))


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_by_image_from_url(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  execute_query(
    test_images_dir,
    monkeypatch,
    shared_model_cache_dir,
    "https://raw.githubusercontent.com/yurijmikhalevich/rclip/5630d6279ee94f0cad823777433d7fbeb921d19e/tests/e2e/images/cat.jpg",  # noqa
  )


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_by_non_existing_file(
  test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str
):
  with pytest.raises(SystemExit):
    execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "./non-existing-file.jpg")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_by_not_an_image(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  with pytest.raises(SystemExit):
    execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, str(test_images_dir / "not-an-image.txt"))


@pytest.mark.usefixtures("assert_output_snapshot")
def test_add_queries(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  execute_query(
    test_images_dir, monkeypatch, shared_model_cache_dir, "kitty", "--add", "puppy", "-a", "roof", "+", "fence"
  )


@pytest.mark.usefixtures("assert_output_snapshot")
def test_subtract_queries(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  execute_query(
    test_images_dir, monkeypatch, shared_model_cache_dir, "kitty", "--subtract", "puppy", "-s", "roof", "-", "fence"
  )


@pytest.mark.usefixtures("assert_output_snapshot")
def test_add_and_subtract_queries(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "kitty", "+", "roof", "-", "fence")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_query_multipliers(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "kitty", "+", "2:night", "-", "0.5:fence")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_combine_text_query_with_image_query(
  test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str
):
  execute_query(
    test_images_dir, monkeypatch, shared_model_cache_dir, str(test_images_dir / "cat.jpg"), "-", "3:cat", "+", "2:bee"
  )


@pytest.mark.usefixtures("assert_output_snapshot")
def test_combine_image_query_with_text_query(
  test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str
):
  execute_query(
    test_images_dir, monkeypatch, shared_model_cache_dir, "kitty", "-", str(test_images_dir / "cat.jpg"), "+", "1.5:bee"
  )


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_empty_dir(test_empty_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  execute_query(test_empty_dir, monkeypatch, shared_model_cache_dir, "kitty")


@pytest.mark.usefixtures("assert_output_snapshot_nested_directories")
def test_search_dir_with_multiple_nested_directories(
  test_dir_with_nested_directories: Path,
  monkeypatch: pytest.MonkeyPatch,
  shared_model_cache_dir: str,
):
  execute_query(test_dir_with_nested_directories, monkeypatch, shared_model_cache_dir, "kitty")


@pytest.mark.usefixtures("assert_output_snapshot_nested_directories")
def test_search_dir_with_deeply_nested_directories(
  test_dir_with_nested_directories: Path,
  monkeypatch: pytest.MonkeyPatch,
  shared_model_cache_dir: str,
):
  # output should contain a nested path to the bee image
  execute_query(test_dir_with_nested_directories, monkeypatch, shared_model_cache_dir, "bee")


@pytest.mark.usefixtures("assert_output_snapshot_nested_directories")
def test_handles_addition_and_deletion_of_images(
  test_dir_with_nested_directories: Path,
  monkeypatch: pytest.MonkeyPatch,
  shared_model_cache_dir: str,
):
  execute_query(test_dir_with_nested_directories, monkeypatch, shared_model_cache_dir, "bee")

  bee_image_path = test_dir_with_nested_directories / "misc" / "bees" / "bee.jpg"
  assert bee_image_path.exists()

  bee_image_path_copy = bee_image_path.with_name("bee_copy.jpg")
  try:
    # copy bee image
    bee_image_path_copy.write_bytes(bee_image_path.read_bytes())

    # should include bee image copy in the output snapshot
    execute_query(test_dir_with_nested_directories, monkeypatch, shared_model_cache_dir, "bee")

    # delete bee image copy
    bee_image_path_copy.unlink()

    # should not include bee image copy in the output snapshot
    execute_query(test_dir_with_nested_directories, monkeypatch, shared_model_cache_dir, "bee")

  finally:
    bee_image_path_copy.unlink(missing_ok=True)


@pytest.mark.usefixtures("assert_output_snapshot_raw_images")
def test_ignores_raw_files_if_raw_support_is_disabled(
  test_dir_with_raw_images: Path,
  monkeypatch: pytest.MonkeyPatch,
  shared_model_cache_dir: str,
):
  # output should not contain any raw images
  execute_query(test_dir_with_raw_images, monkeypatch, shared_model_cache_dir, "boat on a lake")


@pytest.mark.usefixtures("assert_output_snapshot_raw_images")
def test_ignores_raw_if_there_is_a_png_named_the_same_way_in_the_same_dir(
  test_dir_with_raw_images: Path,
  monkeypatch: pytest.MonkeyPatch,
  shared_model_cache_dir: str,
):
  # output should not contain "boat on a lake.ARW" image
  execute_query(
    test_dir_with_raw_images, monkeypatch, shared_model_cache_dir, "--experimental-raw-support", "boat on a lake"
  )


@pytest.mark.usefixtures("assert_output_snapshot_raw_images")
def test_can_read_arw_images(
  test_dir_with_raw_images: Path,
  monkeypatch: pytest.MonkeyPatch,
  shared_model_cache_dir: str,
):
  results = execute_query(
    test_dir_with_raw_images, monkeypatch, shared_model_cache_dir, "--experimental-raw-support", "green ears of rye"
  )
  assert results[0].filepath.endswith("DSC08882.ARW")


@pytest.mark.usefixtures("assert_output_snapshot_raw_images")
def test_can_read_cr2_images(
  test_dir_with_raw_images: Path,
  monkeypatch: pytest.MonkeyPatch,
  shared_model_cache_dir: str,
):
  results = execute_query(
    test_dir_with_raw_images, monkeypatch, shared_model_cache_dir, "--experimental-raw-support", "dragon in a cave"
  )
  assert results[0].filepath.endswith("RAW_CANON_400D_ARGB.CR2")


@pytest.mark.usefixtures("assert_output_snapshot_raw_images")
def test_can_read_dng_images(
  test_dir_with_raw_images: Path,
  monkeypatch: pytest.MonkeyPatch,
  shared_model_cache_dir: str,
):
  results = execute_query(
    test_dir_with_raw_images,
    monkeypatch,
    shared_model_cache_dir,
    "--experimental-raw-support",
    "two parrots on a tree branch",
  )
  assert results[0].filepath.endswith("DSC03671.dng")


@pytest.mark.usefixtures("assert_output_snapshot_unicode_filepaths")
def test_unicode_filepaths(
  test_dir_with_unicode_filenames: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str
):
  execute_query(test_dir_with_unicode_filenames, monkeypatch, shared_model_cache_dir, "鳥")
