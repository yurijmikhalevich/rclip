from pathlib import Path
import os
import subprocess
import sys
import tempfile

import pytest


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
def shared_model_cache_dir():
  with tempfile.TemporaryDirectory() as tmpdirname:
    yield tmpdirname


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


def execute_query(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str, *args: str):
  with tempfile.TemporaryDirectory() as tmpdirname:
    run_system_rclip = os.getenv("RCLIP_TEST_RUN_SYSTEM_RCLIP")
    if run_system_rclip:
      completed_run = subprocess.run(
        ["rclip", *args],
        cwd=test_images_dir,
        env={
          **os.environ,
          "RCLIP_DATADIR": tmpdirname,
          "RCLIP_MODEL_CACHE_DIR": shared_model_cache_dir,
          "RCLIP_TEST_RUN_SYSTEM_RCLIP": "",
        },
      )
      if completed_run.returncode != 0:
        raise SystemExit(completed_run.returncode)
    else:
      monkeypatch.setenv("RCLIP_DATADIR", tmpdirname)
      monkeypatch.setenv("RCLIP_MODEL_CACHE_DIR", shared_model_cache_dir)

      from rclip.main import main

      monkeypatch.chdir(test_images_dir)
      set_argv(*args)
      main()


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "kitty")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_webp(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  # this test result snapshot should contain a webp image
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "tree")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_png(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  # this test result snapshot should contain a png image
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "boats on a lake")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_heic(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str):
  # this test result snapshot should contain a heic image
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "bee")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_repeated_searches_should_be_the_same(
  test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str
):
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "boats on a lake")
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "boats on a lake")
  execute_query(test_images_dir, monkeypatch, shared_model_cache_dir, "boats on a lake")


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
  # DSC08882.ARW should be at the top of the results
  execute_query(
    test_dir_with_raw_images, monkeypatch, shared_model_cache_dir, "--experimental-raw-support", "green ears of rye"
  )


@pytest.mark.usefixtures("assert_output_snapshot_raw_images")
def test_can_read_cr2_images(
  test_dir_with_raw_images: Path,
  monkeypatch: pytest.MonkeyPatch,
  shared_model_cache_dir: str,
):
  # RAW_CANON_400D_ARGB.CR2 should be at the top of the results
  execute_query(
    test_dir_with_raw_images, monkeypatch, shared_model_cache_dir, "--experimental-raw-support", "dragon in a cave"
  )


@pytest.mark.usefixtures("assert_output_snapshot_unicode_filepaths")
def test_unicode_filepaths(
  test_dir_with_unicode_filenames: Path, monkeypatch: pytest.MonkeyPatch, shared_model_cache_dir: str
):
  execute_query(test_dir_with_unicode_filenames, monkeypatch, shared_model_cache_dir, "鳥")
