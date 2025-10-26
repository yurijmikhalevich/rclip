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
    snapshot_path.write_text(snapshot, encoding="utf-8")
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


def execute_query(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, *args: str):
  with tempfile.TemporaryDirectory() as tmpdirname:
    run_system_rclip = os.getenv("RCLIP_TEST_RUN_SYSTEM_RCLIP")
    if run_system_rclip:
      completed_run = subprocess.run(
        ["rclip", *args],
        cwd=test_images_dir,
        env={**os.environ, "RCLIP_DATADIR": tmpdirname, "RCLIP_TEST_RUN_SYSTEM_RCLIP": ""},
      )
      if completed_run.returncode != 0:
        raise SystemExit(completed_run.returncode)
    else:
      from rclip.main import main

      monkeypatch.setenv("RCLIP_DATADIR", tmpdirname)
      monkeypatch.chdir(test_images_dir)
      set_argv(*args)
      main()


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, "kitty")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_webp(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  # this test result snapshot should contain a webp image
  execute_query(test_images_dir, monkeypatch, "tree")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_png(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  # this test result snapshot should contain a png image
  execute_query(test_images_dir, monkeypatch, "boats on a lake")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_heic(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  # this test result snapshot should contain a heic image
  execute_query(test_images_dir, monkeypatch, "bee")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_repeated_searches_should_be_the_same(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, "boats on a lake")
  execute_query(test_images_dir, monkeypatch, "boats on a lake")
  execute_query(test_images_dir, monkeypatch, "boats on a lake")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_by_image(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, str(test_images_dir / "cat.jpg"))


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_by_image_from_url(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(
    test_images_dir,
    monkeypatch,
    "https://raw.githubusercontent.com/yurijmikhalevich/rclip/5630d6279ee94f0cad823777433d7fbeb921d19e/tests/e2e/images/cat.jpg",  # noqa
  )


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_by_non_existing_file(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  with pytest.raises(SystemExit):
    execute_query(test_images_dir, monkeypatch, "./non-existing-file.jpg")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_by_not_an_image(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  with pytest.raises(SystemExit):
    execute_query(test_images_dir, monkeypatch, str(test_images_dir / "not-an-image.txt"))


@pytest.mark.usefixtures("assert_output_snapshot")
def test_add_queries(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, "kitty", "--add", "puppy", "-a", "roof", "+", "fence")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_subtract_queries(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, "kitty", "--subtract", "puppy", "-s", "roof", "-", "fence")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_add_and_subtract_queries(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, "kitty", "+", "roof", "-", "fence")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_query_multipliers(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, "kitty", "+", "2:night", "-", "0.5:fence")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_combine_text_query_with_image_query(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, str(test_images_dir / "cat.jpg"), "-", "3:cat", "+", "2:bee")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_combine_image_query_with_text_query(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, "kitty", "-", str(test_images_dir / "cat.jpg"), "+", "1.5:bee")


@pytest.mark.usefixtures("assert_output_snapshot")
def test_search_empty_dir(test_empty_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_empty_dir, monkeypatch, "kitty")


@pytest.mark.usefixtures("assert_output_snapshot_nested_directories")
def test_search_dir_with_multiple_nested_directories(
  test_dir_with_nested_directories: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  execute_query(test_dir_with_nested_directories, monkeypatch, "kitty")


@pytest.mark.usefixtures("assert_output_snapshot_nested_directories")
def test_search_dir_with_deeply_nested_directories(
  test_dir_with_nested_directories: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  # output should contain a nested path to the bee image
  execute_query(test_dir_with_nested_directories, monkeypatch, "bee")


@pytest.mark.usefixtures("assert_output_snapshot_nested_directories")
def test_handles_addition_and_deletion_of_images(
  test_dir_with_nested_directories: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  execute_query(test_dir_with_nested_directories, monkeypatch, "bee")

  bee_image_path = test_dir_with_nested_directories / "misc" / "bees" / "bee.jpg"
  assert bee_image_path.exists()

  bee_image_path_copy = bee_image_path.with_name("bee_copy.jpg")
  try:
    # copy bee image
    bee_image_path_copy.write_bytes(bee_image_path.read_bytes())

    # should include bee image copy in the output snapshot
    execute_query(test_dir_with_nested_directories, monkeypatch, "bee")

    # delete bee image copy
    bee_image_path_copy.unlink()

    # should not include bee image copy in the output snapshot
    execute_query(test_dir_with_nested_directories, monkeypatch, "bee")

  finally:
    bee_image_path_copy.unlink(missing_ok=True)


@pytest.mark.usefixtures("assert_output_snapshot_raw_images")
def test_ignores_raw_files_if_raw_support_is_disabled(
  test_dir_with_raw_images: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  # output should not contain any raw images
  execute_query(test_dir_with_raw_images, monkeypatch, "boat on a lake")


@pytest.mark.usefixtures("assert_output_snapshot_raw_images")
def test_ignores_raw_if_there_is_a_png_named_the_same_way_in_the_same_dir(
  test_dir_with_raw_images: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  # output should not contain "boat on a lake.ARW" image
  execute_query(test_dir_with_raw_images, monkeypatch, "--experimental-raw-support", "boat on a lake")


@pytest.mark.usefixtures("assert_output_snapshot_raw_images")
def test_can_read_arw_images(
  test_dir_with_raw_images: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  # DSC08882.ARW should be at the top of the results
  execute_query(test_dir_with_raw_images, monkeypatch, "--experimental-raw-support", "green ears of rye")


@pytest.mark.usefixtures("assert_output_snapshot_raw_images")
def test_can_read_cr2_images(
  test_dir_with_raw_images: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  # RAW_CANON_400D_ARGB.CR2 should be at the top of the results
  execute_query(test_dir_with_raw_images, monkeypatch, "--experimental-raw-support", "dragon in a cave")


@pytest.mark.usefixtures("assert_output_snapshot_unicode_filepaths")
def test_unicode_filepaths(test_dir_with_unicode_filenames: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_dir_with_unicode_filenames, monkeypatch, "鳥")


def test_handles_renamed_images(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  """Test that renamed images are detected and don't require re-indexing."""
  import shutil
  import tempfile
  from rclip.main import init_rclip

  # Create a temporary directory with a copy of the cat image
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    original_image = test_images_dir / "cat.jpg"
    temp_image = temp_path / "cat.jpg"

    # Copy the image to temp directory
    shutil.copy2(original_image, temp_image)

    # Index the image initially
    rclip, _, db = init_rclip(str(temp_path), 8, "cpu", None, False, False)

    # Get the initial image record
    initial_image = db.get_image(filepath=str(temp_image))
    assert initial_image is not None
    initial_hash = initial_image["hash"]

    # Close the initial database connection
    db.close()

    # Rename the image
    renamed_image = temp_path / "renamed_cat.jpg"
    temp_image.rename(renamed_image)

    # Re-index (should detect the rename)
    rclip, _, db = init_rclip(str(temp_path), 8, "cpu", None, False, False)

    # Check that the renamed image exists with the same hash
    renamed_db_image = db.get_image(filepath=str(renamed_image))
    assert renamed_db_image is not None
    assert renamed_db_image["hash"] == initial_hash

    # Check that the old filepath no longer exists in the database
    old_db_image = db.get_image(filepath=str(temp_image))
    assert old_db_image is None

    # Verify we can still search for the image
    results = rclip.search("cat", str(temp_path), top_k=10)
    assert len(results) == 1
    assert results[0].filepath == str(renamed_image)

    db.close()