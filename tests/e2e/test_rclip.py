from pathlib import Path
import os
import sys
import tempfile

import pytest

from rclip.main import main


def set_argv(*args: str):
  script_name = sys.argv[0]
  sys.argv.clear()
  sys.argv.append(script_name)
  sys.argv.extend(args)


@pytest.fixture
def test_images_dir():
  return Path(__file__).parent / 'images'


@pytest.fixture
def test_empty_dir():
  return Path(__file__).parent / 'empty_directory'


@pytest.fixture
def test_dir_with_nested_directories():
  return Path(__file__).parent / 'images nested directories'


def _assert_output_snapshot(images_dir: Path, request: pytest.FixtureRequest, capsys: pytest.CaptureFixture[str]):
  out, _ = capsys.readouterr()
  snapshot_path = Path(__file__).parent / 'output_snapshots' / f'{request.node.name}.txt'
  snapshot = out.replace(str(images_dir) + os.path.sep, '<test_images_dir>').replace(os.path.sep, '/')
  if not snapshot_path.exists():
    snapshot_path.write_text(snapshot)
  assert snapshot == snapshot_path.read_text()


@pytest.fixture
def assert_output_snapshot(test_images_dir: Path, request: pytest.FixtureRequest, capsys: pytest.CaptureFixture[str]):
  yield
  _assert_output_snapshot(test_images_dir, request, capsys)


@pytest.fixture
def assert_output_snapshot_nested_directories(
  test_dir_with_nested_directories: Path,
  request: pytest.FixtureRequest,
  capsys: pytest.CaptureFixture[str],
):
  yield
  _assert_output_snapshot(test_dir_with_nested_directories, request, capsys)


def execute_query(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch, *args: str):
  with tempfile.TemporaryDirectory() as tmpdirname:
    monkeypatch.setenv('RCLIP_DATADIR', tmpdirname)
    monkeypatch.chdir(test_images_dir)
    set_argv(*args)
    main()


@pytest.mark.usefixtures('assert_output_snapshot')
def test_search(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, 'kitty')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_search_webp(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  # this test result snapshot should contain a webp image
  execute_query(test_images_dir, monkeypatch, 'tree')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_search_png(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  # this test result snapshot should contain a png image
  execute_query(test_images_dir, monkeypatch, 'boats on a lake')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_repeated_searches_should_be_the_same(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, 'boats on a lake')
  execute_query(test_images_dir, monkeypatch, 'boats on a lake')
  execute_query(test_images_dir, monkeypatch, 'boats on a lake')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_search_by_image(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, str(test_images_dir / 'cat.jpg'))


@pytest.mark.usefixtures('assert_output_snapshot')
def test_search_by_image_from_url(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(
    test_images_dir,
    monkeypatch,
    'https://raw.githubusercontent.com/yurijmikhalevich/rclip/main/tests/e2e/images/cat.jpg'
  )


@pytest.mark.usefixtures('assert_output_snapshot')
def test_search_by_non_existing_file(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  with pytest.raises(SystemExit):
    execute_query(test_images_dir, monkeypatch, './non-existing-file.jpg')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_search_by_not_an_image(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  with pytest.raises(SystemExit):
    execute_query(test_images_dir, monkeypatch, str(test_images_dir / 'not-an-image.txt'))


@pytest.mark.usefixtures('assert_output_snapshot')
def test_add_queries(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, 'kitty', '--add', 'puppy', '-a', 'roof', '+', 'fence')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_subtract_queries(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, 'kitty', '--subtract', 'puppy', '-s', 'roof', '-', 'fence')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_add_and_subtract_queries(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, 'kitty', '+', 'roof', '-', 'fence')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_query_multipliers(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, 'kitty', '+', '2:night', '-', '0.5:fence')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_combine_text_query_with_image_query(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, str(test_images_dir / 'cat.jpg'), '-', '3:cat', '+', '2:bee')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_combine_image_query_with_text_query(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_images_dir, monkeypatch, 'kitty', '-', str(test_images_dir / 'cat.jpg'), '+', '1.5:bee')


@pytest.mark.usefixtures('assert_output_snapshot')
def test_seach_empty_dir(test_empty_dir: Path, monkeypatch: pytest.MonkeyPatch):
  execute_query(test_empty_dir, monkeypatch, 'kitty')


@pytest.mark.usefixtures('assert_output_snapshot_nested_directories')
def test_seach_dir_with_multiple_nested_directories(
  test_dir_with_nested_directories: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  execute_query(test_dir_with_nested_directories, monkeypatch, 'kitty')


@pytest.mark.usefixtures('assert_output_snapshot_nested_directories')
def test_seach_dir_with_deeply_nested_directories(
  test_dir_with_nested_directories: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  # output should contain a nested path to the bee image
  execute_query(test_dir_with_nested_directories, monkeypatch, 'bee')


@pytest.mark.usefixtures('assert_output_snapshot_nested_directories')
def test_handles_addition_and_deletion_of_images(
  test_dir_with_nested_directories: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  execute_query(test_dir_with_nested_directories, monkeypatch, 'bee')

  bee_image_path = test_dir_with_nested_directories / 'misc' / 'bees' / 'bee.jpg'
  assert bee_image_path.exists()

  bee_image_path_copy = bee_image_path.with_name('bee_copy.jpg')
  try:
    # copy bee image
    bee_image_path_copy.write_bytes(bee_image_path.read_bytes())

    # should include bee image copy in the output snapshot
    execute_query(test_dir_with_nested_directories, monkeypatch, 'bee')

    # delete bee image copy
    bee_image_path_copy.unlink()

    # should not include bee image copy in the output snapshot
    execute_query(test_dir_with_nested_directories, monkeypatch, 'bee')

  finally:
    bee_image_path_copy.unlink(missing_ok=True)
