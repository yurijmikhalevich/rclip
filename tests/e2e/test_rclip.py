from pathlib import Path
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
def assert_output_snapshot(test_images_dir: Path, request: pytest.FixtureRequest, capsys: pytest.CaptureFixture[str]):
  yield
  out, _ = capsys.readouterr()
  snapshot_path = Path(__file__).parent / 'output_snapshots' / f'{request.node.name}.txt'
  snapshot = out.replace(str(test_images_dir), '<test_images_dir>')
  if not snapshot_path.exists():
    snapshot_path.write_text(snapshot)
  assert snapshot == snapshot_path.read_text()


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
