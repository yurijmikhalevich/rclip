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
  if not snapshot_path.exists():
    snapshot_path.write_text(out.replace(str(test_images_dir), ''))
  assert out.replace(str(test_images_dir), '') == snapshot_path.read_text()


@pytest.mark.usefixtures('assert_output_snapshot')
def test_search(test_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
  with tempfile.TemporaryDirectory() as tmpdirname:
    monkeypatch.setenv('RCLIP_DATADIR', tmpdirname)
    monkeypatch.chdir(test_images_dir)
    set_argv('kitty')
    main()
