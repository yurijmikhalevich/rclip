import os
from pathlib import Path


def pytest_configure() -> None:
  local_models = Path(__file__).resolve().parent.parent / "local-models"
  if local_models.is_dir() and "RCLIP_DATADIR" not in os.environ:
    os.environ["RCLIP_DATADIR"] = str(local_models)
