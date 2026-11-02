import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rac.config import load_config  # noqa: E402
from rac.control_law import HealthModel  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def model():
    from rac.control_law import DEFAULT_MODEL
    return HealthModel.load(DEFAULT_MODEL)
