import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rescuebot_core import RescueConfig  # noqa: E402


@pytest.fixture
def cfg():
    return RescueConfig()


@pytest.fixture
def model(cfg):
    return cfg.public_model()
