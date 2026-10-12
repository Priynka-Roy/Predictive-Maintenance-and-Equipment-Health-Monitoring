"""Small deterministic telemetry fixtures; synthetic values are only test inputs."""

import numpy as np
import pandas as pd
import pytest

from industrialguard.data import ANALOG, DIGITAL


@pytest.fixture
def sensors():
    index = pd.date_range("2020-01-01", periods=40, freq="5min", name="timestamp")
    frame = pd.DataFrame(index=index)
    for offset, sensor in enumerate(ANALOG):
        frame[sensor] = np.arange(len(index), dtype=float) + offset
    for sensor in DIGITAL:
        frame[sensor] = np.arange(len(index)) % 2
    return frame
