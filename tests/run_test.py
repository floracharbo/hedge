"""Quick pre-run tests."""

import numpy as np
from unittest import mock
import pytest

from importer import get_days_clnr
from filling_in import _number_missing_points

def test_number_missing_points():
    """Check that the _number_missing_points function does not add unnecessary 0 at the start."""
    dt = 24 * 60 / 48
    t = 0
    mins = [0.0, 30.0, 60.0]
    n_miss, fill_t = _number_missing_points(dt, t, mins)
    assert len(fill_t) == 0 or not (fill_t[0] == 0 and day["mins"][
        0] == 0), f"why add zero if already 0? day['mins'] {day['mins']} t {t} fill_t {fill_t}"
