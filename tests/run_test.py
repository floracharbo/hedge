"""Quick pre-run tests."""

from filling_in import _number_missing_points


def test_number_missing_points():
    """Check _number_missing_points does not add unnecessary 0 at the start."""
    step_len = 24 * 60 / 48
    t = 0
    mins = [0.0, 30.0, 60.0]
    n_miss, fill_t = _number_missing_points(step_len, t, mins)
    assert len(fill_t) == 0 or not fill_t[0] == 0, \
        f"why add zero if already 0? t {t} fill_t {fill_t}"
