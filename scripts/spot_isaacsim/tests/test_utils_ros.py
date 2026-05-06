from scripts.spot_isaacsim.utils.ros import duration_to_s


class _Duration:
    def __init__(self, sec, nanosec):
        self.sec = sec
        self.nanosec = nanosec


def test_duration_to_s_whole():
    assert duration_to_s(_Duration(3, 0)) == 3.0


def test_duration_to_s_fractional():
    assert abs(duration_to_s(_Duration(1, 500_000_000)) - 1.5) < 1e-9


def test_duration_to_s_zero():
    assert duration_to_s(_Duration(0, 0)) == 0.0
