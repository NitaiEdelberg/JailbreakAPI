"""Unit tests for the rate limiter (deterministic via an injectable clock)."""
from rate_limit import RateLimiter


def test_allows_up_to_limit_then_blocks():
    rl = RateLimiter(max_per_min=3, window=60, clock=lambda: 1000)
    assert all(rl.allow("ip") for _ in range(3))
    assert rl.allow("ip") is False  # 4th within the window is blocked


def test_limits_are_per_key():
    rl = RateLimiter(max_per_min=2, window=60, clock=lambda: 1000)
    assert rl.allow("a") and rl.allow("a")
    assert rl.allow("a") is False
    assert rl.allow("b") is True  # a different IP has its own budget


def test_window_expiry_frees_budget():
    now = {"t": 1000}
    rl = RateLimiter(max_per_min=2, window=60, clock=lambda: now["t"])
    assert rl.allow("ip") and rl.allow("ip")
    assert rl.allow("ip") is False
    now["t"] = 1061  # past the window
    assert rl.allow("ip") is True


def test_retry_after_is_positive_when_blocked():
    now = {"t": 1000}
    rl = RateLimiter(max_per_min=1, window=60, clock=lambda: now["t"])
    assert rl.allow("ip") is True
    assert rl.allow("ip") is False
    assert rl.retry_after("ip") >= 1
