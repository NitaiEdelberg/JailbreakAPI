"""Test config: put the app's `src/` on sys.path so tests import the same way
the app does (top-level `routes`, `services`, `scanners`, ...), and point the
model loader at the real pickle regardless of the working directory.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def pytest_sessionstart(session):
    # Most API tests assume the ML scanner is loaded. If the local environment
    # has the wrong scikit-learn, it silently falls back to regex-only and a
    # handful of tests fail with confusing "detected is False" assertions.
    # Stop up front with the real reason instead.
    from scanners import unavailable
    if unavailable:
        import pytest
        reasons = "; ".join(f"{k}: {v}" for k, v in unavailable.items())
        pytest.exit(
            f"ML scanner failed to load ({reasons}). Install the pinned versions: "
            "pip install -r requirements.txt -r requirements-dev.txt", returncode=1)
