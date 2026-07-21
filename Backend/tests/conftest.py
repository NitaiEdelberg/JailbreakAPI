"""Test config: put the app's `src/` on sys.path so tests import the same way
the app does (top-level `routes`, `services`, `scanners`, ...), and point the
model loader at the real pickle regardless of the working directory.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
