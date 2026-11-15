"""Run the SDK against the real API in-process: the transport hands requests to
FastAPI's TestClient instead of the network, so these are end-to-end tests of
real verdicts without a server, a port or a key."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                                  # sdk/
sys.path.insert(0, os.path.join(HERE, "..", "..", "Backend", "src"))       # the API
os.environ["JUDGE_ENABLED"] = "0"
os.environ["RATE_LIMIT_PER_MIN"] = "100000"


@pytest.fixture(scope="session")
def api():
    from fastapi.testclient import TestClient
    from server import app
    return TestClient(app)


@pytest.fixture
def transport(api):
    calls = []

    def send(method, url, body, timeout):
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        calls.append((method, "/" + path, body))
        r = api.request(method, "/" + path, json=body)
        return r.status_code, r.json(), dict(r.headers)

    send.calls = calls
    return send
