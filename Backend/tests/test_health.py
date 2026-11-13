"""/health must tell the truth about which scanners are running."""
from fastapi.testclient import TestClient

import scanners
from server import app

client = TestClient(app)


def test_health_ok_when_all_core_scanners_loaded():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "RegexScanner" in body["scanners"]
    assert "CustomMLScanner" in body["scanners"]
    assert body["unavailable"] == {}


def test_health_reports_degraded_when_ml_scanner_missing(monkeypatch):
    monkeypatch.setitem(scanners.unavailable, "CustomMLScanner", "RuntimeError: boom")
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert "CustomMLScanner" in body["unavailable"]
