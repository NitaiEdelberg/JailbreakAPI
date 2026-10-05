"""The aggregation layer: how per-scanner verdicts become one answer.

Driven with stub scanners rather than real prompts, because the interesting
cases are about the *combination rule* — a real prompt that happens to land in
the "suspicious" band today would drift out of it on the next retrain and turn
this into a flaky test.
"""
import pytest

import services.detect_service as detect_service
from models.message_model import Message


class StubScanner:
    def __init__(self, name, flagged, risk, matched=None):
        self.name = name
        self._verdict = {"flagged": flagged, "risk": risk, "matched": matched}

    def scan(self, text):
        return dict(self._verdict)


class ExplodingScanner:
    name = "Exploding"

    def scan(self, text):
        raise RuntimeError("boom")


@pytest.fixture
def stub_scanners(monkeypatch):
    def _set(*scanners):
        monkeypatch.setattr(detect_service, "scanners", list(scanners))
        # no DB in tests
        monkeypatch.setattr(detect_service, "collection", None)
    return _set


def run(text="anything"):
    return detect_service.process_message(Message(text=text))


def test_no_flags_and_low_risk_is_safe(stub_scanners):
    stub_scanners(StubScanner("A", False, 0.05), StubScanner("B", False, 0.1))
    body = run()
    assert body["verdict"] == "safe"
    assert body["detected"] is False
    assert body["risk_score"] == 0.1


def test_no_flags_but_elevated_risk_is_suspicious(stub_scanners):
    """The band that stops a 40%-risk prompt being reported as a flat "safe"."""
    stub_scanners(StubScanner("A", False, 0.4))
    body = run()
    assert body["verdict"] == "suspicious"
    # `detected` stays False so "suspicious" can't silently become a block for
    # legacy clients that only read that field.
    assert body["detected"] is False
    assert body["flagged_by"] == []


def test_any_flag_makes_it_malicious(stub_scanners):
    stub_scanners(StubScanner("A", False, 0.02), StubScanner("B", True, 0.9, "matched bit"))
    body = run()
    assert body["verdict"] == "malicious"
    assert body["detected"] is True
    assert [f["scanner"] for f in body["flagged_by"]] == ["B"]


def test_aggregate_risk_is_the_max_not_a_sum(stub_scanners):
    """Scores are calibrated probabilities; adding them would break that meaning."""
    stub_scanners(StubScanner("A", False, 0.3), StubScanner("B", False, 0.3))
    assert run()["risk_score"] == 0.3


def test_suspicious_boundary_is_inclusive(stub_scanners):
    stub_scanners(StubScanner("A", False, detect_service.SUSPICIOUS_THRESHOLD))
    assert run()["verdict"] == "suspicious"
    stub_scanners(StubScanner("A", False, detect_service.SUSPICIOUS_THRESHOLD - 0.01))
    assert run()["verdict"] == "safe"


def test_one_broken_scanner_does_not_break_the_request(stub_scanners):
    stub_scanners(ExplodingScanner(), StubScanner("B", True, 0.8, "hit"))
    body = run()
    assert body["detected"] is True
    assert [s["scanner"] for s in body["scanners"]] == ["B"]


def test_every_scanner_failing_yields_a_safe_default(stub_scanners):
    stub_scanners(ExplodingScanner())
    body = run()
    assert body["scanners"] == []
    assert body["risk_score"] == 0.0
    assert body["verdict"] == "safe"
