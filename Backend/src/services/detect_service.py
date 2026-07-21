from datetime import datetime
from models.message_model import Message
from db import collection
from scanners import scanners
import logging


def process_message(message: Message):
    """Run every scanner and return a full breakdown.

    Design note: earlier this endpoint short-circuited on the first hit and raised
    a 403. It now runs all scanners and always returns 200 with a rich body — the
    caller gets each scanner's score plus an aggregate verdict, which is far more
    useful for tuning and for the UI. Legacy fields (`detected`, `flagged_by`) are
    kept so existing clients keep working.
    """
    results = []
    for scanner in scanners:
        name = getattr(scanner, "name", scanner.__class__.__name__)
        # One misbehaving scanner shouldn't take down the whole request; log and skip it.
        try:
            verdict = scanner.scan(message.text)
        except Exception as e:
            logging.warning(f"{name} failed, skipping: {e}")
            continue
        results.append({
            "scanner": name,
            "flagged": bool(verdict["flagged"]),
            "risk_score": round(float(verdict["risk"]), 2),
            "matched": verdict.get("matched"),
        })

    flagged_scanners = [r for r in results if r["flagged"]]
    detected = len(flagged_scanners) > 0
    # Aggregate risk = the most confident scanner (defaults to 0 if none ran).
    risk_score = max((r["risk_score"] for r in results), default=0.0)

    # Logging to MongoDB is best-effort: no DB (collection is None) or a DB
    # outage must not break detection.
    if collection is not None:
        try:
            collection.insert_one({
                "text": message.text,
                "detected": detected,
                "flagged_by": [r["scanner"] for r in flagged_scanners],
                "risk_score": risk_score,
                "timestamp": datetime.now(),
            })
        except Exception as e:
            logging.warning(f"Could not log attempt to MongoDB: {e}")

    if detected:
        logging.warning(f"Jailbreak detected: {message.text!r} | by: {[r['scanner'] for r in flagged_scanners]}")

    return {
        "detected": detected,                 # legacy field (kept for old clients)
        "verdict": "malicious" if detected else "safe",
        "risk_score": risk_score,
        "scanners": results,                  # every scanner's verdict + score
        # legacy shape: list of {scanner, risk_score} for the flagged ones
        "flagged_by": [
            {"scanner": r["scanner"], "risk_score": r["risk_score"]}
            for r in flagged_scanners
        ],
    }
