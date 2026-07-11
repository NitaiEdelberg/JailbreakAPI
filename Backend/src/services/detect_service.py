from fastapi import HTTPException
from datetime import datetime
from models.message_model import Message
from db import collection
from scanners import scanners
import logging

def process_message(message: Message):
    print(f"Received message: {message.text}")
    flagged = []
    scanner_name = None

    for scanner in scanners:
        # One misbehaving scanner shouldn't take down the whole request; log and skip it.
        try:
            _, is_valid, risk = scanner.scan(message.text)
        except Exception as e:
            logging.warning(f"{scanner.__class__.__name__} failed, skipping: {e}")
            continue
        if not is_valid:
            scanner_name = scanner.__class__.__name__
            flagged.append({
                "scanner": scanner.__class__.__name__,
                "risk_score": risk
            })
            break #stop scanning if our scanner detects a risk
        

    # Logging to MongoDB is best-effort: no DB (collection is None) or a DB
    # outage must not break detection.
    if collection is not None:
        try:
            collection.insert_one({
                "text": message.text,
                "detected": bool(flagged),
                "flagged_by": scanner_name,
                "timestamp": datetime.now()
            })
        except Exception as e:
            logging.warning(f"Could not log attempt to MongoDB: {e}")

    if flagged:
        logging.warning(f"Jailbreak detected: {message.text} | Details: {flagged}")
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Jailbreak attempt detected",
                "flagged_by": flagged
            }
        )

    return {"message": "Safe input", "detected": False}
