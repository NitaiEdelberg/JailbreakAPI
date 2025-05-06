from fastapi import HTTPException
from datetime import datetime
from models.message_model import Message
from db import collection
from scanners import scanners
import logging

def process_message(message: Message):
    print(f"Received message: {message.text}")
    flagged = []

    for scanner in scanners:
        _, is_valid, risk = scanner.scan(message.text)
        if not is_valid:
            flagged.append({
                "scanner": scanner.__class__.__name__,
                "risk_score": risk
            })

    collection.insert_one({
        "text": message.text,
        "detected": bool(flagged),
        "flagged_by": flagged,
        "timestamp": datetime.utcnow()
    })

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
