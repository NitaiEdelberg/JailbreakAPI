from fastapi import HTTPException
from datetime import datetime
from src.models.message_model import Message
from src.db import collection
from src.scanners import scanners
import logging

def process_message(message: Message):
    print(f"Received message: {message.text}")
    flagged = []
    scanner_name = None

    for scanner in scanners:
        _, is_valid, risk = scanner.scan(message.text)
        if not is_valid:
            scanner_name = scanner.__class__.__name__
            flagged.append({
                "scanner": scanner.__class__.__name__,
                "risk_score": risk
            })
            break #stop scanning if our scanner detects a risk
        

    collection.insert_one({
        "text": message.text,
        "detected": bool(flagged),
        "flagged_by": scanner_name,
        "timestamp": datetime.now()
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
