from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from pymongo import MongoClient
from llm_guard.input_scanners import PromptInjection
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Connect to MongoDB
db_client = MongoClient("mongodb://localhost:27017/")
db = db_client["jailbreak_db"]
collection = db["jailbreak_attempts"]

# Define scanners from llm-guard
scanners = [PromptInjection()]


# Define the message model
class Message(BaseModel):
    text: str


@app.post("/detect")
def detect_jailbreak(message: Message):
    print(f"Received message: {message.text}")

    flagged = []

    for scanner in scanners:
        _, is_valid, risk = scanner.scan(message.text)
        if not is_valid:
            flagged.append({
                "scanner": scanner.__class__.__name__,
                "risk_score": risk  # add whatever info you like
            })

    # Save to DB
    log_entry = {
        "text": message.text,
        "detected": bool(flagged),
        "flagged_by": flagged,
        "timestamp": datetime.utcnow()
    }
    # collection.insert_one(log_entry)

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