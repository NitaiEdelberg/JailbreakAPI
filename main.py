from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
import logging
import uvicorn
from llm_guard.input_scanners import PromptInjection
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
print("MONGO_URI:", os.getenv("MONGO_URI"))

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Connect to MongoDB
db_client = MongoClient(os.getenv("MONGO_URI"))
db = db_client["jailbreak_db"]
collection = db["jailbreak_attempts"]

# Define scanners from llm-guard
scanners = [PromptInjection()]


# Define the message model
class Message(BaseModel):
    text: str

@app.get("/ping")
def ping():
    return {"message": "hey"}

@app.post("/detect")
def detect_jailbreak(message: Message):
    print(f"Received message: {message.text}")

    flagged = []

    for scanner in scanners:
        _, is_valid, risk = scanner.scan(message.text)
        if not is_valid:
            flagged.append({
                "scanner": scanner.__class__.__name__,
                "risk_score": risk
            })

    # Save to DB
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=False)