from fastapi import APIRouter
from models.message_model import BatchRequest, Message
from services.detect_service import process_message

router = APIRouter()


@router.post("/detect")
def detect_jailbreak(message: Message):
    return process_message(message)


@router.post("/detect/batch")
def detect_batch(batch: BatchRequest):
    """Scan many pieces of content in one call (RAG chunks, tool outputs, the
    pages an agent opened). Each item is judged on its own; the summary says
    which ones the caller should drop or quarantine before they reach the model."""
    results = []
    for i, item in enumerate(batch.items):
        r = process_message(item)
        r["id"] = item.id if item.id is not None else str(i)
        results.append(r)
    flagged = [r["id"] for r in results if r["verdict"] == "malicious"]
    suspicious = [r["id"] for r in results if r["verdict"] == "suspicious"]
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "malicious": flagged,
            "suspicious": suspicious,
            "safe_count": len(results) - len(flagged) - len(suspicious),
        },
    }
